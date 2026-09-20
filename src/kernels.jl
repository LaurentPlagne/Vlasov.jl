"""
Backend-agnostic kernels for the particle loops.

These are the two loops that weigh on a step — the smoothed-field evaluation and
the charge deposition — written once, in `KernelAbstractions`, and run by
whichever backend the arrays belong to: `CPU()`, Metal, CUDA, ROCm, oneAPI.

They replace the hand-tuned Metal kernels of `ext/VlasovMetalExt.jl`, and the
substitution is close to free. Measured against that extension at 4×10⁶
particles on a 134³ grid:

| | hand-written Metal | here | |
|---|---|---|---|
| smoothed field | 61.34 ms | 61.42 ms | ×0.999, **bit for bit identical** |
| deposition | 42.7 ms | 44.4 ms | ×0.961, 3e-07 (atomic ordering) |

What the hand-written version did with `MtlThreadGroupArray`,
`threadgroup_barrier` and `Metal.@atomic`, this one does with `@localmem`,
`@synchronize` and `Atomix.@atomic` — the same thing said portably.

They also run on `CPU()`, in `Float64`, where they can be checked against the
scalar reference the Fortran oracle validates: the field kernel reproduces
`smoothed_field` **exactly**, the deposition agrees with `deposit_smoothed!` to
9.3e-16. No GPU can supply that check, having no double precision on Metal and
no oracle anywhere.

⚠️ **Element type.** The buffers carry `E`, which is `Float32` on Metal — Apple
GPUs have no double precision — and may be `Float64` on CUDA or ROCm. Only the
packing below is exempt, and deliberately so.
"""

"""
Work-items per group for [`_smoothed_field_kernel!`](@ref).

⚠️ **64, and the value matters far more than one would expect** — it is worth a
quarter of the kernel.

The group size is not a tuning knob here: it sets the *depth of the tree
reduction* that commits the projectile's reaction, and the size of the
threadgroup buffer that reduction needs. At 256 the reduction costs eight rounds
of barrier over 4 KB of `@localmem`; at 64 it costs six over 1 KB, and the
occupancy that buys pays for the rest.

Measured on the real cloud, 8×10⁷ particles on a 258³ grid:

| group | projectile | reduction | ms |
|---|---|---|---|
| 256 | yes | yes | 860.5 |
| 256 | yes | no | 633.1 |
| 256 | no | no | 607.5 |
| **64** | **yes** | **yes** | **632.9** |
| 128 | yes | yes | 742.6 |
| 32 | yes | yes | 756.7 |

So at 64 the whole projectile — its `exp`, its `sqrt`, its `erf` **and** its
reduction — costs 25 ms over a kernel that does none of it, where at 256 it cost
253. Below 64 it turns again: four times as many groups means four times as many
atomics onto `red`, and too few work-items to hide the contraction's latency.

⚠️ The contraction is register-hungry enough that Metal refuses 512 work-items
outright ("should not exceed 448"), which is the same story seen from the other
side.
"""
const FIELD_GROUPSIZE = 64

"""Work-items per group for [`_deposit_sorted_kernel!`](@ref): the 8³ stencil."""
const DEPOSIT_GROUPSIZE = 512

"""Particles staged through threadgroup memory at a time, in the deposition.

128 and not 256, which measures 2 % faster here: the staging buffer holds 24
values per particle, so 256 would ask 24 KB of threadgroup memory in `Float32`
and **49 KB in `Float64`** — past what a CUDA block gets by default. 128 costs
30 KB in `Float64` and keeps the kernel portable."""
const DEPOSIT_STAGE = 128

"""Group index of a work-item, from its global index — `@index(Group, Linear)`
is unavailable in the scope where the deposition needs it."""
@inline _group_of(gi) = (Int32(gi) - Int32(1)) ÷ Int32(DEPOSIT_GROUPSIZE) + Int32(1)

"""The `(i,j,k)` offset, within the 8³ stencil, owned by work-item `t`."""
@inline function _stencil_offset(t)
    t0 = t - Int32(1)
    ii = t0 % Int32(8) + Int32(1); t0 ÷= Int32(8)
    jj = t0 % Int32(8) + Int32(1); t0 ÷= Int32(8)
    (ii, jj, t0 + Int32(1))
end

"""Corner of the stencil of linear cell `cell`, and whether it fits in an `n³`
grid."""
@inline function _cell_base(cell, nk, n)
    c0 = cell - Int32(1)
    kx = c0 % nk + Int32(1); c0 ÷= nk
    ky = c0 % nk + Int32(1); c0 ÷= nk
    kz = c0 + Int32(1)
    bx = Int32(2) * kx - Int32(5)
    by = Int32(2) * ky - Int32(5)
    bz = Int32(2) * kz - Int32(5)
    inside = bx >= Int32(0) && by >= Int32(0) && bz >= Int32(0) &&
             bx + Int32(8) <= n && by + Int32(8) <= n && bz + Int32(8) <= n
    (bx, by, bz, inside)
end

"""
Packs the positions into `(k, δ)` form: index of the nearest knot, and offset
from that knot.

⚠️ **This one runs in `T`, the simulation's own precision**, and not in `E`.
`δ` is bounded by half a step and its `Float32` encoding resolves 1.2e-07, far
finer than a column of the tables; but the *subtraction* `p − knot` cancels the
leading digits and must happen in full precision, once, before the result is
narrowed.

Which is why the backend that runs this kernel is not always the one that runs
the others: on Metal, where `Float64` does not exist, it runs on `CPU()` over
the host positions — free, the memory being unified. On a backend with hardware
`Float64` it runs on the device like everything else.
"""
@kernel function _pack_kd_kernel!(parts, @Const(positions), x0::T, h::T,
                                  nk) where {T}
    i = @index(Global, Linear)
    @inbounds begin
        p = positions[i]
        E = offset_type(eltype(parts))
        ks = ntuple(d -> clamp(round(Int32, (p[d] - x0) / h) + Int32(1), Int32(1), nk), 3)
        ds = ntuple(d -> E(p[d] - (x0 + (ks[d] - Int32(1)) * h)), 3)
        old = parts[i]
        parts[i] = PackedParticle(ks, ds, old.pknode, old.pdelta)
    end
end

"""
    contract_tile(tile, ovl, grad, cx, cy, cz)

[`contract_spline_10`](@ref) reading its 10³ stencil from threadgroup memory
instead of from `csol`.

The arithmetic is the same, in the same order — the two agree bit for bit. Only
the indexing differs: the stencil has been copied into a dense `10×10×10` block,
so the strides are 1, 10 and 100 rather than those of the grid.

⚠️ **The index width must be the same throughout.** An `Int32` offset added to
an `Int` loop variable costs a factor of four here, silently. Measured, 8×10⁷
particles:

| offset | loop variable | ms |
|---|---|---:|
| `Int` | `Int` | 521.0 |
| **`Int32`** | **`Int`** | **1969.1** |
| `Int32` | `Int32` | 522.8 |
| `Int` | `Int32` | 519.4 |

Only the mixture is slow, and only in that direction. `for ii in 1:10` yields an
`Int`, so narrowing *the offsets alone* to match the file's `Int32` style — which
is the natural way to write this — is exactly the way to fall in. The
accumulator's type has nothing to do with it: `eltype(ovl)` and a hard-wired
`Float32` measure the same in every index regime.

The same narrowing is *right* in [`_deposit_sorted_kernel!`](@ref), whose stencil
offsets already come out of `_stencil_offset` as `Int32`: there everything is
`Int32` and widening the offsets to `Int` costs 17 % (242.2 → 282.9 ms). So the
rule is not "prefer one width" but "do not mix them".
"""
@inline function contract_tile(tile, ovl, grad, cx, cy, cz)
    T = eltype(ovl)
    fx = zero(T); fy = zero(T); fz = zero(T)
    @inbounds for kk in Int32(1):Int32(10)
        oz = ovl[kk, cz]; gz = grad[kk, cz]
        bk = Int32(100) * (kk - Int32(1))
        for jj in Int32(1):Int32(10)
            oy = ovl[jj, cy]; gy = grad[jj, cy]
            b = bk + Int32(10) * (jj - Int32(1))
            dxp = zero(T); val = zero(T)
            for ii in Int32(1):Int32(10)
                c = tile[b+ii]
                dxp = fma(c, grad[ii, cx], dxp)
                val = fma(c, ovl[ii, cx], val)
            end
            fx = fma(oy * oz, dxp, fx)
            fy = fma(gy * oz, val, fy)
            fz = fma(oy * gz, val, fz)
        end
    end
    (fx, fy, fz)
end

"""
Smoothed field at each pseudo-particle, with the projectile fused in.

**One group per occupied cell**, and the cell's 10³ stencil of `csol` — 4 KB —
staged once in threadgroup memory. Every particle of a cell shares the *same*
stencil, so the version that gave one work-item to each particle made each of
them fetch those 4 KB for itself.

That was not a guess. Apple's counters, on this kernel at 8×10⁷ particles:

| counter | |
|---|---:|
| **Buffer Read Limiter** | **98.5 %** |
| GPU Last Level Cache Limiter | 85.3 % |
| **Threadgroup/Imageblock Load Limiter** | **0.0 %** |
| Compute Occupancy | 18.4 % |
| ALU Utilization | 12.9 % |
| GPU Read Bandwidth | 7.6 GB/s |

The load port was saturated while the threadgroup path sat idle and the ALUs
ran at 13 %. Staging moves the traffic from the port that was full onto the one
that was empty:

    619.4 -> 518.3 ms   ×1.20   at 8×10⁷ particles, 428 per occupied cell

Bit for bit identical: the summation order is untouched, only where the values
are read from changes.

⚠️ **The cell index *is* `knode`.** [`CellSort`](@ref) keys on
`round((p−x₀)/h)+1` clamped to `[1,nk]`, which is exactly what
[`_pack_kd_kernel!`](@ref) stores. So `(bx,by,bz)` follows from the cell and the
per-particle gather of `knode` disappears with it.

⚠️ For the same reason the "does the stencil fit in the grid" test depends only
on the cell, so it is **uniform over the group** and is evaluated once rather
than per particle. Particles of a cell that does not fit all write `NaN` and are
handed back to the caller through the compacted list.

⚠️ The projectile's contribution is accumulated **even for those particles** —
it does not need the grid — then reduced in a tree and committed with one atomic
per group. A work-item now handles several particles, so it sums them into a
register first and touches threadgroup memory once.

What this did *not* fix, and the measurements that say so: the kernel now runs
at 756 GFLOP/s against the 836 it reaches with every read served from L1, so
90 % of the remaining distance is arithmetic, not memory. Rejected on the way,
each by measurement — hoisting the `x` tables into registers (×0.96, LLVM
already does it), splitting the accumulator into four chains (×1.01, not latency
bound), and reading `csol` in pairs (×0.99, already vectorised by the
compiler).

⚠️ `unsafe_indices = true`: the tree reduction below calls `@synchronize`, which
every work-item of a group must reach. KernelAbstractions' automatic bounds
check would wrap the body in a conditional and make that false, so the guard is
written by hand and `ndrange` is padded to a whole number of groups.
"""
@kernel unsafe_indices = true function _smoothed_field_kernel!(
        force, @Const(csol), @Const(ovl), @Const(grad),
        @Const(parts), @Const(cellids), @Const(bounds), nk,
        x0, h, spacing, nbdt, w, nc, px0, py0, pz0, coef, σ, red)
    tid = @index(Local, Linear)
    gi = @index(Global, Linear)
    @uniform GS = Int32(FIELD_GROUPSIZE)
    @uniform E = eltype(force)
    # ⚠️ `@uniform`, and declared here rather than beside its loop: the CPU
    # backend cuts the kernel into one work-item loop per `@synchronize`, and a
    # plain local assigned in one segment does not exist in the next. The tree
    # reduction's counter crosses every barrier, so it has to live outside them.
    @uniform redhalf = Int32(FIELD_GROUPSIZE) ÷ Int32(2)
    sh = @localmem eltype(force) (4 * FIELD_GROUPSIZE,)
    "The cell's 10³ stencil of `csol`, read once for all its particles."
    tile = @localmem eltype(force) (1000,)
    "`[lo, hi, bx, by, bz, ok]` — group-uniform, hence threadgroup memory."
    st = @localmem Int32 (6,)

    for c in Int32(0):Int32(3)
        @inbounds sh[c*GS+tid] = zero(E)
    end
    @inbounds if tid == Int32(1)
        g = (Int32(gi) - Int32(1)) ÷ GS + Int32(1)
        c0 = cellids[g] - Int32(1)
        kx = c0 % nk + Int32(1); c0 ÷= nk
        ky = c0 % nk + Int32(1); c0 ÷= nk
        kz = c0 + Int32(1)
        bx = Int32(2) * kx - Int32(5)
        by = Int32(2) * ky - Int32(5)
        bz = Int32(2) * kz - Int32(5)
        nn = Int32(size(csol, 1))
        st[1] = bounds[g]; st[2] = bounds[g+Int32(1)]
        st[3] = bx; st[4] = by; st[5] = bz
        st[6] = (bx >= Int32(1) && by >= Int32(1) && bz >= Int32(1) &&
                 bx + Int32(9) <= nn && by + Int32(9) <= nn &&
                 bz + Int32(9) <= nn) ? Int32(1) : Int32(0)
    end
    @synchronize

    # --- the stencil, fetched once for the whole cell ----------------------
    @inbounds if st[6] == Int32(1)
        bx = st[3]; by = st[4]; bz = st[5]
        e = tid
        while e <= Int32(1000)
            e0 = e - Int32(1)
            ii = e0 % Int32(10)
            jj = (e0 ÷ Int32(10)) % Int32(10)
            kk = e0 ÷ Int32(100)
            tile[e] = csol[bx+ii, by+jj, bz+kk]
            e += GS
        end
    end
    @synchronize

    # --- the cell's particles ----------------------------------------------
    @inbounds begin
        lo = st[1]; hi = st[2]; ok = st[6] == Int32(1)
        kx = (st[3] + Int32(5)) ÷ Int32(2)
        ky = (st[4] + Int32(5)) ÷ Int32(2)
        kz = (st[5] + Int32(5)) ÷ Int32(2)
        halfsp = spacing * E(0.5)
        nan = E(NaN)
        ax = zero(E); ay = zero(E); az = zero(E); ae = zero(E)
        s = lo + tid
        while s <= hi
            # The cloud is sorted, so `s` *is* the particle. What used to be
            # `perm[s]` then `delta[·, i]` was a gather of 132 ms; the store
            # below was a scatter of 153.
            dd = parts[s].delta
            dx0 = dd[1]; dy0 = dd[2]; dz0 = dd[3]
            fxp = nan; fyp = nan; fzp = nan
            if ok
                cx = min(max(floor(Int32, (dx0 + halfsp) / spacing * nbdt + E(0.5)) +
                             Int32(1), Int32(1)), nc)
                cy = min(max(floor(Int32, (dy0 + halfsp) / spacing * nbdt + E(0.5)) +
                             Int32(1), Int32(1)), nc)
                cz = min(max(floor(Int32, (dz0 + halfsp) / spacing * nbdt + E(0.5)) +
                             Int32(1), Int32(1)), nc)
                fx, fy, fz = contract_tile(tile, ovl, grad, cx, cy, cz)
                fxp = -w * fx; fyp = -w * fy; fzp = -w * fz
            end

            # --- projectile ↔ pseudo-electron, fused ------------------------
            # Runs whether or not the stencil fits: it needs no grid, and the
            # reaction on the projectile is owed by every particle.
            if coef != zero(E)
                px = x0 + E(kx - Int32(1)) * h + dx0
                py = x0 + E(ky - Int32(1)) * h + dy0
                pz = x0 + E(kz - Int32(1)) * h + dz0
                dx = px0 - px; dy = py0 - py; dz = pz0 - pz
                d2 = dx * dx + dy * dy + dz * dz
                mm = coef * gaussian_force_kernel(d2, σ)
                fx2 = mm * dx; fy2 = mm * dy; fz2 = mm * dz
                if ok
                    fxp -= fx2; fyp -= fy2; fzp -= fz2       # reaction
                end
                r = sqrt(d2)
                ax += fx2; ay += fy2; az += fz2
                ae += r < E(1.0e-4) * σ ? E(0.7978845608) / σ :
                      erf(r / E(1.4142135624) / σ) / r
            end

            force[1, s] = fxp; force[2, s] = fyp; force[3, s] = fzp
            s += GS
        end
        # One visit to threadgroup memory per work-item, not one per particle.
        sh[tid] = ax; sh[GS+tid] = ay; sh[2*GS+tid] = az; sh[3*GS+tid] = ae
    end

    @synchronize
    # ⚠️ Not named `stride`: that is `Base.stride`, a function, and the CPU
    # backend resolves the bare name to the global rather than to a local — the
    # comparison then fails with `isless(::Int32, ::Function)`.
    while redhalf > Int32(0)
        if tid <= redhalf
            @inbounds for c in Int32(0):Int32(3)
                sh[c*GS+tid] += sh[c*GS+tid+redhalf]
            end
        end
        @synchronize
        redhalf ÷= Int32(2)
    end
    if tid == Int32(1)
        @inbounds for c in Int32(0):Int32(3)
            Atomix.@atomic red[c+Int32(1)] += sh[c*GS+Int32(1)]
        end
    end
end

"""
Charge deposition, **sorted** version.

Deposition is the hard half to port: each particle writes into 8³ grid points,
and neighbouring particles write into the same ones. The naive route — one
atomic per point per particle — is *three times slower than the CPU*: 410
million contending atomics, which the hardware serialises.

So the particles are ordered by cell first ([`CellSort`](@ref)), one group is
given one cell, and each of its 512 work-items owns **one** of the 512 points of
the stencil. Each sweeps every particle of the cell into a register and performs
a single atomic at the end: one atomic per point per **cell**, a hundred times
fewer.

The particles of a cell are staged through threadgroup memory
[`DEPOSIT_STAGE`](@ref) at a time, so that the 512 work-items read each one once
from global memory rather than 512 times.

⚠️ **What is staged is the 24 tabulated values, not the 3 column indices.**

The obvious staging — and the one this kernel had — put the three columns of
each particle in threadgroup memory, and the inner loop used them as *addresses*
into `nodes`:

    s += nodes[ii, shared[…]] * nodes[jj, shared[…]] * nodes[kk, shared[…]]

That is a chain of two dependent loads — threadgroup, then global — and nothing
can hide it: the column changes with every particle, so no compiler may hoist
the second load, and the loop cannot be software-pipelined. It ran at **108
cycles per iteration** for a body worth three.

Staging the values instead — `nodes[1:8, c]` for each of the three axes, 24
floats per particle — leaves the inner loop three *independent* threadgroup
reads. Measured on the real Thomas-Fermi cloud, 8×10⁷ particles on a 258³ grid,
428 particles per occupied cell:

| | ms | cycles/iter |
|---|---|---|
| columns staged (stage 64) | 834.1 | 108.1 |
| values staged (stage 64) | 274.1 | 35.5 |
| **values staged (stage 128)** | **241.6** | **31.3** |
| values staged (stage 256) | 236.7 | 30.7 |

⚠️ On a **synthetic** cloud the same A/B reads ×2.81 rather than ×3.45, because
it spreads the particles over 28 per cell instead of 428 — and the staging cost
is amortised over exactly that number. The coarse deposition was rejected twice
on that mistake; see [`_deposit_cic_kernel!`](@ref).

The density agrees with the previous version to 4.2e-07 and the charge to the
last printed digit — the residue is atomic ordering, not the change.

Against the hand-written Metal kernel this one replaces, and which staged
columns too, portability now costs nothing at all: it is the faster of the two.
It also runs on `CPU()`, where it agrees with `deposit_smoothed!` to 9.3e-16 in
`Float64` — the check Metal can never perform.

⚠️ **What crosses a barrier must live in threadgroup memory or be `@uniform`.**
The CPU backend cuts a kernel into one work-item loop per `@synchronize`, and a
plain local assigned before a barrier is simply undefined after it. So the
per-item accumulator sits in `@localmem` rather than in a register, and the
stencil offsets are **recomputed** on each side of a barrier — they are three
integer divisions, cheaper than the machinery needed to carry them across.
"""
@kernel unsafe_indices = true function _deposit_sorted_kernel!(
        ρ, @Const(nodes), @Const(cols), @Const(cellids),
        @Const(bounds), nk)
    @uniform E = eltype(ρ)
    t = @index(Local, Linear)

    shared = @localmem Int32 (3 * DEPOSIT_STAGE,)
    "The 8 tabulated values of each axis, for each staged particle."
    vals = @localmem eltype(ρ) (24 * DEPOSIT_STAGE,)
    accum = @localmem eltype(ρ) (DEPOSIT_GROUPSIZE,)
    # ⚠️ The cursor lives in threadgroup memory, not in a `@uniform`. The loop
    # below straddles barriers, so KernelAbstractions emits it in the group's
    # scope — where a plain local is out of reach, and where the CPU backend
    # refuses `@index(Group, Linear)` outright. Threadgroup memory is the one
    # thing both scopes can see. `state` is `[cursor, hi]`.
    state = @localmem Int32 (2,)

    # ⚠️ `@index` is read at the top level of each segment, never inside an
    # `@inbounds` block: the macro rewrites it by looking at the syntax around
    # it, and wrapped in `@inbounds` it loses the work-item index and lowers to
    # a call that has no method.
    gi = @index(Global, Linear)
    @inbounds begin
        accum[t] = zero(E)
        if t == Int32(1)
            g = _group_of(gi)
            state[1] = bounds[g]
            state[2] = bounds[g+Int32(1)]
        end
    end
    @synchronize

    while state[1] < state[2]
        @inbounds begin
            cursor = state[1]
            chunk = min(Int32(DEPOSIT_STAGE), state[2] - cursor)
            if t <= chunk
                # The sort lives here, in the staging: one gather of 12 bytes
                # per particle per cell, overlapped with the inner loop below —
                # rather than a pass of its own over every particle.
                # Sorted cloud: the slot *is* the particle, so these three
                # reads are contiguous instead of gathered through `perm`.
                q = cursor + t
                b = Int32(3) * (t - Int32(1))
                shared[b+Int32(1)] = cols[1, q]
                shared[b+Int32(2)] = cols[2, q]
                shared[b+Int32(3)] = cols[3, q]
            end
        end
        @synchronize

        # The gather into `nodes` happens **here**, once per value, instead of
        # inside the inner loop where it happened once per value per work-item.
        # The 24·chunk values are spread over the whole group, each work-item
        # taking every `DEPOSIT_GROUPSIZE`-th of them.
        @inbounds begin
            chunk = min(Int32(DEPOSIT_STAGE), state[2] - state[1])
            nval = Int32(24) * chunk
            e = t
            while e <= nval
                e0 = e - Int32(1)
                q = e0 ÷ Int32(24)          # particle within the stage
                r = e0 % Int32(24)
                d = r ÷ Int32(8)            # axis, 0-based
                a = r % Int32(8) + Int32(1) # which of the 8 stencil nodes
                vals[e] = nodes[a, shared[Int32(3)*q+d+Int32(1)]]
                e += Int32(DEPOSIT_GROUPSIZE)
            end
        end
        @synchronize

        # Recomputed on this side of the barrier rather than carried across it:
        # three integer divisions are cheaper than the machinery that would be
        # needed to make them survive.
        gj = @index(Global, Linear)
        @inbounds begin
            chunk = min(Int32(DEPOSIT_STAGE), state[2] - state[1])
            ii, jj, kk = _stencil_offset(t)
            g = _group_of(gj)
            _, _, _, inside = _cell_base(cellids[g], nk, Int32(size(ρ, 1)))
            if inside
                s = zero(E)
                # ⚠️ The offsets into `vals` are computed in `Int`, not `Int32`
                # — see [`contract_tile`](@ref), where the same narrowing on the
                # same kind of access cost a factor of 3.8. `ii`, `jj` and `kk`
                # stay `Int32`: they index nothing here, they are added.
                for q in Int32(1):chunk
                    b = Int32(24) * (q - Int32(1))
                    s += vals[b+ii] * vals[b+Int32(8)+jj] * vals[b+Int32(16)+kk]
                end
                accum[t] += s
            end
        end
        @synchronize

        @inbounds if t == Int32(1)
            state[1] += min(Int32(DEPOSIT_STAGE), state[2] - state[1])
        end
        @synchronize
    end

    gk = @index(Global, Linear)
    @inbounds begin
        ii, jj, kk = _stencil_offset(t)
        g = _group_of(gk)
        bx, by, bz, inside = _cell_base(cellids[g], nk, Int32(size(ρ, 1)))
        if inside && accum[t] != zero(E)
            Atomix.@atomic ρ[bx+ii, by+jj, bz+kk] += accum[t]
        end
    end
end

"""
Table columns, in sorted order, for the deposition.

One work-item per particle, in the cloud's **own** order — not the sorted one.
Everything it reads and everything it writes is then sequential.

⚠️ It used to walk the sorted order, which cost a gather of `knode` and `delta`
— 24 bytes scattered over `npart` — and that gather was the whole expense:
392 ms of the step at 8×10⁷ particles, for arithmetic worth nothing. Sorting is
still needed, but by the **deposition**, which gathers 12 bytes instead of 24
and hides the latency behind its inner loop.

The position is **rebuilt** from its packed form, `p = x₀ + (k−1)h + δ`, rather
than read from the cloud: that is what lets this run on the device without the
positions having to live there too.

The rebuild happens in `E`, so where `E` is `Float32` the boundary test and the
column rounding shift by a few ulp, and particles on the very edge could in
principle fall on the other side. Measured rather than feared: on 2×10⁶
particles deliberately banked against the edge of the fine grid, Metal and the
`Float64` host reference reject **the same 136 140** — not one reclassified.
The density then agrees to 4.3e-05, which is the atomic ordering and not this.

Particles outside are parked on column 1 — they deposit nothing, the kernel
skips them — and counted into `nout`, one atomic each. They are rare.
"""
@kernel function _columns_kernel!(cols, @Const(parts),
                                  x0, h, lo, hi, spacing, nbdt, ncol, nout)
    i = @index(Global, Linear)
    @inbounds begin
        q = parts[i]
        knode = q.knode
        delta = q.delta
        E = eltype(delta)
        half = spacing * E(0.5)
        inside = true
        for d in Int32(1):Int32(3)
            p = x0 + E(knode[d] - Int32(1)) * h + delta[d]
            inside &= (lo <= p) & (p <= hi)
        end
        if inside
            for d in Int32(1):Int32(3)
                cols[d, i] = min(max(floor(Int32, (delta[d] + half) / spacing *
                                           nbdt + E(0.5)) + Int32(1),
                                     Int32(1)), ncol)
            end
        else
            cols[1, i] = Int32(1); cols[2, i] = Int32(1); cols[3, i] = Int32(1)
            Atomix.@atomic nout[1] += Int32(1)
        end
    end
end

"""
Copies the interior solution into the full grid — `φ[2:end-1, …] = rhs`.

⚠️ **Written as a kernel because the broadcast is not.**
`@views φ[2:end-1, 2:end-1, 2:end-1] .= rhs` is the obvious spelling and reads
better, but on Metal a broadcast into a **non-contiguous view** falls back to an
element-at-a-time path. Measured on a 258³ grid, 134 MB of traffic:

| | ms | GB/s |
|---|---:|---:|
| `@views … .= rhs` | 24.8 | 5.4 |
| this kernel | 3.1 | 43 |

×8, and the two calls were 30 % of the whole nested Poisson chain — more than
any of its fifteen GEMMs. The host path keeps the broadcast: there it is a
strided copy like any other, and `Array` handles it.

⚠️ **The work-item index is linear, and the `(i,j,k)` computed from it**, rather
than the obvious `@index(Global, NTuple)` over a 3-D `ndrange` — on the same
reasoning as the `vec` in `_solve!`, where a flat traversal beats a cartesian
one on the same buffer. 43 GB/s is still a fifth of what a flat copy reaches, so
there is more here for whoever wants it.
"""
@kernel function _fill_interior_kernel!(φ, @Const(rhs), nx, ny, n)
    t = @index(Global, Linear)
    @inbounds if t <= n
        t0 = t - 1
        i = t0 % nx
        j = (t0 ÷ nx) % ny
        k = t0 ÷ (nx * ny)
        φ[i+2, j+2, k+2] = rhs[i+1, j+1, k+1]
    end
end

"""
    nface(nx, ny, nz) -> Int

How many points lie on the surface of an `nx × ny × nz` grid.

Two full faces, plus a ring of `2nx + 2ny − 4` for each of the `nz − 2` layers
between them — the `−4` because the corners would otherwise be counted twice.
"""
@inline nface(nx, ny, nz) = 2 * nx * ny + (nz - 2) * (2 * nx + 2 * ny - 4)

"""
    _face_point(t, nx, ny, nz) -> (i, j, k)

The `t`-th surface point, `t` running over `1:nface(nx,ny,nz)`.

This is the kernel's counterpart to `foreach_face`, and it exists for the same
reason: sweeping the volume and rejecting the interior would visit 195 000
points to write 19 500 — nine tenths of the work spent deciding to do nothing.
Here the enumeration is inverted instead, so each work-item lands on a point it
will actually write.
"""
@inline function _face_point(t, nx, ny, nz)
    t0 = t - Int32(1)
    cap = nx * ny
    if t0 < cap                       # the k = 1 face
        return (t0 % nx + Int32(1), t0 ÷ nx + Int32(1), Int32(1))
    end
    t0 -= cap
    if t0 < cap                       # the k = nz face
        return (t0 % nx + Int32(1), t0 ÷ nx + Int32(1), nz)
    end
    t0 -= cap
    ring = Int32(2) * nx + Int32(2) * (ny - Int32(2))
    k = t0 ÷ ring + Int32(2)          # the side walls, k = 2 … nz−1
    r = t0 % ring
    if r < nx
        (r + Int32(1), Int32(1), k)
    elseif r < Int32(2) * nx
        (r - nx + Int32(1), ny, k)
    elseif r < Int32(2) * nx + (ny - Int32(2))
        (Int32(1), r - Int32(2) * nx + Int32(2), k)
    else
        (nx, r - Int32(2) * nx - (ny - Int32(2)) + Int32(2), k)
    end
end

"""
Multipole potential on the faces of the domain — the Dirichlet values the
interior solve is lifted against.

`mp` travels by value: a `Multipole` is three numbers and two small tuples, so
it is `isbits` and needs no buffer of its own.
"""
@kernel function _boundary_potential_kernel!(φ, mp, @Const(gx), @Const(gy),
                                             @Const(gz), nx, ny, nz)
    t = @index(Global, Linear)
    i, j, k = _face_point(Int32(t), nx, ny, nz)
    @inbounds φ[i, j, k] = potential(mp, gx[i], gy[j], gz[k])
end

"""
Dirichlet values read from a coarser grid's solution, onto the faces of a finer
one — the inter-grid junction.

Same inverted enumeration as [`_boundary_potential_kernel!`](@ref), and the same
reason for it.

⚠️ A point outside the coarse grid writes `NaN` rather than raising. That case
is impossible by construction — `NestedMeshes` refuses at build time a level
that sticks out of the one above it — and a per-point check inside a kernel, for
a condition the type system already guarantees, is exactly what one does not put
there. The `NaN` is the visible trace should that guarantee ever be broken.
"""
@kernel function _boundary_from_coarse_kernel!(φ, @Const(csol),
                                               @Const(kx), @Const(ky), @Const(kz),
                                               @Const(gx), @Const(gy), @Const(gz),
                                               nx, ny, nz)
    t = @index(Global, Linear)
    i, j, k = _face_point(Int32(t), nx, ny, nz)
    @inbounds begin
        v = spline_potential((kx, ky, kz), csol, (gx[i], gy[j], gz[k]))
        φ[i, j, k] = v === nothing ? eltype(φ)(NaN) : v
    end
end

"""Linear cell index from an unclamped `knode` — the clamp
[`PackedPositions`](@ref) does not apply lives here, where a stencil is indexed."""
@inline function _cell_key(p::PackedParticle, nk::Int32)
    k = p.knode
    kx = clamp(k[1], Int32(1), nk)
    ky = clamp(k[2], Int32(1), nk)
    kz = clamp(k[3], Int32(1), nk)
    kx + nk * (ky - Int32(1) + nk * (kz - Int32(1)))
end

"""Counts the particles of each cell — one atomic per particle, into `nk³` bins."""
@kernel function _hist_cells_kernel!(counts, @Const(parts), nk::Int32, npart::Int32)
    i = @index(Global, Linear)
    if i <= npart
        @inbounds Atomix.@atomic counts[_cell_key(parts[i], nk)] += Int32(1)
    end
end

"""
The placement — **moving the particles themselves**, not their indices, and
walking the array in its own order.

⚠️ **The natural order is the fast one, because the cloud is already almost
sorted.** It was sorted at the previous step, and one step of drift moves a
particle a median of 2230 places out of 8×10⁷: source and destination are
neighbours, so the scatter stays inside a few pages.

That inverts an earlier design. A two-stage placement — bin into buckets of
neighbouring cells, then sort within — exists to *manufacture* locality for a
cloud in random order, and it works: 118 ms down to 47 on a freshly sampled
cloud. But on a cloud kept sorted it **destroys** the locality that is already
there, by walking the particles in the first stage's order instead of the
array's. Measured at 8×10⁷ particles, on the cloud as the time loop leaves it:

| | ms |
|---|---:|
| two stages, walking the coarse order | 151.0 |
| **one pass, walking the array** | **35.5** |

So the first stage is gone, along with its buckets and its tuning. The first
step of a run pays a disordered placement once; every step after it walks a
cloud it sorted itself.

The destination `out` cannot be `parts`: a scatter has no safe in-place form.
"""
@kernel function _place_particles_kernel!(out, cursor, @Const(parts),
                                          nk::Int32, npart::Int32)
    i = @index(Global, Linear)
    if i <= npart
        @inbounds begin
            q = parts[i]
            p = Atomix.@atomic cursor[_cell_key(q, nk)] += Int32(1)
            out[p] = q
        end
    end
end

"""Gathers the particles into the order `perm` gives. The host route's
counterpart to [`_place_particles_kernel!`](@ref), which scatters."""
@kernel function _gather_particles_kernel!(out, @Const(parts), @Const(perm), npart::Int32)
    s = @index(Global, Linear)
    if s <= npart
        @inbounds out[s] = parts[perm[s]]
    end
end

"""
The position Verlet on a cloud held as `(k, δ)`, on the device.

`k(t+dt) = 2k(t) − k(t−dt)` is exact and every rounding falls on `δ` — see
[`PackedPositions`](@ref). The forces are read **where the field kernel left
them**, in `E`, so the per-step conversion of 8×10⁷ triples to host `Float64`
goes away with this kernel rather than being repeated for it.

The three diagnostics are reductions, and they are done the way
[`_total_charge_kernel!`](@ref) does them rather than with a tree: each
work-item walks the particles by a grid stride — so the reads stay coalesced —
accumulates in registers, and writes one partial. The host sums those in `T`.
Splitting the sum this way is also what keeps it accurate: a few dozen terms per
work-item in `E`, not eighty million.

⚠️ Everything the kernel takes is in `E`. `x0` and `h` rebuild the absolute
position, which only `rcmax` and the angular momentum need; the trajectory never
passes through it, which is the whole point of the packed form.
"""
@kernel function _verlet_packed_kernel!(parts, @Const(force), partials,
                                        a::E, h::E, x0::E, pfac::E, r2max::E,
                                        inv2M::E, npart::Int32,
                                        stride::Int32) where {E}
    t = @index(Global, Linear)
    ek = zero(E); eo = zero(E)
    lx = zero(E); ly = zero(E); lz = zero(E)

    i = Int32(t)
    @inbounds while i <= npart
        q = parts[i]
        k1, k2, k3 = q.knode
        d1, d2, d3 = q.delta
        o1, o2, o3 = q.pknode
        e1, e2, e3 = q.pdelta

        n1 = _verlet_packed(k1, d1, o1, e1, a, force[1, i], h)
        n2 = _verlet_packed(k2, d2, o2, e2, a, force[2, i], h)
        n3 = _verlet_packed(k3, d3, o3, e3, a, force[3, i], h)

        px = n1[3] * pfac; py = n2[3] * pfac; pz = n3[3] * pfac
        e = (px * px + py * py + pz * pz) * inv2M
        ek += e
        qx = x0 + E(k1 - Int32(1)) * h + d1
        qy = x0 + E(k2 - Int32(1)) * h + d2
        qz = x0 + E(k3 - Int32(1)) * h + d3
        qx * qx + qy * qy + qz * qz > r2max && (eo += e)
        lx += qy * pz - qz * py
        ly += qz * px - qx * pz
        lz += qx * py - qy * px

        parts[i] = PackedParticle((n1[1], n2[1], n3[1]), (n1[2], n2[2], n3[2]),
                                  (k1, k2, k3), (d1, d2, d3))
        i += stride
    end

    @inbounds begin
        partials[1, t] = ek; partials[2, t] = eo
        partials[3, t] = lx; partials[4, t] = ly; partials[5, t] = lz
    end
end

"""
Compacts, into `outlist`, the indices of the particles whose 10³ stencil does
not fit inside the fine grid.

They take the coarse-grid path instead, and there are few of them. The list is
built **once per step**, right after the packing, because it depends only on
where the particles are — not on which potential is being evaluated. That is
what lets the forces and the energy budget share it, although the budget runs
first.

The append is one atomic per outside particle, and the slot it returns is the
particle's place in the list. Order is not preserved, and nothing downstream
wants it to be.
"""
@kernel function _outside_kernel!(outlist, outcount, @Const(parts), n, npart)
    i = @index(Global, Linear)
    @inbounds if i <= npart
        # ⚠️ The same test, spelled the same way, as in the field kernel. The
        # two must agree exactly: one decides what to skip, the other what to
        # pick up.
        knode = parts[i].knode
        bx = Int32(2) * knode[1] - Int32(5)
        by = Int32(2) * knode[2] - Int32(5)
        bz = Int32(2) * knode[3] - Int32(5)
        ok = bx >= Int32(1) && by >= Int32(1) && bz >= Int32(1) &&
             bx + Int32(9) <= n && by + Int32(9) <= n && bz + Int32(9) <= n
        if !ok
            slot = Atomix.@atomic outcount[1] += Int32(1)
            outlist[slot] = Int32(i)
        end
    end
end


"""
Cell of `x` on a **stretched** axis, and the weight falling to its left node.

Device form of [`locate`](@ref): takes the collocation points and the lookup
table's three fields rather than the objects holding them. Returns cell `0`
outside the domain — a sentinel rather than `nothing`, so that the caller can
branch without a union.
"""
@inline function _locate_cell(gt, x0, invwidth, cells, x)
    E = eltype(gt)
    (x < gt[1] || x > gt[end]) && return (Int32(0), zero(E))
    @inbounds begin
        b = min(floor(Int32, (x - x0) * invwidth) + Int32(1), Int32(length(cells)))
        c = cells[b]
        lastc = Int32(length(gt) - 1)
        while c < lastc && gt[c+1] < x
            c += Int32(1)
        end
        (c, (gt[c+1] - x) / (gt[c+1] - gt[c]))
    end
end

"""
Coarse-grid deposition — cloud-in-cell, on the device.

One work-item per particle, eight atomics each: the *naive* scheme, and here it
is the right one. The fine deposition had to abandon it — 410 million contending
atomics onto an 8³ stencil — but this grid is coarser and each particle touches
only its 8 corners, so the contention is an order of magnitude milder.

Measured against the threaded host scatter it replaces, 8×10⁷ particles on a
258³ coarse grid: **203 ms against 396, ×1.95**, with the density agreeing to
5.0e-06 and the charge to 2e-08. A *sorted* version was tried too, on the model
of the fine deposition; it fixes nothing that needs fixing here and the coarse
sort it requires costs 356 ms on its own — see `_update_forces_resident!`.

⚠️ Two traps, both paid for:

  * the naive form looks catastrophic — 1450 ms — when measured on a synthetic
    cloud more concentrated than the real one. **Contention depends on the
    distribution**, so this has to be measured on a Thomas-Fermi sample;
  * the position is rebuilt from the fine grid's packed form, so `_pack!` must
    have run **on the current positions**. Compare against a host deposition
    after a Verlet step and the densities differ by 4.5 % — not a numerical
    error, simply two different sets of particles.
"""
@kernel function _deposit_cic_kernel!(ρ, @Const(parts),
                                      @Const(gx), @Const(gy), @Const(gz),
                                      @Const(cells), x0f, hf, x0t, iwt, npart,
                                      stride)
    t = @index(Global, Linear)
    # ⚠️ **Deliberately out of order.** The cloud is sorted by *fine* cell, so
    # consecutive particles share a *coarse* cell — and this kernel's eight
    # atomics then all land on the same handful of addresses at once. Measured
    # at 8×10⁷ particles: **4874 ms** walking the sorted order against **193**
    # walking it by a stride coprime with the count.
    #
    # The sorted order is what makes the fine deposition fast (one group per
    # cell, one atomic each) and what makes this one slow. Reading by a stride
    # costs locality on the load and buys back a factor of 25 on the atomics.
    i = Int32((Int64(t - 1) * Int64(stride)) % Int64(npart)) + Int32(1)
    @inbounds if t <= npart
        E = eltype(ρ)
        q = parts[i]; knode = q.knode; delta = q.delta
        px = x0f + E(knode[1] - Int32(1)) * hf + delta[1]
        py = x0f + E(knode[2] - Int32(1)) * hf + delta[2]
        pz = x0f + E(knode[3] - Int32(1)) * hf + delta[3]
        ix, ax = _locate_cell(gx, x0t, iwt, cells, px)
        iy, ay = _locate_cell(gy, x0t, iwt, cells, py)
        iz, az = _locate_cell(gz, x0t, iwt, cells, pz)
        if ix > Int32(0) && iy > Int32(0) && iz > Int32(0)
            bx = one(E) - ax; by = one(E) - ay; bz = one(E) - az
            Atomix.@atomic ρ[ix, iy, iz]         += ax * ay * az
            Atomix.@atomic ρ[ix, iy, iz+1]       += ax * ay * bz
            Atomix.@atomic ρ[ix, iy+1, iz]       += ax * by * az
            Atomix.@atomic ρ[ix, iy+1, iz+1]     += ax * by * bz
            Atomix.@atomic ρ[ix+1, iy, iz]       += bx * ay * az
            Atomix.@atomic ρ[ix+1, iy, iz+1]     += bx * ay * bz
            Atomix.@atomic ρ[ix+1, iy+1, iz]     += bx * by * az
            Atomix.@atomic ρ[ix+1, iy+1, iz+1]   += bx * by * bz
        end
    end
end
