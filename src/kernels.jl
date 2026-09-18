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

"""Work-items per group for [`_smoothed_field_kernel!`](@ref)."""
const FIELD_GROUPSIZE = 256

"""Work-items per group for [`_deposit_sorted_kernel!`](@ref): the 8³ stencil."""
const DEPOSIT_GROUPSIZE = 512

"""Particles staged through threadgroup memory at a time, in the deposition."""
const DEPOSIT_STAGE = 64

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
@kernel function _pack_kd_kernel!(knode, delta, @Const(positions), x0::T, h::T,
                                  nk) where {T}
    i = @index(Global, Linear)
    @inbounds begin
        p = positions[i]
        for d in 1:3
            k = clamp(round(Int32, (p[d] - x0) / h) + Int32(1), Int32(1), nk)
            knode[d, i] = k
            delta[d, i] = eltype(delta)(p[d] - (x0 + (k - Int32(1)) * h))
        end
    end
end

"""
Smoothed field at each pseudo-particle, with the projectile fused in.

One work-item per particle, a 10×10×10 contraction against `csol` — about 3000
operations for 4 KB read, so the kernel is memory-bound and there is nothing to
gain by rearranging the arithmetic.

Particles whose stencil overflows the grid write a `NaN` and are handed back to
the caller: they are rare, and branching on them here is exactly what one does
not want in a kernel.

The projectile's reaction is accumulated in threadgroup memory, reduced in a
tree, and committed with **one** atomic per group rather than one per particle.

⚠️ `unsafe_indices = true`: the tree reduction below calls `@synchronize`, which
every work-item of a group must reach. KernelAbstractions' automatic bounds
check would wrap the body in a conditional and make that false, so the guard is
written by hand and `ndrange` is padded to a whole number of groups.
"""
@kernel unsafe_indices = true function _smoothed_field_kernel!(
        force, @Const(csol), @Const(ovl), @Const(grad), @Const(knode),
        @Const(delta), x0, h, spacing, nbdt, w, nc, npart,
        px0, py0, pz0, coef, σ, red)
    i = @index(Global, Linear)
    tid = @index(Local, Linear)
    @uniform GS = Int32(FIELD_GROUPSIZE)
    @uniform E = eltype(force)
    # ⚠️ `@uniform`, and declared here rather than beside its loop: the CPU
    # backend cuts the kernel into one work-item loop per `@synchronize`, and a
    # plain local assigned in one segment does not exist in the next. The tree
    # reduction's counter crosses every barrier, so it has to live outside them.
    @uniform redhalf = Int32(FIELD_GROUPSIZE) ÷ Int32(2)
    sh = @localmem eltype(force) (4 * FIELD_GROUPSIZE,)

    for c in Int32(0):Int32(3)
        @inbounds sh[c*GS+tid] = zero(E)
    end

    @inbounds if i <= npart
        kx = knode[1, i]; ky = knode[2, i]; kz = knode[3, i]
        dx0 = delta[1, i]; dy0 = delta[2, i]; dz0 = delta[3, i]
        bx = Int32(2) * kx - Int32(5)
        by = Int32(2) * ky - Int32(5)
        bz = Int32(2) * kz - Int32(5)

        nn = Int32(size(csol, 1))
        ok = bx >= Int32(1) && by >= Int32(1) && bz >= Int32(1) &&
             bx + Int32(9) <= nn && by + Int32(9) <= nn && bz + Int32(9) <= nn

        nan = E(NaN)
        fxp = nan; fyp = nan; fzp = nan
        if ok
            halfsp = spacing * E(0.5)
            cx = min(max(floor(Int32, (dx0 + halfsp) / spacing * nbdt + E(0.5)) +
                         Int32(1), Int32(1)), nc)
            cy = min(max(floor(Int32, (dy0 + halfsp) / spacing * nbdt + E(0.5)) +
                         Int32(1), Int32(1)), nc)
            cz = min(max(floor(Int32, (dz0 + halfsp) / spacing * nbdt + E(0.5)) +
                         Int32(1), Int32(1)), nc)
            fx, fy, fz = contract_spline_10(csol, ovl, grad, bx, by, bz, cx, cy, cz)
            fxp = -w * fx; fyp = -w * fy; fzp = -w * fz
        end

        # --- projectile ↔ pseudo-electron, fused ----------------------------
        if coef != zero(E)
            px = x0 + E(kx - Int32(1)) * h + dx0
            py = x0 + E(ky - Int32(1)) * h + dy0
            pz = x0 + E(kz - Int32(1)) * h + dz0
            dx = px0 - px; dy = py0 - py; dz = pz0 - pz
            d2 = dx * dx + dy * dy + dz * dz
            m = coef * gaussian_force_kernel(d2, σ)
            fx2 = m * dx; fy2 = m * dy; fz2 = m * dz
            if ok
                fxp -= fx2; fyp -= fy2; fzp -= fz2       # reaction
            end
            r = sqrt(d2)
            sh[tid] = fx2
            sh[GS+tid] = fy2
            sh[2*GS+tid] = fz2
            sh[3*GS+tid] = r < E(1.0e-4) * σ ? E(0.7978845608) / σ :
                           erf(r / E(1.4142135624) / σ) / r
        end

        force[1, i] = fxp; force[2, i] = fyp; force[3, i] = fzp
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
from global memory rather than 512 times. That staging **earns its keep** —
unlike the field kernel's, which could be dropped for nothing. Measured on
Metal, 4×10⁶ particles on a 134³ grid:

| | ms |
|---|---|
| hand-written Metal, staged | 42.7 |
| this kernel, staged | 44.4 |
| this kernel, no staging | 53.6 |

So portability costs 4 % here, and dropping the staging would cost 21 % more.
The residual 4 % buys CUDA, ROCm and oneAPI, and a version that also runs on
`CPU()` — where it agrees with `deposit_smoothed!` to 9.3e-16 in `Float64`,
which is the check Metal can never perform.

⚠️ **What crosses a barrier must live in threadgroup memory or be `@uniform`.**
The CPU backend cuts a kernel into one work-item loop per `@synchronize`, and a
plain local assigned before a barrier is simply undefined after it. So the
per-item accumulator sits in `@localmem` rather than in a register, and the
stencil offsets are **recomputed** on each side of a barrier — they are three
integer divisions, cheaper than the machinery needed to carry them across.
"""
@kernel unsafe_indices = true function _deposit_sorted_kernel!(
        ρ, @Const(nodes), @Const(cols), @Const(cellids), @Const(bounds), nk)
    @uniform E = eltype(ρ)
    t = @index(Local, Linear)

    shared = @localmem Int32 (3 * DEPOSIT_STAGE,)
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
                b = Int32(3) * (t - Int32(1))
                shared[b+Int32(1)] = cols[1, cursor+t]
                shared[b+Int32(2)] = cols[2, cursor+t]
                shared[b+Int32(3)] = cols[3, cursor+t]
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
                for q in Int32(1):chunk
                    b = Int32(3) * (q - Int32(1))
                    s += nodes[ii, shared[b+Int32(1)]] *
                         nodes[jj, shared[b+Int32(2)]] *
                         nodes[kk, shared[b+Int32(3)]]
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

One work-item per particle of the sorted order. The position is **rebuilt** from
its packed form, `p = x₀ + (k−1)h + δ`, rather than read from the cloud: that is
what lets this run on the device without the positions having to live there too.

The rebuild happens in `E`, so where `E` is `Float32` the boundary test and the
column rounding shift by a few ulp, and particles on the very edge could in
principle fall on the other side. Measured rather than feared: on 2×10⁶
particles deliberately banked against the edge of the fine grid, Metal and the
`Float64` host reference reject **the same 136 140** — not one reclassified.
The density then agrees to 4.3e-05, which is the atomic ordering and not this.

Particles outside are parked on column 1 — they deposit nothing, the kernel
skips them — and counted into `nout`, one atomic each. They are rare.
"""
@kernel function _columns_kernel!(cols, @Const(knode), @Const(delta), @Const(perm),
                                  x0, h, lo, hi, spacing, nbdt, ncol, nout)
    s = @index(Global, Linear)
    @inbounds begin
        i = perm[s]
        E = eltype(delta)
        half = spacing * E(0.5)
        inside = true
        for d in Int32(1):Int32(3)
            p = x0 + E(knode[d, i] - Int32(1)) * h + delta[d, i]
            inside &= (lo <= p) & (p <= hi)
        end
        if inside
            for d in Int32(1):Int32(3)
                cols[d, s] = min(max(floor(Int32, (delta[d, i] + half) / spacing *
                                           nbdt + E(0.5)) + Int32(1),
                                     Int32(1)), ncol)
            end
        else
            cols[1, s] = Int32(1); cols[2, s] = Int32(1); cols[3, s] = Int32(1)
            Atomix.@atomic nout[1] += Int32(1)
        end
    end
end
