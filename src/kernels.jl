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
        @Const(delta), @Const(perm), x0, h, spacing, nbdt, w, nc, npart,
        px0, py0, pz0, coef, σ, red)
    t = @index(Global, Linear)
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

    @inbounds if t <= npart
        # ⚠️ Work-items walk the particles in **sorted** order. Each reads a 10³
        # stencil of `csol` — 4 KB — and in cell order the neighbours of a
        # work-item read very nearly the same 4 KB, which the cache then serves
        # once instead of once per particle. The price is that `knode` and
        # `delta` become a gather: 24 bytes against the 4 KB it protects.
        i = perm[t]
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
        ρ, @Const(nodes), @Const(cols), @Const(perm), @Const(cellids),
        @Const(bounds), nk)
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
                # The sort lives here, in the staging: one gather of 12 bytes
                # per particle per cell, overlapped with the inner loop below —
                # rather than a pass of its own over every particle.
                q = perm[cursor+t]
                b = Int32(3) * (t - Int32(1))
                shared[b+Int32(1)] = cols[1, q]
                shared[b+Int32(2)] = cols[2, q]
                shared[b+Int32(3)] = cols[3, q]
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
@kernel function _columns_kernel!(cols, @Const(knode), @Const(delta),
                                  x0, h, lo, hi, spacing, nbdt, ncol, nout)
    i = @index(Global, Linear)
    @inbounds begin
        E = eltype(delta)
        half = spacing * E(0.5)
        inside = true
        for d in Int32(1):Int32(3)
            p = x0 + E(knode[d, i] - Int32(1)) * h + delta[d, i]
            inside &= (lo <= p) & (p <= hi)
        end
        if inside
            for d in Int32(1):Int32(3)
                cols[d, i] = min(max(floor(Int32, (delta[d, i] + half) / spacing *
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
@kernel function _outside_kernel!(outlist, outcount, @Const(knode), n, npart)
    i = @index(Global, Linear)
    @inbounds if i <= npart
        # ⚠️ The same test, spelled the same way, as in the field kernel. The
        # two must agree exactly: one decides what to skip, the other what to
        # pick up.
        bx = Int32(2) * knode[1, i] - Int32(5)
        by = Int32(2) * knode[2, i] - Int32(5)
        bz = Int32(2) * knode[3, i] - Int32(5)
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
@kernel function _deposit_cic_kernel!(ρ, @Const(knode), @Const(delta),
                                      @Const(gx), @Const(gy), @Const(gz),
                                      @Const(cells), x0f, hf, x0t, iwt, npart)
    i = @index(Global, Linear)
    @inbounds if i <= npart
        E = eltype(ρ)
        px = x0f + E(knode[1, i] - Int32(1)) * hf + delta[1, i]
        py = x0f + E(knode[2, i] - Int32(1)) * hf + delta[2, i]
        pz = x0f + E(knode[3, i] - Int32(1)) * hf + delta[3, i]
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
