"""
Metal backend: smoothed-field evaluation and charge deposition on Apple GPUs.

Loaded automatically as soon as `Metal` is. See `src/gpu.jl` for the contract,
and `docs/gpu.md` for what the measurements say about it.
"""
module VlasovMetalExt

using Vlasov
using Metal

import Vlasov: ForceAccelerator, forces!, deposit_smoothed!, projectile_forces!,
               smoothed_field, Projectile, GaussianSoftening, Jellium, erf,
               spline_field, nearest_knot, table_column, GaussianSmoothing,
               SplineAxis, SplineMesh, ParticleCloud, CellSort, cellsort!,
               total_charge, uniform_sphere_potential

"""
Tables and buffers resident on the GPU.

`csol` is rewritten at every step — the potential changes — while the tables
never are. The fine grid being **uniform**, the index of the nearest knot is
computed rather than searched for: the kernel has neither a bisection nor a
branch, which is exactly what a GPU asks for.
"""
struct MetalForceAccelerator <: ForceAccelerator
    csol::MtlArray{Float32,3}
    overlap::MtlArray{Float32,2}
    gradient::MtlArray{Float32,2}
    "Deposition table: the Gaussian at the eight neighbouring collocation points."
    nodes::MtlArray{Float32,2}
    force::MtlArray{Float32,2}
    "Reused host buffers: converting `csol` or packing the positions at every
     step would allocate several megabytes per step, and a garbage collection
     on arrival."
    hostcsol::Array{Float32,3}
    hostforce::Matrix{Float32}
    """Positions in `(k, δ)` form — see [`_pack_kd!`](@ref). It is this
    representation, and not the absolute position, that lets the GPU pick the
    right table column: `δ` is bounded by half a step, hence encoded in
    `Float32` thirty thousand times finer than a column."""
    hostknode::Matrix{Int32}
    hostdelta::Matrix{Float32}
    knode::MtlArray{Int32,2}
    delta::MtlArray{Float32,2}
    "Cell sort: permutation, occupied cells, bounds."
    sorter::CellSort{Float64}
    hostcols::Matrix{Int32}
    cols::MtlArray{Int32,2}
    """⚠️ Resized along the way: the number of occupied cells changes as the
    cluster evolves. Hence the `Ref`s, and `_ensure!`."""
    cells::Base.RefValue{MtlVector{Int32,Metal.SharedStorage}}
    bounds::Base.RefValue{MtlVector{Int32,Metal.SharedStorage}}
    rho::MtlArray{Float32,3}
    hostrho::Array{Float32,3}
    hostreduction::Vector{Float32}
    """Projectile reduction: the three components of the force it feels, then
    its interaction energy with the pseudo-electrons. Fused into the force
    kernel, it costs only a few operations on data already loaded — a second
    sweep over 800 000 particles would cost ten times more."""
    reduction::MtlArray{Float32,1}
    x0::Float32                   # first knot of the fine grid
    h::Float32                    # step (uniform grid)
    nknots::Int32
    spacing::Float32
    nbdt::Int32
    ncol::Int32
    npart::Int
end

"""Checks that an axis really is uniform — the kernel depends on it."""
function _uniform_step(ax::SplineAxis)
    k = ax.knots
    h = (k[end] - k[1]) / (length(k) - 1)
    maximum(abs, diff(k) .- h) <= 1e-9 * abs(h) ||
        throw(ArgumentError("the Metal backend assumes a uniform fine grid"))
    h
end

function Vlasov.ForceAccelerator(::Type{MtlArray}, fine::NTuple{3,SplineAxis{T}},
                                 sm::GaussianSmoothing{T}, npart::Integer,
                                 n::Integer) where {T}
    h = _uniform_step(fine[1])
    for d in 2:3
        isapprox(_uniform_step(fine[d]), h; rtol = 1e-12) ||
            throw(ArgumentError("the Metal backend assumes the three axes are identical"))
    end
    # ⚠️ The buffers **exchanged at every step** live in shared memory. By
    # default `MtlArray` allocates in `PrivateStorage`, visible to the GPU
    # alone, and `copyto!` then makes a real copy: measured, 1.30 ms for 9.2 MB
    # against **0.21 ms** when shared. On a unified-memory chip, paying for a
    # copy makes no sense. The constant tables, for their part, stay private.
    shared(T, dims...) = fill!(MtlArray{T,length(dims),Metal.SharedStorage}(undef, dims...),
                               zero(T))
    MetalForceAccelerator(
        shared(Float32, n, n, n),
        MtlArray(Float32.(sm.overlap)), MtlArray(Float32.(sm.gradient)),
        MtlArray(Float32.(sm.nodes)),
        shared(Float32, 3, npart),
        Array{Float32,3}(undef, n, n, n), Matrix{Float32}(undef, 3, npart),
        Matrix{Int32}(undef, 3, npart), Matrix{Float32}(undef, 3, npart),
        shared(Int32, 3, npart), shared(Float32, 3, npart),
        CellSort(fine[1], npart), Matrix{Int32}(undef, 3, npart),
        shared(Int32, 3, npart), Ref(shared(Int32, 1)),
        Ref(shared(Int32, 1)), shared(Float32, n, n, n),
        Array{Float32,3}(undef, n, n, n),
        Vector{Float32}(undef, 4), shared(Float32, 4),
        Float32(fine[1].knots[1]), Float32(h), Int32(length(fine[1].knots)),
        Float32(sm.spacing), Int32(sm.nbdt), Int32(size(sm.overlap, 2)), npart)
end

"""Packs the positions in `(k, δ)` form: index of the nearest knot, and the
**offset from that knot**.

⚠️ This is the point that decides the port's accuracy. A position reaches 78 a₀,
where the `Float32` ULP is 7.6e-06 — that is 0.22 % of the width of a table
column (0.00355 a₀). Forming `x − knot` on the GPU therefore flips one particle
in five hundred onto the neighbouring column, which is not a rounding error that
averages out but a **wrong Gaussian sample**.

`δ` is bounded by half a step, 1.8 a₀: encoded in `Float32`, its resolution is
1.2e-07, thirty thousand times finer than a column. The subtraction happens
here, in `Float64`, and once only — both kernels use the result.

The absolute position is reconstructed where needed as `x₀ + (k−1)h + δ`, which
loses nothing more than the old packing did.

Kept in its own function so that the loop is typed: written inline in `forces!`,
it would capture variables whose type is unknown at compile time.
"""
function _pack_kd!(knode::Matrix{Int32}, delta::Matrix{Float32},
                   src::Vector{NTuple{3,T}}, x0::T, h::T, nk::Int) where {T}
    Threads.@threads for i in eachindex(src)
        @inbounds begin
            p = src[i]
            for d in 1:3
                k = clamp(round(Int32, (p[d] - x0) / h) + Int32(1), Int32(1), Int32(nk))
                knode[d, i] = k
                delta[d, i] = Float32(p[d] - (x0 + (k - 1) * h))
            end
        end
    end
    nothing
end

"""
Kernel: one particle per thread, a 10×10×10 contraction against `csol`.

About 3000 operations for 4 KB read: the kernel is **memory-bound**, not
compute-bound. Two particles of the same cell read the same tile, whence the
value (not yet exploited here) of sorting the particles — the thesis's sort, for
the same reason as in 1997.

Particles whose stencil overflows the grid write a `NaN`: they are handed back
to the CPU. Signalling beats truncating in silence.
"""
function _field_kernel!(force, csol, ovl, grad, knode, delta, x0, h,
                        spacing, nbdt, w, nc, npart,
                        px0, py0, pz0, coef, σ, red)
    i = thread_position_in_grid_1d()
    tid = thread_index_in_threadgroup()
    nthr = Int32(256)

    # ⚠️ Threadgroup reduction buffer. Every thread must reach every barrier:
    # no early `return` in this kernel, only flags.
    sh = MtlThreadGroupArray(Float32, 4 * 256)
    @inbounds for c in Int32(0):Int32(3)
        sh[c * nthr + tid] = 0.0f0
    end

    @inbounds if i <= npart
        # `(k, δ)` come from the host, computed in `Float64`: the kernel never
        # forms `x − knot`, which would be its largest loss of precision.
        kx = knode[1, i]; ky = knode[2, i]; kz = knode[3, i]
        dx0 = delta[1, i]; dy0 = delta[2, i]; dz0 = delta[3, i]
        bx = Int32(2) * kx - Int32(5)
        by = Int32(2) * ky - Int32(5)
        bz = Int32(2) * kz - Int32(5)

        nn = Int32(size(csol, 1))
        ok = bx >= Int32(1) && by >= Int32(1) && bz >= Int32(1) &&
             bx + Int32(9) <= nn && by + Int32(9) <= nn && bz + Int32(9) <= nn

        fxp = NaN32; fyp = NaN32; fzp = NaN32
        if ok
            # The column depends only on `δ`, so it is **exact** here: no large
            # subtraction left anywhere.
            half = spacing * 0.5f0
            cx = min(max(floor(Int32, (dx0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)
            cy = min(max(floor(Int32, (dy0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)
            cz = min(max(floor(Int32, (dz0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)

            fx = 0.0f0; fy = 0.0f0; fz = 0.0f0
            for kk in Int32(1):Int32(10)
                k = bz + kk - Int32(1)
                oz = ovl[kk, cz]; gz = grad[kk, cz]
                for jj in Int32(1):Int32(10)
                    j = by + jj - Int32(1)
                    oy = ovl[jj, cy]; gy = grad[jj, cy]
                    dxp = 0.0f0; val = 0.0f0
                    for ii in Int32(1):Int32(10)
                        c = csol[bx + ii - Int32(1), j, k]
                        dxp = fma(c, grad[ii, cx], dxp)
                        val = fma(c, ovl[ii, cx], val)
                    end
                    fx = fma(oy * oz, dxp, fx)
                    fy = fma(gy * oz, val, fy)
                    fz = fma(oy * gz, val, fz)
                end
            end
            fxp = -w * fx; fyp = -w * fy; fzp = -w * fz
        end

        # --- projectile ↔ pseudo-electron, fused -----------------------------
        # Computed for **every** particle, including those the CPU will take
        # back: the reduction must count them. `coef = −weight·charge`.
        if coef != 0.0f0
            # Absolute position reconstructed as `x₀ + (k−1)h + δ`. It carries
            # the same precision as the old direct packing, and is used only
            # here — the columns no longer depend on it.
            px = x0 + Float32(kx - Int32(1)) * h + dx0
            py = x0 + Float32(ky - Int32(1)) * h + dy0
            pz = x0 + Float32(kz - Int32(1)) * h + dz0
            dx = px0 - px; dy = py0 - py; dz = pz0 - pz
            d2 = dx * dx + dy * dy + dz * dz
            u2 = d2 / (σ * σ)
            # Near zero the two terms of the Gaussian force cancel at leading
            # order: the series avoids the subtraction.
            kf = if u2 <= 0.25f0
                    q = 1.0f0 / 685440.0f0
                    q = -1.0f0 / 49920.0f0 + u2 * q
                    q =  1.0f0 / 4224.0f0  + u2 * q
                    q = -1.0f0 / 432.0f0   + u2 * q
                    q =  1.0f0 / 56.0f0    + u2 * q
                    q = -1.0f0 / 10.0f0    + u2 * q
                    q =  1.0f0 / 3.0f0     + u2 * q
                    0.7978845608f0 * q / (σ * σ * σ)
                else
                    r = sqrt(d2)
                    (erf(r / 1.4142135624f0 / σ) -
                     0.7978845608f0 * (r / σ) * exp(-u2 * 0.5f0)) / (d2 * r)
                end
            m = coef * kf
            fx2 = m * dx; fy2 = m * dy; fz2 = m * dz
            if ok
                fxp -= fx2; fyp -= fy2; fzp -= fz2       # reaction
            end
            r = sqrt(d2)
            sh[tid]            = fx2
            sh[nthr + tid]     = fy2
            sh[2 * nthr + tid] = fz2
            sh[3 * nthr + tid] = r < 1.0f-4 * σ ? 0.7978845608f0 / σ :
                                 erf(r / 1.4142135624f0 / σ) / r
        end

        force[1, i] = fxp; force[2, i] = fyp; force[3, i] = fzp
    end

    # Tree reduction, then a single atomic per group.
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    stride = nthr ÷ Int32(2)
    while stride > Int32(0)
        if tid <= stride
            @inbounds for c in Int32(0):Int32(3)
                sh[c * nthr + tid] += sh[c * nthr + tid + stride]
            end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        stride ÷= Int32(2)
    end
    if tid == Int32(1)
        @inbounds for c in Int32(0):Int32(3)
            Metal.@atomic red[c + Int32(1)] += sh[c * nthr + Int32(1)]
        end
    end
    nothing
end

"""
Deposition kernel, **sorted** version.

One group of 512 threads per occupied cell, and each thread owns **one** of the
512 points of the 8³ stencil. It sweeps every particle of the cell accumulating
into a register, and performs **one single** atomic addition at the end.

It is the loop inversion that pays: the naive route does 512 atomics per
particle, this one does one per stencil point per cell — a hundred times fewer,
since each occupied cell holds about a hundred particles.

The columns go through threadgroup memory, loaded in batches of 64: without that
each of the 512 threads would reread every particle's data.
"""
function _deposit_kernel!(ρ, nodes, cols, cellids, bounds, nk, ncell)
    g = threadgroup_position_in_grid_1d()
    g > ncell && return nothing
    t = thread_index_in_threadgroup()

    @inbounds begin
        c0 = cellids[g] - Int32(1)
        kx = c0 % nk + Int32(1); c0 ÷= nk
        ky = c0 % nk + Int32(1); c0 ÷= nk
        kz = c0 + Int32(1)
        bx = Int32(2) * kx - Int32(5); by = Int32(2) * ky - Int32(5); bz = Int32(2) * kz - Int32(5)

        t0 = t - Int32(1)
        ii = t0 % Int32(8) + Int32(1); t0 ÷= Int32(8)
        jj = t0 % Int32(8) + Int32(1); t0 ÷= Int32(8)
        kk = t0 + Int32(1)

        n = Int32(size(ρ, 1))
        inside = bx >= Int32(0) && by >= Int32(0) && bz >= Int32(0) &&
                 bx + Int32(8) <= n && by + Int32(8) <= n && bz + Int32(8) <= n

        shared = MtlThreadGroupArray(Int32, 3 * 64)
        lo = bounds[g]; hi = bounds[g + Int32(1)]
        acc = 0.0f0
        p = lo
        while p < hi
            m = min(Int32(64), hi - p)
            if t <= m
                b = 3 * (t - Int32(1))
                shared[b + Int32(1)] = cols[1, p + t]
                shared[b + Int32(2)] = cols[2, p + t]
                shared[b + Int32(3)] = cols[3, p + t]
            end
            threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            if inside
                for q in Int32(1):m
                    b = 3 * (q - Int32(1))
                    acc += nodes[ii, shared[b + Int32(1)]] *
                           nodes[jj, shared[b + Int32(2)]] *
                           nodes[kk, shared[b + Int32(3)]]
                end
            end
            threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            p += m
        end
        inside && acc != 0.0f0 && (Metal.@atomic ρ[bx + ii, by + jj, bz + kk] += acc)
    end
    nothing
end

"""Grows the cell buffers if the sort found more of them. We never shrink: the
size settles within a few steps."""
function _ensure!(acc::MetalForceAccelerator, ncell::Integer)
    grow(m) = fill!(MtlArray{Int32,1,Metal.SharedStorage}(undef, m), Int32(0))
    length(acc.cells[]) < ncell && (acc.cells[] = grow(ncell))
    length(acc.bounds[]) < ncell + 1 && (acc.bounds[] = grow(ncell + 1))
    nothing
end

"""Table columns in sorted order, and particles outside the domain.

Computed in `Float64` on the host, deliberately: see the `hostcols` field.
Particles outside the useful domain receive column 1 and are excluded from the
deposition by a zero weight — simpler than a separate list, and with no branch
in the kernel."""
function _fill_columns!(acc::MetalForceAccelerator, mesh, sm, positions)
    knots = mesh.axes[1].knots
    half = sm.spacing / 2
    lo = knots[2] + half; hi = knots[end-1] - half
    sp = sm.spacing; nbdt = sm.nbdt; ncol = size(sm.nodes, 2)
    perm = acc.sorter.perm
    cols = acc.hostcols
    delta = acc.hostdelta
    nout = Threads.Atomic{Int}(0)

    # `δ` has already been computed in `Float64` by `_pack_kd!`: all that is
    # left is to reread it in sorted order. The column depends on it alone.
    Threads.@threads for s in eachindex(perm)
        @inbounds begin
            i = perm[s]
            p = positions[i]
            if lo <= p[1] <= hi && lo <= p[2] <= hi && lo <= p[3] <= hi
                for d in 1:3
                    cols[d, s] = clamp(floor(Int32, (Float64(delta[d, i]) + half) /
                                             sp * nbdt + 0.5) + Int32(1),
                                       Int32(1), Int32(ncol))
                end
            else
                cols[1, s] = cols[2, s] = cols[3, s] = Int32(1)
                Threads.atomic_add!(nout, 1)
            end
        end
    end
    nout[]
end

function Vlasov.deposit_smoothed!(ρ::Array{T,3}, acc::MetalForceAccelerator,
                                  mesh::SplineMesh{3,T}, sm::GaussianSmoothing{T},
                                  positions; charge::T) where {T}
    # The deposition opens the step: it is the one that packs `(k, δ)`, which
    # `forces!` will then reuse.
    _pack_kd!(acc.hostknode, acc.hostdelta, positions,
              acc.sorter.x0, acc.sorter.h, acc.sorter.nknots)
    copyto!(acc.knode, acc.hostknode)
    copyto!(acc.delta, acc.hostdelta)
    cellsort!(acc.sorter, positions)
    nout = _fill_columns!(acc, mesh, sm, positions)

    ncell = length(acc.sorter.occupied)
    _ensure!(acc, ncell)
    copyto!(acc.cols, acc.hostcols)
    copyto!(acc.cells[], 1:ncell, acc.sorter.occupied, 1:ncell)
    copyto!(acc.bounds[], 1:(ncell + 1), acc.sorter.bounds, 1:(ncell + 1))
    fill!(acc.rho, 0f0)

    Metal.@sync @metal threads=512 groups=ncell _deposit_kernel!(
        acc.rho, acc.nodes, acc.cols, acc.cells[], acc.bounds[],
        Int32(acc.sorter.nknots), Int32(ncell))

    copyto!(acc.hostrho, acc.rho)

    # ⚠️ **A single pass** over the 729 000 points. The literal form — convert,
    # multiply by the charge, then renormalise — makes three of them, and the
    # three cost more than the GPU kernel they follow.
    #
    # The two rescalings compose: `total_charge` is linear, so multiplying by
    # `charge` then renormalising amounts to a single multiplication, whose
    # factor is computed on the raw density.
    ρ .= acc.hostrho
    q = total_charge(ρ, mesh)
    ρ .*= (length(positions) - nout) * charge / q
    nout
end

function Vlasov.forces!(cloud::ParticleCloud{T}, acc::MetalForceAccelerator,
                        fine::NTuple{3,SplineAxis{T}}, csol_fine::Array{T,3},
                        coarse::NTuple{3,SplineAxis{T}}, csol_coarse::Array{T,3},
                        sm::GaussianSmoothing{T}; escaped::Integer = 0,
                        projectile = nothing, packed::Bool = false) where {T}
    npart = length(cloud.positions)
    npart == acc.npart || throw(DimensionMismatch("accelerator sized for $(acc.npart)"))
    w = Float32(cloud.weight)

    acc.hostcsol .= csol_fine
    copyto!(acc.csol, acc.hostcsol)
    # `packed = true` says the deposition has just done it for the same
    # positions — which is the case in `update_forces!`, where it opens the
    # step. Redoing the packing would cost two milliseconds for nothing.
    if !packed
        _pack_kd!(acc.hostknode, acc.hostdelta, cloud.positions,
                  acc.sorter.x0, acc.sorter.h, acc.sorter.nknots)
        copyto!(acc.knode, acc.hostknode)
        copyto!(acc.delta, acc.hostdelta)
    end

    # The projectile is fused into this kernel: its arguments are zero when
    # there is none, and the branch disappears.
    pp = projectile === nothing ? (0f0, 0f0, 0f0) : Float32.(projectile.position)
    coef = projectile === nothing ? 0f0 : Float32(-cloud.weight * projectile.charge)
    σ = projectile === nothing ? 1f0 : Float32(Vlasov.scale(projectile.softening))
    fill!(acc.reduction, 0f0)

    groupsize = 256
    Metal.@sync @metal threads = groupsize groups = cld(npart, groupsize) _field_kernel!(
        acc.force, acc.csol, acc.overlap, acc.gradient, acc.knode, acc.delta,
        acc.x0, acc.h, acc.spacing, acc.nbdt, w, acc.ncol, Int32(npart),
        pp[1], pp[2], pp[3], coef, σ, acc.reduction)

    copyto!(acc.hostforce, acc.force)
    copyto!(acc.hostreduction, acc.reduction)

    # CPU fallback for what the GPU refused — the boundaries, and those only.
    # ⚠️ The **projectile reaction must be reinjected** there: the kernel
    # counted it in the reduction but could not add it to a force it did not
    # compute.
    n = 0
    ww = T(cloud.weight); w2 = ww * ww
    @inbounds for i in 1:npart
        if isnan(acc.hostforce[1, i])
            n += 1
            p = cloud.positions[i]
            E = spline_field(coarse, csol_coarse, p)
            f = if E === nothing
                r3 = (p[1]^2 + p[2]^2 + p[3]^2)^T(1.5)
                (-w2 * escaped / r3) .* p
            else
                ww .* E
            end
            if projectile !== nothing
                d = projectile.position .- p
                m = -cloud.weight * projectile.charge *
                    Vlasov.force_kernel(projectile.softening, sum(abs2, d))
                f = f .- m .* d
            end
            cloud.forces[i] = f
        else
            cloud.forces[i] = (T(acc.hostforce[1, i]), T(acc.hostforce[2, i]),
                               T(acc.hostforce[3, i]))
        end
    end
    n
end

"""
    projectile_forces!(cloud, acc, proj, jel) -> (force, e_electrons, e_jellium)

Computes **nothing** over the particles: the sum has already been done by the
force kernel, which fuses it rather than opening a second sweep over 800 000
particles. All that is left here is the jellium part, which is a scalar.

⚠️ Assumes therefore that [`forces!`](@ref) has just been called on the **same**
accelerator, with this projectile.
"""
function Vlasov.projectile_forces!(cloud::ParticleCloud{T}, acc::MetalForceAccelerator,
                                   proj::Projectile{T}, jel::Jellium{T}) where {T}
    p = proj.position
    q = proj.charge
    r2 = sum(abs2, p)
    modf = r2 > jel.radius^2 ? jel.nions * q / r2^T(1.5) : q / Vlasov.WIGNER_SEITZ_NA^3
    fjel = modf .* p
    fel = ntuple(d -> T(acc.hostreduction[d]), 3)
    (fjel .+ fel, T(cloud.weight) * q * T(acc.hostreduction[4]),
     q * Vlasov.potential(jel, sqrt(r2)))
end

end # module
