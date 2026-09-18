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
               total_charge, uniform_sphere_potential, contract_spline_10,
               gaussian_force_kernel, DualBuffer, dual_buffer

"""
Zero-copy buffers on Apple Silicon — the whole of what this backend has to say
about the generic accelerator.

`Metal.SharedStorage` places the array in memory that both the CPU and the GPU
address, so `unsafe_wrap` hands back an `Array` over the very same bytes.
[`Vlasov.upload!`](@ref) and [`Vlasov.download!`](@ref) then do nothing, which
is the point: on unified memory the transfers the generic path would otherwise
perform are pure waste.
"""
function Vlasov.dual_buffer(::Metal.MetalBackend, ::Type{T},
                            dims::Integer...) where {T}
    mtl = MtlArray{T,length(dims),Metal.SharedStorage}(undef, dims...)
    host = unsafe_wrap(Array, mtl)
    fill!(host, zero(T))
    DualBuffer(mtl, host, true)
end

"""
    SharedBuffer{T,N}

A unified-memory buffer on Apple Silicon allocated with `Metal.SharedStorage`.
Shared without copying between CPU and GPU:
- `buf.mtl` is passed to Metal kernels;
- `buf.cpu` provides an in-place `Array{T,N}` view for direct CPU read/write.
Both point to the exact same physical memory.
"""
struct SharedBuffer{T,N}
    mtl::MtlArray{T,N,Metal.SharedStorage}
    cpu::Array{T,N}
end

function SharedBuffer(::Type{T}, dims::Integer...) where {T}
    mtl = MtlArray{T,length(dims),Metal.SharedStorage}(undef, dims...)
    cpu = unsafe_wrap(Array, mtl)
    fill!(cpu, zero(T))
    SharedBuffer(mtl, cpu)
end

Base.fill!(b::SharedBuffer, val) = fill!(b.cpu, val)

"""
Tables and buffers resident on the GPU.

All buffers exchanged at each step (`csol`, `force`, `knode`, `delta`, `cols`,
`rho`, `reduction`) live in `SharedBuffer`: on Apple Silicon's unified memory,
CPU and GPU share the same physical RAM. There is zero copying between host and device.
The constant tables (`overlap`, `gradient`, `nodes`) stay in GPU private storage.
"""
struct MetalForceAccelerator <: ForceAccelerator
    # Shared unified memory buffers (zero copy)
    csol::SharedBuffer{Float32,3}
    force::SharedBuffer{Float32,2}
    knode::SharedBuffer{Int32,2}
    delta::SharedBuffer{Float32,2}
    cols::SharedBuffer{Int32,2}
    rho::SharedBuffer{Float32,3}
    reduction::SharedBuffer{Float32,1}
    cells::Base.RefValue{MtlVector{Int32,Metal.SharedStorage}}
    bounds::Base.RefValue{MtlVector{Int32,Metal.SharedStorage}}

    # Constant GPU tables (private storage)
    overlap::MtlArray{Float32,2}
    gradient::MtlArray{Float32,2}
    nodes::MtlArray{Float32,2}

    # Sorter & Grid parameters
    sorter::CellSort{Float64}
    x0::Float32                   # first knot of the fine grid
    h::Float32                    # step (uniform grid)
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

"""
    ForceAccelerator(MtlArray, fine, smoothing, npart, n)

The entry point the scripts already name. It now builds the backend-agnostic
[`Vlasov.DeviceAccelerator`](@ref) on `MetalBackend()`, in `Float32` — Apple
GPUs having no double precision — so the scripts change nothing and gain the
portable kernels.
"""
Vlasov.ForceAccelerator(::Type{MtlArray}, fine::NTuple{3,SplineAxis{T}},
                        sm::GaussianSmoothing{T}, npart::Integer,
                        n::Integer) where {T} =
    Vlasov.DeviceAccelerator(Metal.MetalBackend(), Float32, fine, sm, npart, n)

"""
Packs the positions in `(k, δ)` form directly into shared memory:
index of the nearest knot, and the offset from that knot.

`δ` is bounded by half a step (1.8 a₀): encoded in `Float32`, its resolution is
1.2e-07, thirty thousand times finer than a column. The subtraction happens
in `Float64`, and once only — both kernels use the result.
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

About 3000 operations for 4 KB read: the kernel is memory-bound.
Particles whose stencil overflows the grid write a `NaN` and are handed back
to the CPU.
"""
function _field_kernel!(force, csol, ovl, grad, knode, delta, x0, h,
                        spacing, nbdt, w, nc, npart,
                        px0, py0, pz0, coef, σ, red)
    i = thread_position_in_grid_1d()
    tid = thread_index_in_threadgroup()
    nthr = Int32(256)

    sh = MtlThreadGroupArray(Float32, 4 * 256)
    @inbounds for c in Int32(0):Int32(3)
        sh[c * nthr + tid] = 0.0f0
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

        fxp = NaN32; fyp = NaN32; fzp = NaN32
        if ok
            half = spacing * 0.5f0
            cx = min(max(floor(Int32, (dx0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)
            cy = min(max(floor(Int32, (dy0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)
            cz = min(max(floor(Int32, (dz0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)

            fx, fy, fz = contract_spline_10(csol, ovl, grad, bx, by, bz, cx, cy, cz)
            fxp = -w * fx; fyp = -w * fy; fzp = -w * fz
        end

        # --- projectile ↔ pseudo-electron, fused -----------------------------
        if coef != 0.0f0
            px = x0 + Float32(kx - Int32(1)) * h + dx0
            py = x0 + Float32(ky - Int32(1)) * h + dy0
            pz = x0 + Float32(kz - Int32(1)) * h + dz0
            dx = px0 - px; dy = py0 - py; dz = pz0 - pz
            d2 = dx * dx + dy * dy + dz * dz
            kf = gaussian_force_kernel(d2, σ)
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

"""Grows the cell buffers if the sort found more of them."""
function _ensure!(acc::MetalForceAccelerator, ncell::Integer)
    grow(m) = fill!(MtlArray{Int32,1,Metal.SharedStorage}(undef, m), Int32(0))
    length(acc.cells[]) < ncell && (acc.cells[] = grow(ncell))
    length(acc.bounds[]) < ncell + 1 && (acc.bounds[] = grow(ncell + 1))
    nothing
end

"""Table columns in sorted order, directly in shared memory."""
function _fill_columns!(acc::MetalForceAccelerator, mesh, sm, positions)
    knots = mesh.axes[1].knots
    half = sm.spacing / 2
    lo = knots[2] + half; hi = knots[end-1] - half
    sp = sm.spacing; nbdt = sm.nbdt; ncol = size(sm.nodes, 2)
    perm = acc.sorter.perm
    cols = acc.cols.cpu
    delta = acc.delta.cpu
    nout = Threads.Atomic{Int}(0)

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
    _pack_kd!(acc.knode.cpu, acc.delta.cpu, positions,
              acc.sorter.x0, acc.sorter.h, acc.sorter.nknots)
    cellsort!(acc.sorter, positions)
    nout = _fill_columns!(acc, mesh, sm, positions)

    ncell = length(acc.sorter.occupied)
    _ensure!(acc, ncell)
    copyto!(acc.cells[], 1:ncell, acc.sorter.occupied, 1:ncell)
    copyto!(acc.bounds[], 1:(ncell + 1), acc.sorter.bounds, 1:(ncell + 1))
    fill!(acc.rho.mtl, 0f0)

    Metal.@sync @metal threads=512 groups=ncell _deposit_kernel!(
        acc.rho.mtl, acc.nodes, acc.cols.mtl, acc.cells[], acc.bounds[],
        Int32(acc.sorter.nknots), Int32(ncell))

    ρ .= acc.rho.cpu
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

    acc.csol.cpu .= csol_fine
    if !packed
        _pack_kd!(acc.knode.cpu, acc.delta.cpu, cloud.positions,
                  acc.sorter.x0, acc.sorter.h, acc.sorter.nknots)
    end

    pp = projectile === nothing ? (0f0, 0f0, 0f0) : Float32.(projectile.position)
    coef = projectile === nothing ? 0f0 : Float32(-cloud.weight * projectile.charge)
    σ = projectile === nothing ? 1f0 : Float32(Vlasov.scale(projectile.softening))
    fill!(acc.reduction.mtl, 0f0)

    groupsize = 256
    Metal.@sync @metal threads = groupsize groups = cld(npart, groupsize) _field_kernel!(
        acc.force.mtl, acc.csol.mtl, acc.overlap, acc.gradient, acc.knode.mtl, acc.delta.mtl,
        acc.x0, acc.h, acc.spacing, acc.nbdt, w, acc.ncol, Int32(npart),
        pp[1], pp[2], pp[3], coef, σ, acc.reduction.mtl)

    n = 0
    ww = T(cloud.weight); w2 = ww * ww
    f_cpu = acc.force.cpu
    @inbounds for i in 1:npart
        if isnan(f_cpu[1, i])
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
            cloud.forces[i] = (T(f_cpu[1, i]), T(f_cpu[2, i]), T(f_cpu[3, i]))
        end
    end
    n
end

function Vlasov.projectile_forces!(cloud::ParticleCloud{T}, acc::MetalForceAccelerator,
                                   proj::Projectile{T}, jel::Jellium{T}) where {T}
    p = proj.position
    q = proj.charge
    r2 = sum(abs2, p)
    modf = r2 > jel.radius^2 ? jel.nions * q / r2^T(1.5) : q / Vlasov.WIGNER_SEITZ_NA^3
    fjel = modf .* p
    red = acc.reduction.cpu
    fel = ntuple(d -> T(red[d]), 3)
    (fjel .+ fel, T(cloud.weight) * q * T(red[4]),
     q * Vlasov.potential(jel, sqrt(r2)))
end

end # module
