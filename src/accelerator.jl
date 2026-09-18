"""
A generic [`ForceAccelerator`](@ref), built on the portable kernels of
`kernels.jl`.

It is what the Metal extension used to be, minus the Metal: the same staging,
the same sorted deposition, the same fused projectile — expressed once, for any
backend `KernelAbstractions` supports.

One thing stays backend-specific, and by design: **where the buffers live**. See
[`DualBuffer`](@ref).
"""

"""
    DualBuffer

A buffer the host writes and the device reads, or the reverse.

On hardware where the two share memory — Apple Silicon's unified RAM — `host`
and `device` are *the same bytes*, and [`upload!`](@ref)/[`download!`](@ref) do
nothing. On a discrete GPU they are two allocations and the transfers are real
copies.

That difference is hardware, not convenience, which is why it gets an extension
point rather than being flattened: allocating uniformly would cost Apple Silicon
a copy of every buffer at every step, on the very machine the campaigns run on.
Call sites are identical either way; only the price changes.
"""
struct DualBuffer{T,N,D<:AbstractArray{T,N},H<:AbstractArray{T,N}}
    device::D
    host::H
    "`true` when `host` and `device` address the same memory."
    shared::Bool
end

"""
    dual_buffer(backend, T, dims...) -> DualBuffer

Allocates a [`DualBuffer`](@ref). The default gives a device array and a
separate host array — except on `CPU()`, where one array plays both parts. A
backend with unified memory overrides this; see `ext/VlasovMetalExt.jl`.
"""
function dual_buffer(backend, ::Type{T}, dims::Integer...) where {T}
    d = KernelAbstractions.zeros(backend, T, dims...)
    d isa Array ? DualBuffer(d, d, true) : DualBuffer(d, zeros(T, dims...), false)
end

"""Host → device. The copy is a no-op when the two are the same memory."""
upload!(b::DualBuffer) = (b.shared || copyto!(b.device, b.host); b)

"""
    download!(b, backend)

Device → host, and **a synchronisation point in every case**.

⚠️ The backend argument is not decoration. On unified memory the copy is a
no-op, so a `download!` that only copied would order nothing — and the host
would read buffers the device had not finished writing. That bug is silent and
looks like a wrong physical result: caught here as a density that integrated to
the raw particle count, the scaling kernel not having run yet.

So the synchronisation comes first and unconditionally, and the copy happens
only where the memories are distinct.
"""
function download!(b::DualBuffer, backend)
    synchronize(backend)
    b.shared || copyto!(b.host, b.device)
    b
end

Base.length(b::DualBuffer) = length(b.device)

"""
    DeviceGrid

Device-side companion to a [`SplineMesh`](@ref): the tables the kernels read,
copied once, in the kernels' own precision.

A mesh **describes a discretisation**; it is not itself computational data. It
holds banded factorisations, locator tables and an eigenbasis — none of which
belong on a GPU, and each of which would need a ruling of its own were the mesh
made adaptable wholesale. So the few tables the kernels actually touch are
copied here instead, and the mesh stays what it is.

It grows as functions move across: for now, what the charge reduction needs.
"""
struct DeviceGrid{E,V,M}
    "`∫φ` per direction, already carrying `S⁻ᵀ` — see `SplineMesh.dual_moments`."
    moments0::NTuple{3,V}
    "Scratch for the reduction: ten rows, one column per `(j,k)` pair."
    partials::M
end

"""
    DeviceGrid(backend, E, fine_axes, n)

Built from the axes, not from a mesh: the accelerator is handed axes, and the
two moment vectors it needs follow from [`dual_moments`](@ref). The collocation
matrices are rebuilt here to get them — a factorisation of a few hundred rows,
paid once at setup.
"""
function DeviceGrid(backend, ::Type{E}, fine::NTuple{3,SplineAxis{T}},
                    n::Integer) where {E,T}
    function dev(x)
        a = KernelAbstractions.zeros(backend, E, size(x)...)
        copyto!(a, E.(x))
        a
    end
    m0 = map(ax -> dev(dual_moments(CollocationMatrices(ax))[1]), fine)
    partials = dev(zeros(T, 10, n * n))
    DeviceGrid{E,eltype(m0),typeof(partials)}(m0, partials)
end

"""Charge of a density already on the device, read through its companion."""
total_charge(ρ::AbstractArray, g::DeviceGrid) =
    _total_charge(ρ, g.moments0, g.partials)

"""
Tables and buffers held for the whole run.

Everything is allocated **once**: allocations inside a time loop do not merely
cost their price, they trigger a collection that stops every thread.

`E` is the element type the kernels work in — `Float32` on Metal, which has no
double precision, and `Float64` wherever the hardware offers it.
"""
struct DeviceAccelerator{E,T,B,G,BC,BF,BR,BI,BU,BX,C} <: ForceAccelerator
    backend::B
    "Tables of the fine mesh, device-side — see [`DeviceGrid`](@ref)."
    grid::G
    csol::BC
    rho::BC
    force::BF
    delta::BF
    knode::BI
    cols::BI
    reduction::BR
    cells::BU
    bounds::BU
    "Sorted order, and the count of particles the deposition drops."
    perm::BU
    nout::BU
    """Particles whose stencil leaves the fine grid, compacted once per step.
       Shared by the forces and the energy budget — see
       [`_outside_kernel!`](@ref)."""
    outlist::BU
    outcount::BU
    "Constant tables, device-resident: they never change during a run."
    overlap::BX
    gradient::BX
    nodes::BX
    sorter::C
    x0::T
    h::T
    spacing::E
    nbdt::Int32
    ncol::Int32
    npart::Int
end

"""Checks that an axis really is uniform — the kernels depend on it."""
function _uniform_step(ax::SplineAxis)
    k = ax.knots
    h = (k[end] - k[1]) / (length(k) - 1)
    maximum(abs, diff(k) .- h) <= 1e-9 * abs(h) ||
        throw(ArgumentError("the accelerated path assumes a uniform fine grid"))
    h
end

"""
    DeviceAccelerator(backend, E, fine_axes, smoothing, npart, n)

Prepares the accelerated path for one grid and one particle count.

`E` is the working precision. ⚠️ The **packing** does not run in `E`: the offset
`p − knot` cancels its leading digits and is computed in `T` before being
narrowed. It therefore runs on `CPU()`, over the host face of the buffers — free
where the memory is unified, one upload where it is not. It moves to the device
on the day the cloud itself lives there, and not before.
"""
function DeviceAccelerator(backend, ::Type{E}, fine::NTuple{3,SplineAxis{T}},
                           sm::GaussianSmoothing{T}, npart::Integer,
                           n::Integer) where {E,T}
    h = _uniform_step(fine[1])
    for d in 2:3
        isapprox(_uniform_step(fine[d]), h; rtol = 1e-12) ||
            throw(ArgumentError("the accelerated path assumes three identical axes"))
    end

    function dev(x)
        a = KernelAbstractions.zeros(backend, E, size(x)...)
        copyto!(a, E.(x))
        a
    end
    DeviceAccelerator(
        backend,
        DeviceGrid(backend, E, fine, n),
        dual_buffer(backend, E, n, n, n),      # csol
        dual_buffer(backend, E, n, n, n),      # rho
        dual_buffer(backend, E, 3, npart),     # force
        dual_buffer(backend, E, 3, npart),     # delta
        dual_buffer(backend, Int32, 3, npart), # knode
        dual_buffer(backend, Int32, 3, npart), # cols
        dual_buffer(backend, E, 4),            # reduction
        # Sized for the worst case — one occupied cell per cell of the grid —
        # rather than grown on demand: the sort already pays for `nk³` counters,
        # so this adds nothing anyone was not already spending, and it spares a
        # `Ref` in the hot path.
        dual_buffer(backend, Int32, length(fine[1].knots)^3),
        dual_buffer(backend, Int32, length(fine[1].knots)^3 + 1),
        dual_buffer(backend, Int32, npart),    # perm
        dual_buffer(backend, Int32, 1),        # nout
        dual_buffer(backend, Int32, npart),    # outlist
        dual_buffer(backend, Int32, 1),        # outcount
        dev(sm.overlap), dev(sm.gradient), dev(sm.nodes),
        CellSort(fine[1], npart),
        T(fine[1].knots[1]), T(h), E(sm.spacing), Int32(sm.nbdt),
        Int32(size(sm.overlap, 2)), Int(npart))
end

"""
Packs the positions into `(k, δ)`, orders the particles by cell, and builds the
out-of-stencil list — everything that depends on **where the particles are** and
on nothing else.

⚠️ The sort belongs here and not in the deposition, although the deposition is
what first needed it: the force kernel now walks the particles in that order
too, and a `forces!` called on its own would otherwise read a stale `perm`.
"""
function _pack!(acc::DeviceAccelerator{E,T}, positions) where {E,T}
    cpu = KernelAbstractions.CPU()
    _pack_kd_kernel!(cpu)(acc.knode.host, acc.delta.host, positions,
                          acc.x0, acc.h, Int32(acc.sorter.nknots);
                          ndrange = length(positions))
    synchronize(cpu)
    upload!(acc.knode); upload!(acc.delta)

    cellsort!(acc.sorter, positions)
    copyto!(acc.perm.host, 1, acc.sorter.perm, 1, acc.npart)
    upload!(acc.perm)

    # The out-of-stencil list, built here because it depends only on where the
    # particles are. Both the forces and the energy budget read it, and the
    # budget runs first.
    fill!(acc.outcount.device, Int32(0))
    _outside_kernel!(acc.backend)(acc.outlist.device, acc.outcount.device,
                                  acc.knode.device, Int32(size(acc.csol.device, 1)),
                                  Int32(acc.npart); ndrange = acc.npart)
    download!(acc.outcount, acc.backend)
    download!(acc.outlist, acc.backend)
    Int(acc.outcount.host[1])
end

"""
Table columns in sorted order — on the device.

This used to be a threaded host loop, and it was the **largest** host item of
the deposition: 5.42 ms against 3.39 for the sort and 1.71 for the packing, at
2×10⁶ particles on Metal. It is here now because it need not have been there:
it read `δ` through a `Float64` conversion that recovered nothing, `δ` having
already been narrowed to `E` by the packing.

`perm` comes from [`_pack!`](@ref), which sorts.
"""
function _fill_columns!(acc::DeviceAccelerator{E,T}, mesh, sm, positions) where {E,T}
    knots = mesh.axes[1].knots
    half = sm.spacing / 2
    fill!(acc.nout.device, Int32(0))

    _columns_kernel!(acc.backend)(
        acc.cols.device, acc.knode.device, acc.delta.device,
        E(acc.x0), E(acc.h), E(knots[2] + half), E(knots[end-1] - half),
        acc.spacing, acc.nbdt, acc.ncol, acc.nout.device; ndrange = acc.npart)
    synchronize(acc.backend)

    download!(acc.nout, acc.backend)
    Int(acc.nout.host[1])
end

function deposit_smoothed!(ρ::AbstractArray, acc::DeviceAccelerator{E,T},
                           mesh::SplineMesh{3,T}, sm::GaussianSmoothing{T},
                           positions; charge::T) where {E,T}
    _pack!(acc, positions)          # packs, sorts, and lists the boundary cases
    nout = _fill_columns!(acc, mesh, sm, positions)

    ncell = length(acc.sorter.occupied)
    copyto!(acc.cells.host, 1, acc.sorter.occupied, 1, ncell)
    copyto!(acc.bounds.host, 1, acc.sorter.bounds, 1, ncell + 1)
    upload!(acc.cells); upload!(acc.bounds)

    # A `ρ` that already lives on this backend is deposited into **directly**:
    # on the resident path the density never leaves the device, and the staging
    # buffer is not touched at all.
    resident = get_backend(ρ) === acc.backend && eltype(ρ) === E
    target = resident ? ρ : acc.rho.device
    fill!(target, zero(E))

    _deposit_sorted_kernel!(acc.backend, DEPOSIT_GROUPSIZE)(
        target, acc.nodes, acc.cols.device, acc.perm.device, acc.cells.device,
        acc.bounds.device, Int32(acc.sorter.nknots); ndrange = ncell * DEPOSIT_GROUPSIZE)
    synchronize(acc.backend)

    # Reduction and scaling stay on the device. What came back to the host
    # before was the reduction plus two full passes over the n³ grid — 4.0 ms
    # of the 33 the deposition takes, at 2×10⁶ particles on Metal.
    q = total_charge(target, acc.grid)
    target .*= E((length(positions) - nout) * charge / q)
    if resident
        synchronize(acc.backend)
    else
        download!(acc.rho, acc.backend)
        ρ .= acc.rho.host
    end
    nout
end

function forces!(cloud::ParticleCloud{T}, acc::DeviceAccelerator{E,T},
                 fine::NTuple{3,SplineAxis{T}}, csol_fine::AbstractArray,
                 coarse::NTuple{3,SplineAxis{T}}, csol_coarse::AbstractArray,
                 sm::GaussianSmoothing{T}; escaped::Integer = 0,
                 projectile = nothing, packed::Bool = false) where {E,T}
    npart = length(cloud.positions)
    npart == acc.npart || throw(DimensionMismatch("accelerator sized for $(acc.npart)"))

    # A `csol` already sitting on this backend, in this precision, is used where
    # it is. Staging it through the host would be a copy of `n³` for nothing —
    # and on the resident path it is exactly what `spline_coefficients!` just
    # produced there.
    csol = if get_backend(csol_fine) === acc.backend && eltype(csol_fine) === E
        csol_fine
    else
        acc.csol.host .= csol_fine
        upload!(acc.csol)
        acc.csol.device
    end
    packed || _pack!(acc, cloud.positions)

    pp = projectile === nothing ? (zero(E), zero(E), zero(E)) :
         E.(projectile.position)
    coef = projectile === nothing ? zero(E) :
           E(-cloud.weight * projectile.charge)
    σ = projectile === nothing ? one(E) : E(scale(projectile.softening))
    fill!(acc.reduction.device, zero(E))

    _smoothed_field_kernel!(acc.backend, FIELD_GROUPSIZE)(
        acc.force.device, csol, acc.overlap, acc.gradient,
        acc.knode.device, acc.delta.device, acc.perm.device,
        E(acc.x0), E(acc.h), acc.spacing,
        acc.nbdt, E(cloud.weight), acc.ncol, Int32(npart),
        pp[1], pp[2], pp[3], coef, σ, acc.reduction.device;
        ndrange = cld(npart, FIELD_GROUPSIZE) * FIELD_GROUPSIZE)
    synchronize(acc.backend)
    download!(acc.force, acc.backend); download!(acc.reduction, acc.backend)

    # ⚠️ Two passes, not one branchy loop over every particle. The conversion
    # `E → T` has to touch all of them — `cloud.forces` is a host vector until
    # the cloud itself moves across — but with the boundary cases no longer
    # interleaved it becomes a flat loop, and a flat loop can be **threaded**.
    # The one it replaces was serial.
    #
    # Measured: broadcasting through `reinterpret(reshape, …)` instead looks
    # tidier and costs ×1.6 — the reinterpreted view does not vectorise the way
    # a direct store of tuples does.
    w = cloud.weight
    w2 = w * w
    f = acc.force.host
    tforeach(npart) do slice
        @inbounds for i in slice
            cloud.forces[i] = (T(f[1, i]), T(f[2, i]), T(f[3, i]))
        end
    end

    n = Int(acc.outcount.host[1])
    @inbounds for s in 1:n
        i = Int(acc.outlist.host[s])
        p = cloud.positions[i]
        Ec = spline_field(coarse, csol_coarse, p)
        force = if Ec === nothing
            r3 = (p[1]^2 + p[2]^2 + p[3]^2)^T(1.5)
            (-w2 * escaped / r3) .* p
        else
            w .* Ec
        end
        if projectile !== nothing
            d = projectile.position .- p
            m = -w * projectile.charge *
                force_kernel(projectile.softening, sum(abs2, d))
            force = force .- m .* d
        end
        cloud.forces[i] = force
    end
    n
end

function projectile_forces!(cloud::ParticleCloud{T}, acc::DeviceAccelerator{E,T},
                            proj::Projectile{T}, jel::Jellium{T}) where {E,T}
    p = proj.position
    q = proj.charge
    r2 = sum(abs2, p)
    modf = r2 > jel.radius^2 ? jel.nions * q / r2^T(1.5) : q / WIGNER_SEITZ_NA^3
    fjel = modf .* p
    red = acc.reduction.host
    fel = ntuple(d -> T(red[d]), 3)
    (fjel .+ fel, T(cloud.weight) * q * T(red[4]), q * potential(jel, sqrt(r2)))
end
