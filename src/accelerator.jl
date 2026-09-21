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
struct DeviceAccelerator{E,T,B,G,BC,BF,BP,BR,BI,BU,BX,C} <: ForceAccelerator
    backend::B
    "Tables of the fine mesh, device-side — see [`DeviceGrid`](@ref)."
    grid::G
    csol::BC
    rho::BC
    force::BF
    """The cloud itself, one [`PackedParticle`](@ref) per particle — current
       and previous `(k, δ)` in a single record.

       ⚠️ One array, not four. The sort **moves the particles**, and a record
       that spans one cache line is what makes that affordable: 145.5 ms
       against 429.7 to permute the same data as separate `knode`/`delta`
       arrays. Everything downstream then reads in order, with no permutation
       to indirect through."""
    particles::BP
    "Where the sort places them, before they are copied back. Empty unless the
     cloud lives here — a scatter cannot be done in place."
    sorted::BP
    """Per-work-item partials of the integrator's three diagnostics. Empty
       unless the cloud lives here — at 8×10⁷ particles the two `previous`
       buffers alone are 1.9 GB, which a host-held cloud has no use for."""
    vpartials::BF
    """Counting sort on the device: the per-cell counts and the running
       cursor of the placement. Empty unless the cloud lives
       here."""
    counts::BU
    cursor::BU
    "Exclusive offsets, host side: the scan is `O(cells)` and stays here."
    offsets::Vector{Int32}
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
    """Whether the device has ever been given the cloud. A resident cloud is not
       uploaded per step — the device copy is the authoritative one — but it has
       to cross **once**, and whoever built it on the host may not be
       [`prime_leapfrog!`](@ref): the suite builds one by hand, and so does any
       script that drives an accelerator directly."""
    primed::Base.RefValue{Bool}
    x0::T
    h::T
    spacing::E
    nbdt::Int32
    ncol::Int32
    npart::Int
end

"""
Work-items the packed Verlet splits the cloud over.

Enough of them to fill the machine and keep each one's register accumulation
short — a few dozen terms, so the diagnostics stay accurate in `E` — and few
enough that the host's final sum over the partials is free.
"""
verlet_workitems(npart::Integer) = min(Int(npart), 1 << 20)

"""
    scatter_stride(npart) -> Int32

A stride that walks all of `1:npart` while breaking the sorted order.

For [`_deposit_cic_kernel!`](@ref), whose atomics collide when consecutive
particles share a coarse cell — which is precisely what sorting by fine cell
arranges. Any value coprime with `npart` visits every particle exactly once.

⚠️ **It is a balance, not a maximum.** The first version took the biggest prime
to hand, on the reasoning that the further apart the neighbours, the fewer the
collisions. That is true and it is not the only cost: a work-item then reads
its 48-byte particle from its own page, and the kernel becomes bound on address
translation. Apple's counters on it, at 8×10⁷ particles:

| | stride 7919 | **stride 509** |
|---|---:|---:|
| **MMU Limiter** | **70.7 %** | **61.5 %** |
| GPU Last Level Cache Limiter | 55.9 % | 52.0 % |
| Buffer Read Limiter | 40.7 % | 15.0 % |
| MMU TLB Miss Rate | 15.1 % | 11.9 % |
| ALU Limiter | 7.4 % | 14.4 % |
| GPU Read Bandwidth | 42.3 GB/s | 35.8 GB/s |

Same window of wall clock, and the right-hand column does **2.2× the work** in
it: the kernel measures 195.0 ms against 89.7 (A-B-A at 8×10⁷, `ndrange` over
the whole cloud). Per particle it reads 2.6× fewer bytes and walks the page
tables less, because neighbouring work-items now share a cache line and a page.

The curve is flat-bottomed, and the two ends of it are steep:

| stride | 1 | 31 | 127 | 251 | **383** | **509** | 1009 | 2003 | 7919 | 65537 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms (2×10⁷ work-items) | 1148 | 132 | 42.7 | 24.2 | **22.9** | 23.4 | 26.5 | 34.5 | 48.9 | 57.9 |

509 and not 383: on a tenth of the cloud — 8×10⁶ particles, the film's scale —
the basin shifts a little and 509 is its floor (11.5 ms against 13.5), while at
8×10⁷ it costs 2.6 % over 383. The basin sits where it does because of the
*cloud*, not the array: a run of particles sharing a coarse cell is some
thousand long here, and the stride has to be a fraction of that — far enough to
spread a SIMD group over several cells, near enough to stay in a handful of
pages.

⚠️ Confining the scatter to a window of consecutive particles — the obvious way
to bound the translation cost — is **much worse**: 154.8 ms for a window of
4×10⁶ particles and 1009 ms for one of 65 536, against 89.7 for the plain
stride. Inside a small window the same coarse cells come round again and again,
and the collisions the stride exists to break come straight back.
"""
function scatter_stride(npart::Integer)
    s = 509
    while gcd(s, npart) != 1
        s += 2
    end
    Int32(s)
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
                           n::Integer; packed::Bool = false) where {E,T}
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
    ncell = length(fine[1].knots)^3
    DeviceAccelerator(
        backend,
        DeviceGrid(backend, E, fine, n),
        dual_buffer(backend, E, n, n, n),      # csol
        dual_buffer(backend, E, n, n, n),      # rho
        dual_buffer(backend, E, 3, npart),     # force
        dual_buffer(backend, PackedParticle{E}, npart),                  # particles
        dual_buffer(backend, PackedParticle{E}, npart),                  # sorted
        dual_buffer(backend, E, 5, packed ? verlet_workitems(npart) : 0),
        dual_buffer(backend, Int32, packed ? ncell : 0),          # counts
        dual_buffer(backend, Int32, packed ? ncell : 0),          # cursor
        packed ? Vector{Int32}(undef, ncell) : Int32[],
        dual_buffer(backend, Int32, 3, npart), # cols
        dual_buffer(backend, E, 4),            # reduction
        # Sized for the worst case — one occupied cell per cell of the grid —
        # rather than grown on demand: the sort already pays for `nk³` counters,
        # so this adds nothing anyone was not already spending, and it spares a
        # `Ref` in the hot path.
        dual_buffer(backend, Int32, length(fine[1].knots)^3),
        dual_buffer(backend, Int32, length(fine[1].knots)^3 + 1),
        # ⚠️ `perm` stays allocated even for a packed accelerator, where the
        # device sort *moves* the particles and produces no permutation. It is
        # 4 bytes a particle that the resident path never reads — and the one
        # buffer of the host route that cannot be grown on demand, being a
        # device array rather than a `Vector`. The rest of that route's working
        # set is allocated on first use; see [`CellSort`](@ref).
        dual_buffer(backend, Int32, npart),    # perm
        dual_buffer(backend, Int32, 1),        # nout
        dual_buffer(backend, Int32, npart),    # outlist
        dual_buffer(backend, Int32, 1),        # outcount
        dev(sm.overlap), dev(sm.gradient), dev(sm.nodes),
        CellSort(fine[1], npart; buffers = !packed),
        Ref(false),
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
    _pack_kd_kernel!(cpu)(acc.particles.host, positions, acc.x0, acc.h,
                          Int32(acc.sorter.nknots); ndrange = length(positions))
    synchronize(cpu)
    upload!(acc.particles)
    _sort_and_list!(acc, positions)
end

"""
The same preparation for a cloud already held as `(k, δ)` — which is to say,
without the packing.

Nothing is computed here and, when the cloud was built on this accelerator's own
storage, nothing is copied either: `positions.data` **is**
`acc.particles.host`. What the host used to spend reading 24 bytes of `Float64`
per particle to produce them, it now spends on nothing at all.
"""
function _pack!(acc::DeviceAccelerator{E,T}, positions::PackedPositions) where {E,T}
    resident = positions.data === acc.particles.host
    # ⚠️ **A resident cloud is not uploaded.** The device copy is the one the
    # sort reorders and the integrator advances; the host half is a *view* of it
    # on unified memory and a stale mirror everywhere else. Uploading here — as
    # this did, every step — copies that mirror back over the device and undoes
    # both, which on Apple is invisible (the two are the same bytes) and on a
    # discrete GPU freezes the cloud at the state the priming left. Reported
    # from a CUDA machine: capture flat at 0.000 for the whole crossing, and a
    # step time that never varied because nothing ever moved.
    #
    # The host writes the cloud exactly once, in [`prime_leapfrog!`](@ref),
    # which uploads it itself.
    if resident
        # ⚠️ Once, and only once. The host is what *built* the cloud, so it has
        # to cross before the first step — and after that the device copy is the
        # one the sort reorders and the integrator advances, so uploading again
        # would undo both. Without this flag a cloud built outside
        # [`prime_leapfrog!`](@ref) never reached the device at all: on unified
        # memory that is invisible, and on CUDA every particle read as being at
        # the grid's origin, hence outside it.
        acc.primed[] || (upload!(acc.particles); acc.primed[] = true)
    else
        copyto!(acc.particles.host, positions.data)
        upload!(acc.particles)
    end
    _sort_and_list!(acc, positions)
end

"""
Scan of the per-cell counts: exclusive offsets, the occupied-cell list and its
bounds, and the same offsets lumped per bucket for the sort's first stage.

`O(cells)` and sequential, which is why it stays on the host: at 111³ it is a
couple of milliseconds against the hundred the placement costs, and a parallel
scan would buy nothing measurable.
"""
function _scan_counts!(acc::DeviceAccelerator, counts)
    cs = acc.sorter
    offs = acc.offsets
    empty!(cs.occupied); empty!(cs.bounds); push!(cs.bounds, Int32(0))
    a = Int32(0)
    @inbounds for c in eachindex(counts)
        offs[c] = a
        n = counts[c]
        n == 0 && continue
        push!(cs.occupied, Int32(c))
        a += n
        push!(cs.bounds, a)
    end
    a
end

"""
The counting sort, on the device.

Two kernels and one host scan, replacing the threaded host sort: the cell key
is read from the particle where it already sits, and the placement **moves the
particles**, in the array's own order — see
[`_place_particles_kernel!`](@ref).
"""
function _device_sort!(acc::DeviceAccelerator{E,T}) where {E,T}
    nk = Int32(acc.sorter.nknots)
    np = Int32(acc.npart)
    counts = acc.counts

    fill!(counts.device, Int32(0))
    _hist_cells_kernel!(acc.backend)(counts.device, acc.particles.device, nk, np;
                                     ndrange = acc.npart)
    download!(counts, acc.backend)
    _scan_counts!(acc, counts.host)

    # One pass, walking the array in its own order — which is the fast order
    # because the cloud is already almost sorted from the previous step. See
    # `_place_particles_kernel!`: 35.5 ms this way against 151 through a
    # locality-building first stage that this cloud does not need.
    copyto!(acc.cursor.host, acc.offsets); upload!(acc.cursor)
    _place_particles_kernel!(acc.backend)(acc.sorted.device, acc.cursor.device,
                                          acc.particles.device, nk, np;
                                          ndrange = acc.npart)
    # ⚠️ Back into `particles`, rather than swapping the two buffers: the cloud
    # holds a reference to `acc.particles.host` and must keep seeing its own
    # storage. The copy is sequential — the expensive part was the scatter.
    copyto!(acc.particles.device, acc.sorted.device)
    synchronize(acc.backend)
    acc
end

"""
Reorders `acc.particles` into the order `acc.perm` gives — the host route's way
of arriving where [`_device_sort!`](@ref) arrives directly.

A gather, then a copy back: the scatter has no in-place form, and the cloud's
own storage must keep its identity.
"""
function _apply_perm!(acc::DeviceAccelerator)
    _gather_particles_kernel!(acc.backend)(acc.sorted.device, acc.particles.device,
                                           acc.perm.device, Int32(acc.npart);
                                           ndrange = acc.npart)
    copyto!(acc.particles.device, acc.sorted.device)
    synchronize(acc.backend)
    acc
end

"""Everything of [`_pack!`](@ref) that is not the packing: the sort, the cell
list, and the out-of-stencil list."""
function _sort_and_list!(acc::DeviceAccelerator{E,T}, positions) where {E,T}
    # The device sort needs the cloud on the device, which is what a packed
    # cloud gives it; anything else is still ordered on the host.
    #
    # ⚠️ Either way the particles end up **physically sorted** in
    # `acc.particles`, because that is what every kernel downstream now
    # assumes: they read slot `s` and take it to be particle `s`. On the host
    # route `perm` survives for one purpose only — putting the forces back in
    # the caller's own order, in `forces!`.
    if length(acc.counts) > 0 && positions isa PackedPositions
        _device_sort!(acc)
    else
        cellsort!(acc.sorter, positions)
        copyto!(acc.perm.host, 1, acc.sorter.perm, 1, acc.npart)
        upload!(acc.perm)
        _apply_perm!(acc)
    end

    # The occupied-cell list travels with the sort that produced it. It used to
    # be uploaded inside `deposit_smoothed!`, which was harmless while the
    # deposition was its only reader; the forces kernel reads it too now, and
    # would then have depended on the order the two were called in.
    ncell = length(acc.sorter.occupied)
    copyto!(acc.cells.host, 1, acc.sorter.occupied, 1, ncell)
    copyto!(acc.bounds.host, 1, acc.sorter.bounds, 1, ncell + 1)
    upload!(acc.cells); upload!(acc.bounds)

    # The out-of-stencil list, built here because it depends only on where the
    # particles are. Both the forces and the energy budget read it, and the
    # budget runs first.
    fill!(acc.outcount.device, Int32(0))
    _outside_kernel!(acc.backend)(acc.outlist.device, acc.outcount.device,
                                  acc.particles.device, Int32(size(acc.csol.device, 1)),
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
        acc.cols.device, acc.particles.device,
        E(acc.x0), E(acc.h), E(knots[2] + half), E(knots[end-1] - half),
        acc.spacing, acc.nbdt, acc.ncol, acc.nout.device; ndrange = acc.npart)
    synchronize(acc.backend)

    download!(acc.nout, acc.backend)
    Int(acc.nout.host[1])
end

function deposit_smoothed!(ρ::AbstractArray, acc::DeviceAccelerator{E,T},
                           mesh::SplineMesh{3,T}, sm::GaussianSmoothing{T},
                           positions; charge::T) where {E,T}
    # packs, sorts, uploads the cell list, and lists the boundary cases
    _pack!(acc, positions)
    nout = _fill_columns!(acc, mesh, sm, positions)
    ncell = length(acc.sorter.occupied)

    # A `ρ` that already lives on this backend is deposited into **directly**:
    # on the resident path the density never leaves the device, and the staging
    # buffer is not touched at all.
    resident = get_backend(ρ) === acc.backend && eltype(ρ) === E
    target = resident ? ρ : acc.rho.device
    fill!(target, zero(E))

    _deposit_sorted_kernel!(acc.backend, DEPOSIT_GROUPSIZE)(
        target, acc.nodes, acc.cols.device, acc.cells.device,
        acc.bounds.device, Int32(acc.sorter.nknots); ndrange = ncell * DEPOSIT_GROUPSIZE)
    synchronize(acc.backend)

    # Reduction and scaling stay on the device. What came back to the host
    # before was the reduction plus two full passes over the n³ grid — 4.0 ms
    # of the 33 the deposition takes, at 2×10⁶ particles on Metal.
    q = total_charge(target, acc.grid)
    # `vec`: see `_solve!` — a 3-D broadcast is slower than the flat one on the
    # same buffer, at every grid size tried.
    vec(target) .*= E((length(positions) - nout) * charge / q)
    if resident
        synchronize(acc.backend)
    else
        download!(acc.rho, acc.backend)
        ρ .= acc.rho.host
    end
    nout
end

"""
    step!(cloud, dt, acc; rcmax = Inf) -> (; kinetic, escaped, angular)

The position Verlet on the device — see [`_verlet_packed_kernel!`](@ref).

The cloud must be held as `(k, δ)` on this accelerator's own buffers, which is
what `packed_cloud(...; buffers)` arranges. Nothing is uploaded and nothing is
downloaded but the partials: the positions are already where the kernel runs,
and so are the forces, in `E`.

The three diagnostics are summed on the host, in `T`, over the partials — a
final `nw`-term sum against eighty million.
"""
function step!(cloud::ParticleCloud{T,<:PackedPositions}, dt::T,
               acc::DeviceAccelerator{E,T}; rcmax::Real = T(Inf)) where {E,T}
    q, o = cloud.positions, cloud.previous
    q.data === acc.particles.host ||
        throw(ArgumentError("the cloud is not held on this accelerator"))
    length(acc.vpartials) == 0 &&
        throw(ArgumentError("accelerator built without `packed = true`"))

    M = mass(cloud)
    nw = verlet_workitems(acc.npart)
    _verlet_packed_kernel!(acc.backend)(
        acc.particles.device,
        acc.force.device, acc.vpartials.device,
        E(dt^2 / M), E(acc.h), E(acc.x0), E(M / 2dt), E(T(rcmax)^2), E(1 / 2M),
        Int32(acc.npart), Int32(nw); ndrange = nw)
    download!(acc.vpartials, acc.backend)

    p = acc.vpartials.host
    ekin = zero(T); eout = zero(T)
    lx = zero(T); ly = zero(T); lz = zero(T)
    @inbounds for t in 1:nw
        ekin += T(p[1, t]); eout += T(p[2, t])
        lx += T(p[3, t]); ly += T(p[4, t]); lz += T(p[5, t])
    end
    (; kinetic = ekin, escaped = eout, angular = (lx, ly, lz))
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

    # One group per occupied cell, not per block of particles: the kernel stages
    # each cell's stencil in threadgroup memory. `_pack!` has sorted and
    # uploaded the cell list, so this holds however `forces!` was reached.
    ncell = length(acc.sorter.occupied)
    _smoothed_field_kernel!(acc.backend, FIELD_GROUPSIZE)(
        acc.force.device, csol, acc.overlap, acc.gradient,
        acc.particles.device, acc.cells.device, acc.bounds.device,
        Int32(acc.sorter.nknots), E(acc.x0), E(acc.h), acc.spacing,
        acc.nbdt, E(cloud.weight), acc.ncol,
        pp[1], pp[2], pp[3], coef, σ, acc.reduction.device;
        ndrange = ncell * FIELD_GROUPSIZE)
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
    # A cloud held on this accelerator reads the forces where they already are,
    # in `E`: the conversion below would be a pass over 8×10⁷ triples for
    # nothing. ⚠️ The out-of-stencil corrections must then land in `acc.force`
    # too, and not in `cloud.forces` which no kernel reads.
    # ⚠️ `perm` undoes the sort here, and only here. The kernel wrote slot `s`
    # for the `s`-th *sorted* particle; the caller's cloud is in its own order,
    # so the force of slot `s` belongs to particle `perm[s]`.
    resident = _cloud_is_resident(cloud, acc)
    if !resident
        pm = acc.perm.host
        tforeach(npart) do slice
            @inbounds for s in slice
                i = pm[s]
                cloud.forces[i] = (T(f[1, s]), T(f[2, s]), T(f[3, s]))
            end
        end
    end

    # ⚠️ `outlist` holds **sorted** slots, because `_outside_kernel!` walks the
    # device's own order. A resident cloud is in that order too; a host-held one
    # is not, and its particle is `perm[slot]`.
    n = Int(acc.outcount.host[1])
    # ⚠️ The loop below reads the cloud **on the host**, at slots the device
    # chose — and on a discrete GPU the host half is a stale mirror that the
    # sort has since reordered. So those records are gathered on the device and
    # brought back in **one** transfer, into the sort's scratch buffer, which is
    # free by now. Skipped entirely where the two halves are the same bytes.
    #
    # ⚠️ **One transfer, not `n`.** The first version copied each record on its
    # own — 48 bytes per particle outside the fine grid — on the reasoning that
    # `n` was a few hundred. It is a few hundred at the start and **216 793** by
    # the end of a crossing, as the cloud spreads past the fine grid, and each
    # of those little copies costs a PCIe round trip: measured on a CUDA
    # machine, the step went from 111 ms to 1655 as `n` grew, while the same
    # run on unified memory stayed at 158. Reported, not guessed.
    staged = nothing
    if resident && !acc.particles.shared && n > 0
        _gather_particles_kernel!(acc.backend)(
            acc.sorted.device, acc.particles.device, acc.outlist.device,
            Int32(n); ndrange = n)
        synchronize(acc.backend)
        copyto!(acc.sorted.host, 1, acc.sorted.device, 1, n)
        staged = PackedPositions(acc.sorted.host, T(acc.x0), T(acc.h), false)
    end
    @inbounds for s in 1:n
        slot = Int(acc.outlist.host[s])
        i = resident ? slot : Int(acc.perm.host[slot])
        # `staged[s]` and `cloud.positions[i]` are the same particle: the first
        # is the copy just brought back, the second the host half that is only
        # authoritative when it *is* the device half.
        p = staged === nothing ? cloud.positions[i] : staged[s]
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
        if resident
            f[1, slot] = E(force[1]); f[2, slot] = E(force[2]); f[3, slot] = E(force[3])
        else
            cloud.forces[i] = force
        end
    end
    resident && n > 0 && upload!(acc.force)
    n
end

"""Whether this cloud's positions live in `acc`'s own buffers — in which case
the kernels read the cloud where it is, and nothing is staged."""
# ⚠️ One method with a runtime test, not two with a dispatch on
# `ParticleCloud{T,<:PackedPositions}`. That spelling silently selected the
# generic method here — the cloud was resident and reported as not — and the
# forces were then redistributed through a `perm` the device sort no longer
# fills. The types are concrete at every call site, so the branch folds away.
_cloud_is_resident(cloud::ParticleCloud, acc::DeviceAccelerator) =
    cloud.positions isa PackedPositions &&
        cloud.positions.data === acc.particles.host

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
