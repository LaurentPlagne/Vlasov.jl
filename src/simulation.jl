"""
Simulation parameters and time loop.

The Fortran read its parameters from `vlas.inp`, a positional file where each
value is preceded by its comment: swapping two lines goes unnoticed and changes
the physics. Here they are named.
"""

"""
    SimulationParameters(; …)

Parameters of a simulation. The Fortran's names are recalled alongside.

  * `nfine` (`n1xyz`) — intervals of the fine grid, over `[-rcluster, rcluster]`;
  * `ninner`, `nouter` (`n1big`, `n2big`) — subdivision of the coarse grid;
  * `rcluster`, `rbox` (`xclu`, `xboite`) — radii of the two domains;
  * `nions`, `nelectrons` (`nbion`, `nbelec`);
  * `nparticles` (`npart`) — pseudo-particles;
  * `nsteps`, `dt` (`nbt`, `dltt`);
  * `rcmax` — radius beyond which an electron counts as having **left** the
    cluster. Hard-coded to `100.d0` until 1997; became the 19th value of
    `vlas.inp` in the 1998-01-05 version, where the production `vlas.inp`
    documents it as "radius considered as inner cluster". The default therefore
    reproduces the earlier behaviour.
"""
Base.@kwdef struct SimulationParameters{T<:AbstractFloat}
    nfine::Int = 28
    ninner::Int = 14
    nouter::Int = 14
    rcluster::T = 50.0
    rbox::T = 150.0
    nions::T = 196.0
    nelectrons::T = 196.0
    nparticles::Int = 20_000
    nsteps::Int = 10
    dt::T = 1.0
    rcmax::T = 100.0
end

"""
    read_parameters(path) -> SimulationParameters

Reads a `vlas.inp` of the original code: one comment line, one value line,
alternating. The projectile parameters that follow are ignored — they do not
concern the isolated cluster.
"""
function read_parameters(path::AbstractString)
    vals = String[]
    open(path) do io
        for (i, line) in enumerate(eachline(io))
            isodd(i) || push!(vals, strip(line))   # even lines = values
        end
    end
    length(vals) >= 10 ||
        throw(ArgumentError("$path: 10 values expected, $(length(vals)) found"))
    num(s) = parse(Float64, replace(s, "d" => "e", "D" => "e"))
    # `rcmax` is the 19th value, absent from pre-1998 `vlas.inp` files: we then
    # fall back on the `100.d0` the Fortran hard-coded.
    rcmax = length(vals) >= 19 && !isempty(vals[19]) ? num(vals[19]) : 100.0
    SimulationParameters(
        nfine = Int(num(vals[1])), ninner = Int(num(vals[2])), nouter = Int(num(vals[3])),
        rcluster = num(vals[4]), rbox = num(vals[5]),
        nions = num(vals[6]), nelectrons = num(vals[7]),
        nparticles = Int(num(vals[8])), nsteps = Int(num(vals[9])), dt = num(vals[10]),
        rcmax = rcmax)
end

"""
Grid state living on the device, and the mirrors of the two meshes that operate
on it.

Present only when a backend was asked for. `Simulation` carries it as a type
parameter, so the branch in [`update_forces!`](@ref) is resolved at compile time
and the host path keeps exactly the code it had.

`csolc_host` is the **one** readback that remains, and it is deliberate: the
coarse coefficients are read by the handful of particles that fall outside the
fine grid, on the host, through `spline_field`.

On unified memory it costs nothing: `copyto!` between two views of the same RAM.
On a discrete GPU it is one transfer of `n³·sizeof(E)` per step — 4.8 MB at 134³
in `Float32`, about 0.2 ms over PCIe.

⚠️ There used to be a second one, `ρc_host`: the coarse deposition was a host
scatter, `CellSort` assuming a uniform grid where the coarse axis is *stretched*.
It no longer is — [`_deposit_cic_kernel!`](@ref) deposits on the device without
any sort at all — and the buffer went with it.
"""
struct DeviceState{E,B,DM,G,AC}
    backend::B
    meshes::NTuple{2,DM}
    ρ::NTuple{2,G}
    φ::NTuple{2,G}
    csol::NTuple{2,G}
    csolc_host::Array{E,3}
    """⚠️ A resident simulation **owns** its accelerator rather than being handed
       one. Priming calls `update_forces!` from inside the constructor, before
       any caller could have built one — and on this path there is no host
       version of the deposition or the forces to fall back on."""
    accelerator::AC
end

function DeviceState(backend, ::Type{E}, meshes::NestedMeshes{2,3,T},
                     sm::GaussianSmoothing{T}, npart::Integer;
                     packed::Bool = false) where {E,T}
    dms = (DeviceMesh(backend, E, meshes[1]), DeviceMesh(backend, E, meshes[2]))
    dims = map(m -> size(m.scratch[1]), (meshes[1], meshes[2]))
    grid(l) = KernelAbstractions.zeros(backend, E, dims[l]...)
    acc = DeviceAccelerator(backend, E, meshes[1].axes, sm, npart, dims[1][1]; packed)
    DeviceState{E,typeof(backend),eltype(dms),typeof(grid(1)),typeof(acc)}(
        backend, dms,
        (grid(1), grid(2)), (grid(1), grid(2)), (grid(1), grid(2)),
        zeros(E, dims[2]...), acc)
end

"""
    Simulation(params, profile; backend = nothing, precision = T)

Everything that stays constant over a simulation — nested meshes, smoothing
tables, jellium background — plus the state that evolves: the cloud of
pseudo-particles.

Building a `Simulation` does the heavy work once: matrix assembly,
diagonalisations, convolution tables. The steps that follow only ever reuse
multiplications.
"""
struct Simulation{T<:AbstractFloat,P,A,F,D}
    params::SimulationParameters{T}
    meshes::NestedMeshes{2,3,T,BandedMatrix{T,Matrix{T},Base.OneTo{Int}}}
    smoothing::GaussianSmoothing{T}
    jellium::Jellium{T}
    """⚠️ The container is a parameter, not `ParticleCloud{T}`: that spelling is
       a `UnionAll` since the cloud gained its array type, and an abstract field
       here would box the hottest object of the whole loop."""
    cloud::ParticleCloud{T,A,F}
    """Projectile, or `nothing` for an isolated cluster. The type carries it
       rather than a `Union` field: the loop stays specialised in both cases."""
    projectile::P
    ρ::NTuple{2,Array{T,3}}
    φ::NTuple{2,Array{T,3}}
    """Spline coefficients of the potential, one set per level. They live for a
       whole step: giving them their own buffers avoids 2.8 MB of allocations
       per step, and the garbage collection that comes with them."""
    csol::NTuple{2,Array{T,3}}
    """Scatter buffers, one per thread and per level, for the parallel
       deposition. See `ScatterBuffers`: their memory cost grows as the cube of
       the grid, hence their explicit presence here rather than a quiet creation
       at every deposition."""
    scatter::NTuple{2,ScatterBuffers{T,3}}
    """Device grid state, or `nothing` — see [`DeviceState`](@ref). Being a type
       parameter, which of the two paths `update_forces!` takes is settled at
       compile time."""
    device::D
end

"""
`backend` opts the **grids** into device residency: `ρ`, `φ` and `csol` then
live there, and the whole Poisson chain with them. `precision` is the type they
carry — `Float32` on Metal, which has no other choice; `Float64` wherever the
hardware offers it.

The cloud stays on the host either way. Its positions are `T`, and the packing
that reads them needs `T`; see `_pack_kd_kernel!`.

`packed = true` holds the cloud as [`PackedPositions`](@ref) instead — cell
index and offset on the fine axis, the very form the device kernels consume, so
that the per-step packing has nothing left to do. `precision` then also fixes
the type of the offsets.
"""
function Simulation(p::SimulationParameters{T}, profile::PhaseSpaceProfile{T};
                    rng::Ran2 = Ran2(-1), consistent_startup::Bool = false,
                    projectile = nothing, backend = nothing,
                    precision::Type = T, packed::Bool = false) where {T}
    fine = uniform_axis(-p.rcluster, p.rcluster, p.nfine)
    coarse = stretched_axis(p.rcluster, p.rbox, p.ninner ÷ 2, (p.nouter + 2) ÷ 2)
    meshes = NestedMeshes(SplineMesh(fine, fine, fine),
                          SplineMesh(coarse, coarse, coarse))

    weight = p.nelectrons / p.nparticles
    positions, momenta = sample_thomas_fermi(profile, p.nparticles, weight; rng)
    n = nbasis(fine)
    smoothing = GaussianSmoothing(fine)
    device = backend === nothing ? nothing :
             DeviceState(backend, precision, meshes, smoothing, p.nparticles; packed)
    # A packed cloud on a device borrows the accelerator's own `(k, δ)` staging
    # rather than allocating a second copy of it: those buffers are exactly what
    # the kernels read, so the cloud writing into them is what makes the packing
    # step disappear instead of merely moving. Its forces come from there too —
    # see [`StagedForces`](@ref).
    cloud = if packed
        store = device === nothing ? nothing : device.accelerator.particles.host
        force = device === nothing ? nothing : device.accelerator.force.host
        packed_cloud(fine, positions, weight, precision; storage = store,
                     forces = force)
    else
        ParticleCloud(positions, weight)
    end
    # ⚠️ **No host scatter buffers once there is a device.** They are one `n³`
    # array *per thread and per level* — 1.6 GB at 222³ on ten threads, the
    # second largest block in the whole run — and [`update_forces!`](@ref)
    # takes the device path whenever `sim.device` exists, packed or not, so not
    # one of them is ever written. Asking for zero slots is what says so; the
    # host deposition then fails loudly rather than quietly allocating them
    # again.
    slots = device === nothing ? Threads.nthreads() : 0
    sim = Simulation(p, meshes, smoothing, Jellium(p.nions),
                     cloud, projectile,
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (ScatterBuffers(meshes[1]; nslots = slots),
                      ScatterBuffers(meshes[2]; nslots = slots)),
                     device)
    prime_leapfrog!(sim, positions, momenta; consistent = consistent_startup)
end

"""
    prime_leapfrog!(sim, positions, momenta; consistent) -> sim

Primes the Verlet scheme, which needs **two** positions and not a position and a
velocity.

In two stages, like the Fortran's `moveback1` then `moveback2`: half a step
backwards from the momenta, then the full step, which needs the forces — hence a
potential, hence a deposition and a Poisson solve.

⚠️ Skipping the second stage does not leave the cloud "approximately" primed:
`previous` would hold `q(−dt/2)` where the scheme expects `q(−dt)`, that is an
initial velocity wrong by a factor of two. The symptom is a kinetic energy that
rises from the very first steps.
"""
function prime_leapfrog!(sim::Simulation{T}, positions, momenta;
                         consistent::Bool = false) where {T}
    M = mass(sim.cloud)
    dt = sim.params.dt
    half = half_step_back(positions, momenta, M, dt)

    # The forces are evaluated at q(−dt/2), not at q(0). The projectile, for its
    # part, does not move: the Fortran calls `incproj` only once priming is
    # over, and advancing it here would have it enter the cluster one step early.
    # ⚠️ `q(0)` travels in the `previous` half rather than in a local array.
    # The sort **moves the particles**, so anything held outside the cloud goes
    # out of correspondence with it the moment `update_forces!` runs; the two
    # halves of one record can never drift apart, because the sort moves them
    # together. Without this the priming wrote `previous` in the old order over
    # positions already reordered, and the trajectory started wrong.
    copyto!(sim.cloud.positions, half)
    copyto!(sim.cloud.previous, positions)
    update_forces!(sim; advance = false)
    # ⚠️ The priming reads the forces **on the host**. A resident cloud reads
    # them where the kernel left them — its `forces` is a view of the
    # accelerator's buffer ([`StagedForces`](@ref)) — so there is nothing to
    # stage here, and nothing to allocate for the staging.
    coef = consistent ? dt^2 / 4M : dt / M
    @inbounds for i in eachindex(sim.cloud.positions)
        q0 = sim.cloud.previous[i]      # q(0), carried through the sort
        hp = sim.cloud.positions[i]     # q(−dt/2), in the same order
        f = sim.cloud.forces[i]
        back = .-q0 .+ 2 .* hp .+ coef .* f
        sim.cloud.positions[i] = q0
        sim.cloud.previous[i] = back
    end
    sim
end

"""
    update_forces!(sim) -> T

Deposits the particles, solves Poisson on both grids, adds the mean field and
fills the cloud's forces. Returns the Hartree energy, measured **before** the
mean field is added — that is the only moment when the bare potential is
available.

`advance = false` computes the forces without advancing the projectile, which is
what priming needs.

`energy = false` skips the Hartree energy and returns `nothing`. This is no
token economy: at 800 000 pseudo-particles that single call accounts for 23 % of
the step. It has to be decided **here** and not after the fact — the Hartree
energy is measured on the coefficients before exchange-correlation overwrites
them, and they exist only between two lines.
"""
function update_forces!(sim::Simulation{T}; advance::Bool = true,
                        energy::Bool = true, accelerator = nothing) where {T}
    sim.device === nothing ||
        return _update_forces_resident!(sim, sim.device, accelerator, advance, energy)

    fine, coarse = sim.meshes[1], sim.meshes[2]
    ρf, ρc = sim.ρ
    w = sim.cloud.weight

    if accelerator === nothing
        deposit_smoothed!(ρf, fine, sim.smoothing, sim.cloud.positions;
                          charge = w, buffers = sim.scatter[1])
    else
        deposit_smoothed!(ρf, accelerator, fine, sim.smoothing, sim.cloud.positions;
                          charge = w)
    end
    deposit!(ρc, coarse, sim.cloud.positions; charge = w, buffers = sim.scatter[2])
    poisson!(sim.φ, sim.ρ, sim.meshes)

    csolf = spline_coefficients!(sim.csol[1], sim.φ[1], fine)
    csolc = spline_coefficients!(sim.csol[2], sim.φ[2], coarse)
    hartree = energy ? interaction_energy(sim.cloud, fine.axes, csolf,
                                          coarse.axes, csolc, sim.smoothing) / 2 :
                       nothing

    effective_potential!(csolf, ρf, fine, sim.jellium)
    effective_potential!(csolc, ρc, coarse, sim.jellium)
    if accelerator === nothing
        forces!(sim.cloud, fine.axes, csolf, coarse.axes, csolc, sim.smoothing)
    else
        # The projectile is fused into the force kernel: passing it here avoids
        # a second sweep over every particle.
        # `packed = true`: the accelerated deposition, which opens the step, has
        # already packed the positions for the GPU.
        forces!(sim.cloud, accelerator, fine.axes, csolf, coarse.axes, csolc,
                sim.smoothing; projectile = sim.projectile, packed = true)
    end
    advance && advance_projectile!(sim, accelerator)

    # The total potential is used next for the budget: we keep it at hand.
    sim.φ[1] .= csolf
    sim.φ[2] .= csolc
    hartree
end

"""
    step!(sim; energy = true) -> EnergyBudget or `nothing`

One complete time step, in the order of the original code:

 1. deposition of the particles onto both grids — smoothed on the fine one;
 2. nested Poisson, coarse to fine, giving the Hartree potential;
 3. Hartree energy, measured **on that potential**, before anything is added to
    it;
 4. addition of exchange-correlation and the jellium;
 5. forces, then the Verlet advance;
 6. energy budget, measured on the **total** potential.

The ordering of points 3 and 6 is no matter of convenience: the two terms of the
budget refer to different potentials, and swapping them would make the total
silently wrong.

`accelerator` diverts the **smoothed deposition** and the **field evaluation**
to a [`ForceAccelerator`](@ref) — the GPU. ⚠️ That path works in `Float32`: the
trajectory is no longer the CPU path's, only the same to within `4e-5`.

`energy = false` skips points 3 and 6 and returns `nothing`. **It is the step's
largest item** — the two calls together account for 34 % of the CPU time, and
43 % once the forces are on the GPU, more than the forces themselves. The
Fortran computed `enertot2g` only one step in ten; doing the same yields ×1.63.
The trajectory does not depend on it: the budget feeds back into nothing, it
observes.
"""
function step!(sim::Simulation{T}; energy::Bool = true,
               accelerator = nothing) where {T}
    hartree = update_forces!(sim; energy, accelerator)
    diag = _advance_cloud!(sim)
    energy || return nothing
    total = interaction_energy(sim.cloud, sim.meshes[1].axes, sim.φ[1],
                               sim.meshes[2].axes, sim.φ[2], sim.smoothing)
    energy_budget(sim.jellium, diag.kinetic, hartree, total, diag.escaped)
end

"""
Advances the cloud — on the device when it lives there.

A cloud held on the accelerator's own buffers is integrated by the kernel, which
also spares the step the conversion of every force triple to host `Float64`.
Every other cloud takes the host integrator.
"""
_advance_cloud!(sim::Simulation) =
    step!(sim.cloud, sim.params.dt; rcmax = sim.params.rcmax)

function _advance_cloud!(sim::Simulation{T,P,<:PackedPositions}) where {T<:AbstractFloat,P}
    dev = sim.device
    if dev !== nothing && sim.cloud.positions.data === dev.accelerator.particles.host &&
       length(dev.accelerator.vpartials) > 0
        return step!(sim.cloud, sim.params.dt, dev.accelerator;
                     rcmax = sim.params.rcmax)
    end
    step!(sim.cloud, sim.params.dt; rcmax = sim.params.rcmax)
end

"""
    run!(sim; nsteps, energy_every = 1, accelerator = nothing, callback) -> Vector{EnergyBudget}

Chains `nsteps` steps and returns the history of the energy budget — the
stability observable of chapter 4. `callback(i, budget)` is called after each
step, with `nothing` for budget at the steps where it is not computed.

`energy_every = 10` reproduces the Fortran, which called `enertot2g` only one
step in ten, and **yields ×1.63**: the budget is the step's largest item. The
default stays at 1 so that nothing changes without having been asked for.
"""
function run!(sim::Simulation; nsteps::Integer = sim.params.nsteps,
              energy_every::Integer = 1, accelerator = nothing,
              callback = (i, b) -> nothing)
    energy_every >= 1 || throw(ArgumentError("`energy_every` must be at least 1"))
    history = EnergyBudget{Float64}[]
    for i in 1:nsteps
        b = step!(sim; energy = (i - 1) % energy_every == 0, accelerator)
        b === nothing || push!(history, b)
        callback(i, b)
    end
    history
end

"""
    advance_projectile!(sim)

Adds the projectile's interaction — force on it, reaction on the
pseudo-particles — then advances it by one step.

With no projectile, does nothing: the body is eliminated at compile time, so the
isolated-cluster loop does not pay for it.
"""
advance_projectile!(::Simulation{T,Nothing}, accelerator = nothing) where {T} = nothing

function advance_projectile!(sim::Simulation{T,<:Projectile{T}},
                             accelerator = nothing) where {T}
    force, _, _ = accelerator === nothing ?
        projectile_forces!(sim.cloud, sim.projectile, sim.jellium) :
        projectile_forces!(sim.cloud, accelerator, sim.projectile, sim.jellium)
    step!(sim.projectile, force, sim.params.dt)
    nothing
end

"""
    _update_forces_resident!(sim, dev, accelerator, advance, energy) -> T or nothing

The same step as [`update_forces!`](@ref), with the grids on the device.

Line for line it is the host version; only the arrays differ, which is the whole
point of having split every function of the chain in two. The two readbacks it
performs are named in [`DeviceState`](@ref) and are there for stated reasons,
not for want of porting something.
"""
function _update_forces_resident!(sim::Simulation{T}, dev::DeviceState{E},
                                  _ignored, advance, energy) where {T,E}
    accelerator = dev.accelerator
    fine, coarse = sim.meshes[1], sim.meshes[2]
    dmf, dmc = dev.meshes
    w = sim.cloud.weight

    deposit_smoothed!(dev.ρ[1], accelerator, fine, sim.smoothing,
                      sim.cloud.positions; charge = w)

    # The coarse deposition, on the device — cloud-in-cell, eight atomics per
    # particle. 203 ms against 396 for the threaded host scatter it replaces,
    # at 8×10⁷ particles on a 258³ coarse grid.
    #
    # ⚠️ A *sorted* version was written too, on the model of the fine
    # deposition: one group per occupied coarse cell, eight work-items for its
    # corners, one atomic each. It is not needed. The coarse grid is coarse
    # enough that eight atomics per particle do not contend the way an 8³
    # stencil does, and the coarse sort such a version requires costs 356 ms on
    # its own — more than the whole deposition.
    fine_ax = fine.axes[1]
    hf = (fine_ax.knots[end] - fine_ax.knots[1]) / (length(fine_ax.knots) - 1)
    deposit_cic!(dev.ρ[2], dmc, accelerator, length(sim.cloud.positions), w,
                 fine_ax.knots[1], hf)

    poisson!(dev.φ, dev.ρ, dev.meshes)
    csolf = spline_coefficients!(dev.csol[1], dev.φ[1], dmf)
    csolc = spline_coefficients!(dev.csol[2], dev.φ[2], dmc)

    # The budget is a diagnostic, taken one step in ten or not at all, and it
    # reads the potential particle by particle on the host. Bringing the two
    # coefficient arrays back for it is cheaper than a kernel that would run
    # that rarely — and on unified memory it is not a copy at all.
    hartree = nothing
    if energy
        # ⚠️ `Array(...)` first: a broadcast straight from a device array into a
        # host one of a different element type is dispatched to the *device*,
        # which then refuses the host destination as a non-bitstype argument.
        sim.csol[1] .= Array(csolf)
        sim.csol[2] .= Array(csolc)
        hartree = interaction_energy(sim.cloud, fine.axes, sim.csol[1],
                                     coarse.axes, sim.csol[2], sim.smoothing) / 2
    end

    effective_potential!(csolf, dev.ρ[1], dmf, sim.jellium)
    effective_potential!(csolc, dev.ρ[2], dmc, sim.jellium)

    # The coarse coefficients come back for the particles that left the fine
    # grid: `forces!` resolves those on the host, over the compacted list.
    copyto!(dev.csolc_host, csolc)
    forces!(sim.cloud, accelerator, fine.axes, csolf, coarse.axes,
            dev.csolc_host, sim.smoothing; projectile = sim.projectile,
            packed = true)
    advance && advance_projectile!(sim, accelerator)

    # `copyto!` and not `.=` : this is a copy, and the blit path does it without
    # going through a broadcast at all — see `_solve!` for the measurements.
    copyto!(dev.φ[1], csolf)
    copyto!(dev.φ[2], csolc)
    hartree
end

"""
    sync_host!(sim) -> sim

Copies the device grids back into `sim.ρ`, `sim.φ` and `sim.csol`.

A resident simulation keeps its grids on the device, where a script reading
`sim.φ[1]` would not find them. Call this before looking. A no-op when there is
no device state.
"""
sync_host!(sim::Simulation{T,P,A,F,Nothing}) where {T,P,A,F} = sim

function sync_host!(sim::Simulation)
    dev = sim.device
    for l in 1:2
        sim.ρ[l] .= Array(dev.ρ[l])
        sim.φ[l] .= Array(dev.φ[l])
        sim.csol[l] .= Array(dev.csol[l])
    end
    sim
end
