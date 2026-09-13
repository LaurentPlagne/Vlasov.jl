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
    Simulation(params, profile)

Everything that stays constant over a simulation — nested meshes, smoothing
tables, jellium background — plus the state that evolves: the cloud of
pseudo-particles.

Building a `Simulation` does the heavy work once: matrix assembly,
diagonalisations, convolution tables. The steps that follow only ever reuse
multiplications.
"""
struct Simulation{T<:AbstractFloat,P}
    params::SimulationParameters{T}
    meshes::NestedMeshes{2,3,T,BandedMatrix{T,Matrix{T},Base.OneTo{Int}}}
    smoothing::GaussianSmoothing{T}
    jellium::Jellium{T}
    cloud::ParticleCloud{T}
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
end

function Simulation(p::SimulationParameters{T}, profile::PhaseSpaceProfile{T};
                    rng::Ran2 = Ran2(-1), consistent_startup::Bool = false,
                    projectile = nothing) where {T}
    fine = uniform_axis(-p.rcluster, p.rcluster, p.nfine)
    coarse = stretched_axis(p.rcluster, p.rbox, p.ninner ÷ 2, (p.nouter + 2) ÷ 2)
    meshes = NestedMeshes(SplineMesh(fine, fine, fine),
                          SplineMesh(coarse, coarse, coarse))

    weight = p.nelectrons / p.nparticles
    positions, momenta = sample_thomas_fermi(profile, p.nparticles, weight; rng)
    n = nbasis(fine)
    sim = Simulation(p, meshes, GaussianSmoothing(fine), Jellium(p.nions),
                     ParticleCloud(positions, weight), projectile,
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (zeros(T, n, n, n), zeros(T, n, n, n)),
                     (ScatterBuffers(meshes[1]), ScatterBuffers(meshes[2])))
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
    copyto!(sim.cloud.positions, half)
    update_forces!(sim; advance = false)

    copyto!(sim.cloud.previous,
            full_step_back(positions, half, sim.cloud.forces, M, dt; consistent))
    copyto!(sim.cloud.positions, positions)
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
    diag = step!(sim.cloud, sim.params.dt; rcmax = sim.params.rcmax)
    energy || return nothing
    total = interaction_energy(sim.cloud, sim.meshes[1].axes, sim.φ[1],
                               sim.meshes[2].axes, sim.φ[2], sim.smoothing)
    energy_budget(sim.jellium, diag.kinetic, hartree, total, diag.escaped)
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
