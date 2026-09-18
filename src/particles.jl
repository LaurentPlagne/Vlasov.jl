"""
Pseudo-particles and time integration.

The method samples the phase-space density with pseudo-particles, each
representing `weight` electrons: its mass and charge are those of those `weight`
electrons taken together.

The integration is a position Verlet: the velocity is never stored, it is read
from the gap between two successive positions.
"""

"Electron mass, in atomic units (the Fortran's `mel`)."
const ELECTRON_MASS = 1.0

"Electron charge, in atomic units (the Fortran's `qel`)."
const ELECTRON_CHARGE = -1.0

"""
    ParticleCloud(positions, previous, forces, weight)

Cloud of pseudo-particles for a position Verlet scheme.

`previous` holds the positions at the previous step — that is what stands in for
the velocity. `weight` is the number of electrons each pseudo-particle
represents.

The Fortran stored all of this in `(3, npartmax)` arrays in column order, which
is exactly the memory layout of a `Vector{NTuple{3,T}}`: the port is a
reinterpretation, not a conversion.

That coincidence is what spares us a rewrite for the accelerators. The **element
type stays** `NTuple{3,T}` — it is already the `(3, N)` layout a device wants —
and only the **container** is a parameter, so that `A` may be a `Vector` or any
device vector. `ParticleCloud{T}` remains valid in a signature: the container is
left free there.
"""
struct ParticleCloud{T<:AbstractFloat,A<:AbstractVector{NTuple{3,T}}}
    positions::A
    previous::A
    forces::A
    weight::T
end

function ParticleCloud(positions::AbstractVector{NTuple{3,T}}, weight::T) where {T}
    # `similar` and not `fill`: a device cloud must yield device buffers.
    p = copy(positions)
    zero3 = ntuple(_ -> zero(T), 3)
    ParticleCloud(p, fill!(similar(p), zero3), fill!(similar(p), zero3), weight)
end

Base.length(c::ParticleCloud) = length(c.positions)

"""Moves the three buffers to another backend — `adapt(MtlArray, cloud)` and the
like. `weight` is a scalar and travels as it is."""
Adapt.adapt_structure(to, c::ParticleCloud) =
    ParticleCloud(adapt(to, c.positions), adapt(to, c.previous),
                  adapt(to, c.forces), c.weight)

"""Mass of a pseudo-particle: that of the `weight` electrons it carries."""
mass(c::ParticleCloud) = ELECTRON_MASS * c.weight

"""Charge of a pseudo-particle."""
charge(c::ParticleCloud) = ELECTRON_CHARGE * c.weight

"""Cross product of two triples."""
@inline cross3(a, b) = (a[2] * b[3] - a[3] * b[2],
                        a[3] * b[1] - a[1] * b[3],
                        a[1] * b[2] - a[2] * b[1])

"""
    step!(cloud, dt; rcmax = Inf) -> (; kinetic, escaped, angular)

Advances the cloud by one time step with position Verlet:

    q(t+dt) = 2q(t) − q(t−dt) + dt²·F/M

and returns the diagnostics the Fortran computed in the same loop — kinetic
energy, total angular momentum, and the share of the kinetic energy carried by
the particles that have **left**. They are obtained from the centred momentum
`p = M·(q(t+dt) − q(t−dt))/2dt`, which exists only here: computing it afterwards
would require keeping one more piece of state.

`rcmax` is the radius beyond which a particle counts as gone, measured on the
position **before** the step — like the Fortran's `move`, which tests `ract` on
`qp` and not on the new position. It was hard-coded to `100.d0` until 1997, and
becomes an input parameter in 1998. The default `Inf` counts nothing as gone,
leaving `escaped` at zero for callers who do not use it.
"""
function step!(cloud::ParticleCloud{T}, dt::T; rcmax::Real = T(Inf)) where {T}
    M = mass(cloud)
    acc = dt^2 / M
    pfac = M / 2dt
    r2max = T(rcmax)^2
    ekin = zero(T)
    eout = zero(T)
    angular = ntuple(_ -> zero(T), 3)

    @inbounds for i in eachindex(cloud.positions)
        q, qold, f = cloud.positions[i], cloud.previous[i], cloud.forces[i]
        qnew = 2 .* q .- qold .+ acc .* f
        p = pfac .* (qnew .- qold)
        e = (p[1]^2 + p[2]^2 + p[3]^2) / 2M
        ekin += e
        # Comparing squares: one square root per particle for a mere threshold
        # is one square root too many.
        q[1]^2 + q[2]^2 + q[3]^2 > r2max && (eout += e)
        angular = angular .+ cross3(q, p)
        cloud.previous[i] = q
        cloud.positions[i] = qnew
    end
    (; kinetic = ekin, escaped = eout, angular)
end

"""
    half_step_back(positions, momenta, M, dt)

Leapfrog priming, first stage: `q(−dt/2) = q(0) − (dt/2M)·p`.

This is the Fortran's `moveback1`, where the array of previous positions still
held the momenta produced by the initial sampling.
"""
half_step_back(positions, momenta, M, dt) =
    map((q, p) -> q .- (dt / 2M) .* p, positions, momenta)

"""
    full_step_back(positions, half, forces, M, dt)

Leapfrog priming, second stage: from `q(0)` and `q(−dt/2)` to `q(−dt)`, using
the forces evaluated at `q(−dt/2)`.

⚠️ **Reproduces a dubious coefficient of the original code.** `moveback2`
computes

    coef2 = 0.5·dltt*2·npart/(mel·nbelec)

that is, `dt/M`. But `coef2·F` is then a *velocity*, added to lengths: the
formula is not dimensionally consistent. A Taylor expansion gives `dt²/4M`, and
the neighbouring routine `move` does write `dltt**2`. Everything points to a
typo, `dltt*2` for `dltt**2` — but it is present identically in **all five**
versions of the thesis code, hence never fixed.

The effect is one-off: it only distorts the priming, like an error on the
initial velocity. We reproduce it as is, to stay comparable with the oracle;
`consistent = true` selects the dimensionally consistent variant `dt²/4M`, to
measure what the typo costs.
"""
function full_step_back(positions, half, forces, M, dt; consistent::Bool = false)
    coef = consistent ? dt^2 / 4M : dt / M
    map((q, h, f) -> .-q .+ 2 .* h .+ coef .* f, positions, half, forces)
end

# No standalone `kinetic_energy(cloud, dt)`: the centred momentum at t is read
# between q(t+dt) and q(t−dt), which coexist only inside a step. An after-the-
# fact function would see only q(t) and q(t−dt) and would return the momentum at
# t−dt/2 — a different quantity under the same name. `step!` returns it.
