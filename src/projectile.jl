"""
The projectile ion and its interaction with the cluster.

Subject of chapter 6: a proton crosses the cluster, and the measured quantity is
the **energy it loses there** — the stopping power.

The short-range interaction must be regularised — otherwise a pseudo-particle
passing at contact would feel an infinite force, although it represents a spread
packet of electrons. **How** it is regularised is no detail: the thesis states
that the energy loss depends strongly on it. Hence [`Softening`](@ref) and its
two realisations.
"""

"""
    Softening{T}

Regularisation of the projectile ↔ pseudo-particle interaction at short range.
Without it, a pseudo-particle passing at contact would feel an infinite force,
although it represents a spread packet of electrons.

Two forms, and **they disagree**:

  * [`BallSoftening`](@ref) — what the Fortran does, in all 43 of its versions;
  * [`GaussianSoftening`](@ref) — what the thesis describes, equation
    (`Eforceproj2`) of chapter 2.

The thesis states that "the energy loss of ions crossing the cluster depends
**strongly** on this smoothing", and that `σ_ion = 1` was chosen to reproduce
Lindhard's model. The choice is therefore not an implementation detail: it sets
the stopping power. Hence a type, not an `if`.
"""
abstract type Softening{T<:AbstractFloat} end

"""
    BallSoftening(radius)

Uniformly charged ball of radius `radius`: Coulomb beyond, linear force within.
This is what the Fortran's `forceproji` implements (the `cutoff` of `vlas.inp`),
and hence what the oracle validates against.
"""
struct BallSoftening{T} <: Softening{T}
    radius::T
end

"""
    GaussianSoftening(σ)

Gaussian charge of width `σ` against a point charge — the thesis's `σ_ion`. The
pair potential is

    V(r) = Erf(r / (√2 σ)) / r,     V(0) = √(2/π) / σ

and the force derives from it:

    f⃗ = −Q·Q′ · [Erf(r/(√2σ)) − 2 g(r) r] / r³ · r⃗,
    g(r) = exp(−r²/2σ²) / (√(2π) σ)

`g` is the **one-dimensional** normalised Gaussian: that is what makes `2g(r)·r`
dimensionless, as the bracket requires.

⚠️ The Fortran contains `erfsr`, which is exactly `Erf(r/√2σ)/r`. It is called
in **one version out of 49** — the oldest, July 1996 — where it was already
**tabulated**, along with its force, for a pair-by-pair direct summation that was
abandoned that very month in favour of the grid route. The projectile, added
later, never inherited that table. See `docs/coquilles-fortran.md`, anomaly 10.
"""
struct GaussianSoftening{T} <: Softening{T}
    σ::T
end

Softening(s::Softening) = s

"""
    force_kernel(s, r2) -> T

Factor `m` such that the force equals `Q·Q′·m·r⃗` — the unnormalised vector. `m`
therefore has the dimension of an inverse volume, and `m → 1/r³` far away.
"""
@inline force_kernel(s::BallSoftening{T}, r2::T) where {T} =
    r2 > s.radius^2 ? inv(r2 * sqrt(r2)) : inv(s.radius^3)

# Coefficients of the expansion of [Erf(u/√2) − u·e^{−u²/2}·√(2/π)] / u³ in
# powers of u², up to a factor √(2/π). The two terms cancel at leading order:
# subtracting them as they stand loses every digit when `u` is small, and that
# is precisely the regime of head-on collisions.
const _GAUSS_SERIES = (1/3, -1/10, 1/56, -1/432, 1/4224, -1/49920, 1/685440)
const _SQRT_2_OVER_PI = sqrt(2 / π)

@inline function force_kernel(s::GaussianSoftening{T}, r2::T) where {T}
    σ = s.σ
    u2 = r2 / σ^2
    if u2 <= T(0.25)                      # u ≤ 0.5: series, no subtraction
        p = zero(T)
        @inbounds for k in length(_GAUSS_SERIES):-1:1
            p = T(_GAUSS_SERIES[k]) + u2 * p
        end
        T(_SQRT_2_OVER_PI) * p / σ^3
    else
        r = sqrt(r2)
        x = r / (sqrt(T(2)) * σ)
        (erf(x) - T(_SQRT_2_OVER_PI) * (r / σ) * exp(-u2 / 2)) / (r2 * r)
    end
end

"""
    pair_potential(s, q, r) -> T

Potential created at `r` by a charge `q` regularised by `s`.
"""
@inline pair_potential(s::BallSoftening{T}, q, r) where {T} =
    uniform_sphere_potential(q, s.radius, r)

@inline function pair_potential(s::GaussianSoftening{T}, q, r) where {T}
    x = r / (sqrt(T(2)) * s.σ)
    q * (r < eps(T)^(1//3) * s.σ ? T(_SQRT_2_OVER_PI) / s.σ : erf(x) / r)
end

"Characteristic radius — `radius` for the ball, `σ` for the Gaussian."
scale(s::BallSoftening) = s.radius
scale(s::GaussianSoftening) = s.σ

"""
    Projectile(; mass, charge, energy, impact, x0, dt, cutoff)
    Projectile(; …, softening = BallSoftening(cutoff))

Incident ion, integrated by the same Verlet scheme as the pseudo-particles.

It enters at `x = x0` with the impact parameter `impact` carried by `y`, at the
velocity its kinetic energy `energy` gives it. `initial_energy` is kept: it is
the reference from which the current energy will be subtracted to obtain the
loss.
"""
mutable struct Projectile{T<:AbstractFloat,S<:Softening{T}}
    const mass::T
    "Current charge. It **decreases** if the projectile captures electrons."
    charge::T
    const softening::S
    const initial_energy::T
    position::NTuple{3,T}
    previous::NTuple{3,T}
    velocity::NTuple{3,T}
end

function Projectile(; mass::Real, charge::Real, energy::Real, impact::Real = 0,
                    x0::Real, dt::Real, cutoff::Union{Real,Nothing} = nothing,
                    softening::Union{Softening,Nothing} = nothing)
    (cutoff === nothing) == (softening === nothing) && throw(ArgumentError(
        "supply `cutoff` (ball, the Fortran) **or** `softening` (the thesis), not both"))
    # The type follows from the arguments rather than being a parameter: a
    # default annotated `impact::T = zero(T)` would reference `T` before it is
    # bound.
    T = float(promote_type(typeof(mass), typeof(charge), typeof(energy),
                           typeof(impact), typeof(x0), typeof(dt),
                           cutoff === nothing ? typeof(scale(softening)) : typeof(cutoff)))
    soft = cutoff === nothing ? convert(Softening{T}, softening) : BallSoftening(T(cutoff))
    v = (sqrt(2 * T(energy) / T(mass)), zero(T), zero(T))
    position = (T(x0), T(impact), zero(T))
    Projectile{T,typeof(soft)}(T(mass), T(charge), soft, T(energy),
                               position, position .- T(dt) .* v, v)
end

Base.convert(::Type{Softening{T}}, s::BallSoftening) where {T} = BallSoftening(T(s.radius))
Base.convert(::Type{Softening{T}}, s::GaussianSoftening) where {T} = GaussianSoftening(T(s.σ))

"""Characteristic radius of the softening — the `cutoff` of former times."""
cutoff(p::Projectile) = scale(p.softening)

"""Current kinetic energy of the projectile."""
kinetic_energy(p::Projectile) = p.mass * sum(abs2, p.velocity) / 2

"""
    energy_loss(p) -> T

Energy lost by the projectile since it entered, in atomic units — positive when
it slows down. This is **the observable of chapter 6**.
"""
energy_loss(p::Projectile) = p.initial_energy - kinetic_energy(p)

"Atomic units → electron-volts conversion, as in the original code."
const HARTREE_TO_EV = 27.2116

"""
    projectile_forces!(cloud, proj, jellium) -> (force, e_electrons, e_jellium)

Total force on the projectile, and the **reaction** added to the
pseudo-particles' forces — the interaction is reciprocal, and omitting it would
make the system lose its momentum conservation.

Also returns the two interaction energies, projectile ↔ electrons and
projectile ↔ jellium, which the original code recorded at every step.

Must be called **after** [`forces!`](@ref), whose result it completes rather
than replaces.
"""
function projectile_forces!(cloud::ParticleCloud{T}, proj::Projectile{T},
                            jel::Jellium{T}) where {T}
    p = proj.position
    q = proj.charge
    w = cloud.weight
    soft = proj.softening

    # Projectile ↔ jellium. Inside the background the field grows linearly and
    # no longer depends on the number of ions: only the density matters.
    r2 = sum(abs2, p)
    modf = r2 > jel.radius^2 ? jel.nions * q / r2^T(1.5) :
           q / WIGNER_SEITZ_NA^3
    force = modf .* p
    e_jellium = q * potential(jel, sqrt(r2))

    # Projectile ↔ pseudo-electrons. The shape of the softening is carried by
    # the type of `soft`: the loop does not know which one it applies, and the
    # specialisation happens at compile time.
    coef = -w * q
    e_electrons = zero(T)
    @inbounds for i in eachindex(cloud.positions)
        d = p .- cloud.positions[i]
        d2 = sum(abs2, d)
        f = (coef * force_kernel(soft, d2)) .* d
        force = force .+ f
        cloud.forces[i] = cloud.forces[i] .- f      # reaction
        e_electrons += w * pair_potential(soft, q, sqrt(d2))
    end
    (force, e_electrons, e_jellium)
end

"""
    enclosed_charge(cloud, proj, radius) -> T

Electronic charge contained in a ball of radius `radius` around the projectile
(the Fortran's `capture`, which only ever printed it).

A diagnostic: following this quantity over several radii shows whether the
projectile drags an entourage along.
"""
function enclosed_charge(cloud::ParticleCloud{T}, proj::Projectile{T}, radius) where {T}
    r2 = radius^2
    n = count(p -> sum(abs2, p .- proj.position) < r2, cloud.positions)
    cloud.weight * n
end

"""
    capture!(cloud, proj; radius) -> (ncaptured, internal_energy)

Removes from the cloud the pseudo-particles bound to the projectile — those
closer than `radius` — and reduces its charge accordingly: the ion carries
electrons away.

⚠️ **The softening differs here from that of [`projectile_forces!`](@ref)**, and
it is the original code that decides so. `docapture` uses `2q/c − q·r²/c³` where
`incproj` uses `1.5q/c − 0.5q·r²/c³`. Only the second is the potential of a
uniformly charged ball; the first is continuous at the junction but equals `4/3`
of the other at the centre. Reproduced as is, recorded as anomaly 9.

The energy returned serves only for reporting, which limits the reach of the
discrepancy.
"""
function capture!(cloud::ParticleCloud{T}, proj::Projectile{T};
                  radius::T = T(10)) where {T}
    q, c, w = proj.charge, cutoff(proj), cloud.weight
    vcent, coefcent = 2q / c, -q / c^3
    internal = zero(T)

    keep = Int[]
    for (i, p) in enumerate(cloud.positions)
        r = sqrt(sum(abs2, p .- proj.position))
        if r < radius
            internal += r > c ? -w * q / r : -w * (vcent + r^2 * coefcent)
        else
            push!(keep, i)
        end
    end

    ncaptured = length(cloud) - length(keep)
    if ncaptured > 0
        for field in (cloud.positions, cloud.previous, cloud.forces)
            keepat!(field, keep)
        end
        proj.charge -= w * ncaptured
    end
    (ncaptured, internal)
end

"""
    step!(proj, force, dt) -> Projectile

Advances the projectile by one Verlet step and updates its velocity by a centred
difference — the same as for the pseudo-particles, since it is the only one
consistent with the scheme.
"""
function step!(proj::Projectile{T}, force::NTuple{3,T}, dt::T) where {T}
    new = 2 .* proj.position .- proj.previous .+ (dt^2 / proj.mass) .* force
    proj.velocity = (new .- proj.previous) ./ 2dt
    proj.previous = proj.position
    proj.position = new
    proj
end
