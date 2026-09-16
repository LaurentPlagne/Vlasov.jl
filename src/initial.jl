"""
Initial sampling of the pseudo-particles in phase space (the Fortran's
`initialise`).

Thomas-Fermi model: the position follows the cluster's radial density, and the
momentum is drawn uniformly inside the **local** Fermi sphere, whose radius
`p_F = (3π²ρ)^⅓` depends on the density at the sampled point. That is what gives
the distribution its initial fermionic character — the one the thesis shows
survives the dynamics.
"""

"Coefficient of the Fermi momentum: `p_F = (3π²ρ)^⅓`."
const FERMI_COEFFICIENT = cbrt(3 * π^2)

"""
    PhaseSpaceProfile{T}

What one needs to know about the cluster at rest in order to draw an initial
state from it.

Two realisations, which are two ages of the code: [`RadialProfile`](@ref)
inverts a tabulated radial density (the ported version, 1997),
[`PotentialProfile`](@ref) rejects in phase space against a self-consistent
potential (the target version, 1998). They describe the same Thomas-Fermi state
by two routes; [`sample_thomas_fermi`](@ref) tells them apart, everything else
ignores the difference.
"""
abstract type PhaseSpaceProfile{T<:AbstractFloat} end

"""
    RadialProfile(quantiles, density, rmax)

The cluster's radial profile, in the two forms the sampling needs:

  * `quantiles` — the radius as a function of the cumulative charge fraction,
    that is the **inverse** of the distribution function. Drawing a radius then
    reduces to interpolating a uniform number in it (`hm1.dat`).
  * `density` — the density `ρ(r)` sampled from 0 to `rmax`, which sets the
    local Fermi momentum (`rhoinit.dat`).
"""
struct RadialProfile{T<:AbstractFloat} <: PhaseSpaceProfile{T}
    quantiles::Vector{T}
    density::Vector{T}
    rmax::T
end

"""
    read_radial_profile(dir) -> RadialProfile

Reads `hm1.dat` and `rhoinit.dat` from `dir`. Each file begins with its number
of points; `rhoinit.dat` then gives its maximum radius.
"""
function read_radial_profile(dir::AbstractString)
    read_radial_profile(joinpath(dir, "hm1.dat"), joinpath(dir, "rhoinit.dat"))
end

function read_radial_profile(hm1_path::AbstractString, rhoinit_path::AbstractString)
    readvals(io, n) = [parse(Float64, strip(readline(io))) for _ in 1:n]
    quantiles = open(hm1_path) do io
        readvals(io, parse(Int, strip(readline(io))))
    end
    density, rmax = open(rhoinit_path) do io
        n = parse(Int, strip(readline(io)))
        rmax = parse(Float64, strip(readline(io)))
        (readvals(io, n), rmax)
    end
    RadialProfile(quantiles, density, rmax)
end

"""Linear interpolation of `v` on a regular grid, at the reduced position `u ∈ [0,1)`."""
@inline function _interp_regular(v::Vector{T}, u, n) where {T}
    j = min(floor(Int, u * (n - 1)) + 1, n - 1)
    # The Fortran's literal form: the two bounds then the ratio, and not
    # `u*(n-1) - (j-1)`, mathematically identical but not bit for bit.
    a = (j - 1) / (n - 1)
    b = j / (n - 1)
    v[j] + (v[j+1] - v[j]) * (u - a) / (b - a)
end

"""
    sample_thomas_fermi(profile, npart, weight; rng) -> (positions, momenta)

Draws `npart` pseudo-particles, each of weight `weight` electrons.

Six uniform numbers per particle, in the Fortran's order: radius, two position
angles, momentum magnitude, two momentum angles. That order matters — the `rng`
stream must be consumed exactly as the oracle consumes it for a
particle-by-particle comparison to be possible.

The momentum magnitude is drawn as `x^⅓·p_F`, which populates the volume of the
Fermi sphere uniformly rather than its radius. The momenta returned are those of
the **pseudo-particles**, hence multiplied by `weight`.
"""
function sample_thomas_fermi(profile::RadialProfile{T}, npart::Integer,
                             weight::T; rng::Ran2 = Ran2(-1)) where {T}
    nq, nρ = length(profile.quantiles), length(profile.density)
    rmax = profile.rmax
    positions = Vector{NTuple{3,T}}(undef, npart)
    momenta = Vector{NTuple{3,T}}(undef, npart)

    for i in 1:npart
        x = ntuple(_ -> T(next!(rng)), 6)

        r = _interp_regular(profile.quantiles, x[1], nq)
        positions[i] = _on_sphere(r, x[2], x[3])

        # Density at the sampled radius, on a regular grid from 0 to rmax.
        ρ = _interp_regular(profile.density, r / rmax, nρ)
        p = cbrt(x[4]) * FERMI_COEFFICIENT * cbrt(ρ)
        momenta[i] = _on_sphere(p * weight, x[5], x[6])
    end
    positions, momenta
end

"""
    PotentialProfile(grid, potential, rmax, pmax, fermi)

The cluster's self-consistent radial potential, as `initialise4` (version
1998-01-05) reads it from `pot.dat` — see [`read_potential_profile`](@ref).

It replaces [`RadialProfile`](@ref): instead of inverting a tabulated density,
the later version samples in phase space and rejects whatever exceeds the Fermi
level. `rmax` and `pmax` bound the sampling box, `fermi` is the acceptance
criterion.
"""
struct PotentialProfile{T<:AbstractFloat} <: PhaseSpaceProfile{T}
    grid::Vector{T}
    potential::Vector{T}
    rmax::T
    pmax::T
    fermi::T
end

"""
    read_potential_profile(path) -> PotentialProfile

Reads a `pot.dat`: the number of intervals, then `rmax pmax E_F`, then
`nbgrid+1` triples of which the first and the **third** column are kept — that
is what `initialise4` does, the second being ignored.

⚠️ No original `pot.dat` survived in the thesis archive; the one in
`ref/fortran98/` is reconstructed. See `ref/fortran98/README.md`.
"""
function read_potential_profile(path::AbstractString)
    open(path) do io
        n = parse(Int, strip(readline(io)))
        rmax, pmax, fermi = parse.(Float64, split(strip(readline(io))))
        cols = [parse.(Float64, split(strip(readline(io)))) for _ in 0:n]
        PotentialProfile(first.(cols), last.(cols), rmax, pmax, fermi)
    end
end

"""
    PotentialProfile(grid, density)

Builds the sampling profile from a tabulated **radial density**.

At Thomas-Fermi equilibrium, `p_F(r)² / 2 + V(r) = E_F` everywhere: setting
`V(r) = E_F − p_F(r)²/2` with `p_F = (3π²ρ)^⅓` makes the rejection criterion
`p²/2 + V(r) < E_F` equivalent to `p < p_F(r)`, which is the very definition of
the local Fermi sphere. `E_F` drops out of the inequality — we take it to be
zero.

This is what makes it possible to start from an archived `rhorad.dat`, the only
thing that survives for some of the thesis's clusters, where `initialise4`
expected a self-consistent `pot.dat` that did not.
"""
function PotentialProfile(grid::AbstractVector, density::AbstractVector)
    length(grid) == length(density) ||
        throw(DimensionMismatch("grid and density have different lengths"))
    T = float(promote_type(eltype(grid), eltype(density)))
    pF = [T(FERMI_COEFFICIENT) * cbrt(max(T(ρ), zero(T))) for ρ in density]
    PotentialProfile{T}(collect(T, grid), -pF .^ 2 ./ 2,
                        T(last(grid)), maximum(pF), zero(T))
end

"""
    read_radial_density(path; column = 2) -> (grid, density)

Reads a `rhorad.dat`: columns of reals with no header, the first being the
radius. Negative densities — tail-end noise — are clamped to zero.
"""
function read_radial_density(path::AbstractString; column::Integer = 2)
    rows = [parse.(Float64, split(l)) for l in eachline(path) if !isempty(strip(l))]
    (first.(rows), [max(r[column], 0.0) for r in rows])
end

"""
    sample_thomas_fermi(profile::PotentialProfile, npart, weight; rng)

Sampling by **rejection in phase space**, the `initialise4` of 1998.

A point is drawn uniformly inside the ball of radius `rmax` and a momentum
uniformly inside the one of radius `pmax` (hence the `∛u`, which populate the
volume rather than the radius), and the pair is kept if `p²/2 + V(r) < E_F`. The
accepted set is exactly `{E < E_F}`: the Thomas-Fermi distribution, without going
through an intermediate density.

All **six** uniforms are redrawn on every rejection, including the angles that
play no part in the test. That is what the Fortran does, and it is what
determines the consumption of the `rng` stream — hence particle-by-particle
comparability with the oracle.
"""
function sample_thomas_fermi(profile::PotentialProfile{T}, npart::Integer,
                             weight::T; rng::Ran2 = Ran2(-1)) where {T}
    (; grid, potential, rmax, pmax, fermi) = profile
    n = length(grid) - 1              # number of intervals
    scale = n / rmax
    positions = Vector{NTuple{3,T}}(undef, npart)
    momenta = Vector{NTuple{3,T}}(undef, npart)

    for i in 1:npart
        local x, r, p
        while true
            x = ntuple(_ -> T(next!(rng)), 6)
            r = rmax * cbrt(x[1])
            p = pmax * cbrt(x[4])
            # Cell index the Fortran's way: `int(scale*r - 1e-10)`, shifted by
            # one for Julia's indexing.
            j = trunc(Int, scale * r - 1e-10) + 1
            v = potential[j] + (potential[j+1] - potential[j]) *
                               (r - grid[j]) / (grid[j+1] - grid[j])
            p^2 / 2 + v < fermi && break
        end
        positions[i] = _on_sphere(r, x[2], x[3])
        momenta[i] = _on_sphere(p * weight, x[5], x[6])
    end
    positions, momenta
end

"""Point of norm `r` on the sphere, from two uniforms — `2u−1` for the polar
cosine, which spreads points uniformly over the surface."""
@inline function _on_sphere(r, uφ, uμ)
    μ = 2 * uμ - 1
    s = sqrt(1 - μ^2)
    φ = 2 * π * uφ
    (r * cos(φ) * s, r * sin(φ) * s, r * μ)
end

"""
    initial_cloud(profile, npart, nbelec, dt; rng) -> ParticleCloud

Cloud ready for the Verlet scheme: positions drawn, and previous positions
primed half a step backwards from the momenta.

This is the Fortran's `makeinit` followed by `moveback1`. The second priming
stage ([`full_step_back`](@ref)) needs the forces, hence a potential: it has no
place here.
"""
function initial_cloud(profile::PhaseSpaceProfile{T}, npart::Integer, nbelec::T,
                       dt::T; rng::Ran2 = Ran2(-1)) where {T}
    weight = nbelec / npart
    positions, momenta = sample_thomas_fermi(profile, npart, weight; rng)
    cloud = ParticleCloud(positions, weight)
    copyto!(cloud.previous, half_step_back(positions, momenta, ELECTRON_MASS * weight, dt))
    cloud
end
