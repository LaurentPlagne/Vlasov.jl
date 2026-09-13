"""
Energy budget of the system (the Fortran's `enerele2g` and `enertot2g`).

This is the **validation observable** of chapter 4: on an isolated cluster the
total energy must be conserved. A drift signals a time step that is too large, a
grid that is too loose, or a mistake.
"""

"""
    interaction_energy(cloud, fine, csol_fine, coarse, csol_coarse, sm; escaped, enclosed) -> T

Sums `Σᵢ w·Φ(rᵢ)` over the pseudo-particles — the interaction energy of a
distribution with the potential `Φ` given as spline coefficients.

Same three regimes as [`forces!`](@ref), and for the same reason: a particle
must see the same potential in the energy budget as in the forces, otherwise the
two are not describing the same system.
"""
function interaction_energy(cloud::ParticleCloud{T},
                            fine::NTuple{3,SplineAxis{T}}, csol_fine::Array{T,3},
                            coarse::NTuple{3,SplineAxis{T}}, csol_coarse::Array{T,3},
                            sm::GaussianSmoothing{T};
                            enclosed::T = zero(T)) where {T}
    w = cloud.weight
    lo = ntuple(d -> fine[d].knots[3], 3)
    hi = ntuple(d -> fine[d].knots[end-2], 3)

    tmapreduce(length(cloud.positions)) do slice
        total = zero(T)
        @inbounds for i in slice
            p = cloud.positions[i]
            φ = if all(d -> lo[d] < p[d] < hi[d], 1:3)
                smoothed_potential(fine, csol_fine, sm, p)
            else
                v = spline_potential(coarse, csol_coarse, p)
                # Outside both grids: the potential of the enclosed charge.
                v === nothing ? enclosed / sqrt(p[1]^2 + p[2]^2 + p[3]^2) : v
            end
            total += w * φ
        end
        total
    end
end

"""
    ion_self_energy(jel) -> T

Electrostatic self-energy of the jellium background, `3N²/5r₀` — that of a
uniformly charged ball. Constant over a simulation, but it enters the total.
"""
ion_self_energy(jel::Jellium) = 3 * jel.nions^2 / (5 * jel.radius)

"""
    EnergyBudget(total, kinetic, hartree, meanfield, ions)

Decomposition of the system's energy at a given instant.

  * `hartree` — `½∫ρΦ_H`, repulsion of the electrons among themselves;
  * `meanfield` — `∫ρ(Φ_xc + Φ_jel)`, exchange-correlation and attraction by the
    ionic background;
  * `ions` — the jellium self-energy, constant;
  * `total` — their sum together with the kinetic energy.

The factor ½ on the Hartree term and its absence on `meanfield` are not an
oversight: the first counts an interaction **between** electrons, which would
otherwise be counted twice; the second an interaction with an external
background.
"""
struct EnergyBudget{T<:AbstractFloat}
    total::T
    kinetic::T
    hartree::T
    meanfield::T
    ions::T
    """Share of `kinetic` carried by particles beyond `rcmax` — the Fortran's
    `ekinout`. Zero when the caller supplies no radius: it is an evaporation
    diagnostic and enters no sum."""
    escaped::T
end

"""
    hartree_energy(cloud, …) -> T

`½Σᵢ w·Φ_H(rᵢ)`, to be evaluated on the Hartree potential **alone**, before
exchange-correlation and jellium are added to it.
"""
hartree_energy(args...; kwargs...) = interaction_energy(args...; kwargs...) / 2

"""
    energy_budget(cloud, jellium, kinetic, hartree, total_interaction) -> EnergyBudget

Assembles the budget from the three quantities measured separately: the kinetic
energy returned by [`step!`](@ref), the Hartree energy evaluated on the bare
potential, and the interaction evaluated on the **total** potential.

The mean-field term is obtained by difference — `∫ρΦ_tot − 2·(½∫ρΦ_H)` — and not
by a separate integral: that is how the Fortran proceeds, and it avoids
resampling the potential a third time.

📋 **On the Fortran's ordering, once an open question, now settled.** `pspech2`
is *not* the exact opposite of `pspech`: it contains **two** loops and therefore
adds to `csol` twice — the first undoes `pspech`, the second adds the
exchange-correlation **energy** density. `csol` is thus not meant to return to
Hartree alone; it is converted from *potential* to *energy*, which is what
`enertot2g` requires. Verified: the port reproduces ‖csol‖ = 737.4 for Hartree
alone and 7.8 after `pspech`, exactly.

The genuine defect lies elsewhere, in that second loop: `rr` is computed only
inside the `ρ > 1e-7` branch, so the `else` branch evaluates the jellium
potential at the radius of an *earlier* point — over 90 % of the fine grid.
Fixed by the author in the 1998-01-05 version. See `docs/coquilles-fortran.md`,
anomaly 8.
"""
function energy_budget(jel::Jellium{T}, kinetic::T, hartree::T,
                       total_interaction::T, escaped::T = zero(T)) where {T}
    meanfield = total_interaction - 2hartree
    ions = ion_self_energy(jel)
    EnergyBudget{T}(ions + kinetic + hartree + meanfield,
                    kinetic, hartree, meanfield, ions, escaped)
end
