"""
Bilan d'énergie du système (les `enerele2g` et `enertot2g` du Fortran).

C'est **l'observable de validation** du chapitre 4 : sur un agrégat isolé,
l'énergie totale doit se conserver. Une dérive signale un pas de temps trop
grand, une grille trop lâche, ou une erreur.
"""

"""
    interaction_energy(cloud, fine, csol_fine, coarse, csol_coarse, sm; escaped, enclosed) -> T

Somme `Σᵢ w·Φ(rᵢ)` sur les pseudo-particules — l'énergie d'interaction d'une
distribution avec le potentiel `Φ` donné en coefficients spline.

Mêmes trois régimes que [`forces!`](@ref), et pour la même raison : une
particule doit voir le même potentiel dans le bilan d'énergie que dans les
forces, sans quoi les deux ne parlent pas du même système.
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
                # Hors des deux grilles : le potentiel de la charge enfermée.
                v === nothing ? enclosed / sqrt(p[1]^2 + p[2]^2 + p[3]^2) : v
            end
            total += w * φ
        end
        total
    end
end

"""
    ion_self_energy(jel) -> T

Énergie électrostatique propre du fond de jellium, `3N²/5r₀` — celle d'une
boule uniformément chargée. Constante au cours d'une simulation, mais elle
entre dans le total.
"""
ion_self_energy(jel::Jellium) = 3 * jel.nions^2 / (5 * jel.radius)

"""
    EnergyBudget(total, kinetic, hartree, meanfield, ions)

Décomposition de l'énergie du système à un instant donné.

  * `hartree` — `½∫ρΦ_H`, répulsion des électrons entre eux ;
  * `meanfield` — `∫ρ(Φ_xc + Φ_jel)`, échange-corrélation et attraction du
    fond ionique ;
  * `ions` — l'énergie propre du jellium, constante ;
  * `total` — leur somme avec l'énergie cinétique.

Le facteur ½ du terme de Hartree et son absence sur `meanfield` ne sont pas
une étourderie : le premier compte une interaction **entre** électrons, qui
serait sinon comptée deux fois ; le second une interaction avec un fond
extérieur.
"""
struct EnergyBudget{T<:AbstractFloat}
    total::T
    kinetic::T
    hartree::T
    meanfield::T
    ions::T
end

"""
    hartree_energy(cloud, …) -> T

`½Σᵢ w·Φ_H(rᵢ)`, à évaluer sur le potentiel de Hartree **seul**, avant que
l'échange-corrélation et le jellium n'y soient ajoutés.
"""
hartree_energy(args...; kwargs...) = interaction_energy(args...; kwargs...) / 2

"""
    energy_budget(cloud, jellium, kinetic, hartree, total_interaction) -> EnergyBudget

Assemble le bilan à partir des trois quantités mesurées séparément :
l'énergie cinétique rendue par [`step!`](@ref), l'énergie de Hartree évaluée
sur le potentiel nu, et l'interaction évaluée sur le potentiel **total**.

Le terme de champ moyen s'obtient par différence — `∫ρΦ_tot − 2·(½∫ρΦ_H)` —
et non par une intégrale séparée : c'est ainsi que procède le Fortran, et cela
évite de ré-échantillonner le potentiel une troisième fois.

⚠️ **Question ouverte sur l'ordre des appels du Fortran.** `pspech2` est
l'exacte opposée de `pspech` (`ech = −ech`), et la boucle en temps appelle la
seconde après `move`, juste avant `enertot2g` : le potentiel devrait donc être
revenu à Hartree seul, et `∫ρΦ_tot − 2·enele` valoir ~0. Les nombres disent
l'inverse — `potel = −30.5` pour `enele = 989.8`, soit un terme de champ moyen
de `−2010`, ordre de grandeur attendu pour l'attraction du jellium
(`−N²/r₀ ≈ −1650`) augmentée de l'échange-corrélation.

Le bilan reproduit donc l'oracle au chiffre près sur les mêmes entrées, mais
la lecture de l'enchaînement reste à confirmer. À trancher en instrumentant la
boucle, avant de s'appuyer sur ces énergies pour conclure quoi que ce soit de
physique.
"""
function energy_budget(jel::Jellium{T}, kinetic::T, hartree::T,
                       total_interaction::T) where {T}
    meanfield = total_interaction - 2hartree
    ions = ion_self_energy(jel)
    EnergyBudget{T}(ions + kinetic + hartree + meanfield,
                    kinetic, hartree, meanfield, ions)
end
