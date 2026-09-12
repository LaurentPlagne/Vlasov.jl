"""
L'ion projectile et son interaction avec l'agrégat.

Objet du chapitre 6 : un proton traverse l'agrégat, et la grandeur mesurée est
l'**énergie qu'il y perd** — le pouvoir d'arrêt.

Le projectile n'est pas une charge ponctuelle mais une boule uniformément
chargée de rayon `cutoff`. Ce n'est pas une commodité numérique gratuite :
sans cela une pseudo-particule passant au contact subirait une force infinie,
alors qu'elle représente en réalité un paquet d'électrons étalé.
"""

"""
    Projectile(; mass, charge, energy, impact, x0, dt, cutoff)

Ion incident, intégré par le même schéma de Verlet que les pseudo-particules.

Il entre par `x = x0` avec le paramètre d'impact `impact` porté par `y`, à la
vitesse que lui donne son énergie cinétique `energy`. `initial_energy` est
conservée : c'est la référence dont on soustraira l'énergie courante pour
obtenir la perte.
"""
mutable struct Projectile{T<:AbstractFloat}
    const mass::T
    const charge::T
    const cutoff::T
    const initial_energy::T
    position::NTuple{3,T}
    previous::NTuple{3,T}
    velocity::NTuple{3,T}
end

function Projectile(; mass::Real, charge::Real, energy::Real, impact::Real = 0,
                    x0::Real, dt::Real, cutoff::Real)
    # Le type se déduit des arguments plutôt que d'être un paramètre : un
    # défaut annoté `impact::T = zero(T)` référencerait `T` avant qu'il soit lié.
    T = float(promote_type(typeof(mass), typeof(charge), typeof(energy),
                           typeof(impact), typeof(x0), typeof(dt), typeof(cutoff)))
    v = (sqrt(2 * T(energy) / T(mass)), zero(T), zero(T))
    position = (T(x0), T(impact), zero(T))
    Projectile{T}(T(mass), T(charge), T(cutoff), T(energy),
                  position, position .- T(dt) .* v, v)
end

"""Énergie cinétique courante du projectile."""
kinetic_energy(p::Projectile) = p.mass * sum(abs2, p.velocity) / 2

"""
    energy_loss(p) -> T

Énergie perdue par le projectile depuis son entrée, en unités atomiques —
positive quand il freine. C'est **l'observable du chapitre 6**.
"""
energy_loss(p::Projectile) = p.initial_energy - kinetic_energy(p)

"Conversion unités atomiques → électron-volts, comme dans le code d'origine."
const HARTREE_TO_EV = 27.2116

"""
    projectile_forces!(cloud, proj, jellium) -> (force, e_electrons, e_jellium)

Force totale sur le projectile, et **réaction** ajoutée aux forces des
pseudo-particules — l'interaction est réciproque, et l'omettre ferait perdre
au système sa conservation de l'impulsion.

Renvoie aussi les deux énergies d'interaction, projectile ↔ électrons et
projectile ↔ jellium, que le code d'origine consignait à chaque pas.

Doit être appelée **après** [`forces!`](@ref), dont elle complète le résultat
au lieu de le remplacer.
"""
function projectile_forces!(cloud::ParticleCloud{T}, proj::Projectile{T},
                            jel::Jellium{T}) where {T}
    p = proj.position
    q = proj.charge
    w = cloud.weight
    cut2 = proj.cutoff^2

    # Projectile ↔ jellium. À l'intérieur du fond, le champ croît linéairement
    # et ne dépend plus du nombre d'ions : seule compte la densité.
    r2 = sum(abs2, p)
    modf = r2 > jel.radius^2 ? jel.nions * q / r2^T(1.5) :
           q / WIGNER_SEITZ_NA^3
    force = modf .* p
    e_jellium = q * potential(jel, sqrt(r2))

    # Projectile ↔ pseudo-électrons, avec adoucissement sous `cutoff`.
    coef = -w * q
    e_electrons = zero(T)
    @inbounds for i in eachindex(cloud.positions)
        d = p .- cloud.positions[i]
        d2 = sum(abs2, d)
        # Sous le rayon d'adoucissement, la force reste linéaire en distance,
        # comme à l'intérieur d'une boule uniformément chargée.
        m = coef / (d2 > cut2 ? d2^T(1.5) : cut2^T(1.5))
        f = m .* d
        force = force .+ f
        cloud.forces[i] = cloud.forces[i] .- f      # réaction
        e_electrons += w * uniform_sphere_potential(q, proj.cutoff, sqrt(d2))
    end
    (force, e_electrons, e_jellium)
end

"""
    step!(proj, force, dt) -> Projectile

Avance le projectile d'un pas de Verlet et met à jour sa vitesse par
différence centrée — la même que pour les pseudo-particules, car c'est la
seule qui soit cohérente avec le schéma.
"""
function step!(proj::Projectile{T}, force::NTuple{3,T}, dt::T) where {T}
    new = 2 .* proj.position .- proj.previous .+ (dt^2 / proj.mass) .* force
    proj.velocity = (new .- proj.previous) ./ 2dt
    proj.previous = proj.position
    proj.position = new
    proj
end
