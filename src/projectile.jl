"""
L'ion projectile et son interaction avec l'agrégat.

Objet du chapitre 6 : un proton traverse l'agrégat, et la grandeur mesurée est
l'**énergie qu'il y perd** — le pouvoir d'arrêt.

L'interaction à courte distance doit être régularisée — sans quoi une
pseudo-particule passant au contact subirait une force infinie, alors qu'elle
représente un paquet d'électrons étalé. **Comment** on la régularise n'est pas
un détail : la thèse écrit que la perte d'énergie en dépend fortement. D'où
[`Softening`](@ref) et ses deux réalisations.
"""

"""
    Softening{T}

Régularisation de l'interaction projectile ↔ pseudo-particule à courte
distance. Sans elle, une pseudo-particule passant au contact subirait une
force infinie, alors qu'elle représente un paquet d'électrons étalé.

Deux formes, et **elles ne sont pas d'accord** :

  * [`BallSoftening`](@ref) — ce que fait le Fortran, dans ses 43 versions ;
  * [`GaussianSoftening`](@ref) — ce que décrit la thèse, équation
    (`Eforceproj2`) du chapitre 2.

La thèse écrit que « la perte d'énergie des ions traversant l'agrégat dépend
**fortement** de ce lissage », et que `σ_ion = 1` a été choisi pour reproduire
le modèle de Lindhard. Le choix n'est donc pas un détail d'implémentation : il
fixe le pouvoir d'arrêt. D'où un type, et non un `if`.
"""
abstract type Softening{T<:AbstractFloat} end

"""
    BallSoftening(radius)

Boule uniformément chargée de rayon `radius` : Coulomb au-delà, force linéaire
en deçà. C'est ce qu'implémente `forceproji` du Fortran (le `cutoff` de
`vlas.inp`), et donc ce contre quoi l'oracle valide.
"""
struct BallSoftening{T} <: Softening{T}
    radius::T
end

"""
    GaussianSoftening(σ)

Charge gaussienne de largeur `σ` contre charge ponctuelle — le `σ_ion` de la
thèse. Le potentiel de paire est

    V(r) = Erf(r / (√2 σ)) / r,     V(0) = √(2/π) / σ

et la force en dérive :

    f⃗ = −Q·Q′ · [Erf(r/(√2σ)) − 2 g(r) r] / r³ · r⃗,
    g(r) = exp(−r²/2σ²) / (√(2π) σ)

`g` est la gaussienne **normalisée à une dimension** : c'est ce qui rend
`2g(r)·r` sans dimension, comme l'exige la parenthèse.

⚠️ Le Fortran contient `erfsr`, qui est exactement `Erf(r/√2σ)/r` — **jamais
appelée**, dans aucune version. Voir `docs/coquilles-fortran.md`.
"""
struct GaussianSoftening{T} <: Softening{T}
    σ::T
end

Softening(s::Softening) = s

"""
    force_kernel(s, r2) -> T

Facteur `m` tel que la force vaille `Q·Q′·m·r⃗` — le vecteur non normalisé.
`m` a donc la dimension d'un inverse de volume, et `m → 1/r³` au loin.
"""
@inline force_kernel(s::BallSoftening{T}, r2::T) where {T} =
    r2 > s.radius^2 ? inv(r2 * sqrt(r2)) : inv(s.radius^3)

# Coefficients du développement de [Erf(u/√2) − u·e^{−u²/2}·√(2/π)] / u³ en
# puissances de u², à un facteur √(2/π) près. Les deux termes s'annulent à
# l'ordre dominant : les soustraire tels quels perd tous les chiffres quand
# `u` est petit, et c'est précisément le régime des collisions frontales.
const _GAUSS_SERIES = (1/3, -1/10, 1/56, -1/432, 1/4224, -1/49920, 1/685440)
const _SQRT_2_OVER_PI = sqrt(2 / π)

@inline function force_kernel(s::GaussianSoftening{T}, r2::T) where {T}
    σ = s.σ
    u2 = r2 / σ^2
    if u2 <= T(0.25)                      # u ≤ 0.5 : série, pas de soustraction
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

Potentiel créé en `r` par une charge `q` régularisée par `s`.
"""
@inline pair_potential(s::BallSoftening{T}, q, r) where {T} =
    uniform_sphere_potential(q, s.radius, r)

@inline function pair_potential(s::GaussianSoftening{T}, q, r) where {T}
    x = r / (sqrt(T(2)) * s.σ)
    q * (r < eps(T)^(1//3) * s.σ ? T(_SQRT_2_OVER_PI) / s.σ : erf(x) / r)
end

"Rayon caractéristique — `radius` pour la boule, `σ` pour la gaussienne."
scale(s::BallSoftening) = s.radius
scale(s::GaussianSoftening) = s.σ

"""
    Projectile(; mass, charge, energy, impact, x0, dt, cutoff)
    Projectile(; …, softening = BallSoftening(cutoff))

Ion incident, intégré par le même schéma de Verlet que les pseudo-particules.

Il entre par `x = x0` avec le paramètre d'impact `impact` porté par `y`, à la
vitesse que lui donne son énergie cinétique `energy`. `initial_energy` est
conservée : c'est la référence dont on soustraira l'énergie courante pour
obtenir la perte.
"""
mutable struct Projectile{T<:AbstractFloat,S<:Softening{T}}
    const mass::T
    "Charge courante. Elle **diminue** si le projectile capture des électrons."
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
        "fournir `cutoff` (boule, le Fortran) **ou** `softening` (la thèse), pas les deux"))
    # Le type se déduit des arguments plutôt que d'être un paramètre : un
    # défaut annoté `impact::T = zero(T)` référencerait `T` avant qu'il soit lié.
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

"""Rayon caractéristique de l'adoucissement — le `cutoff` d'autrefois."""
cutoff(p::Projectile) = scale(p.softening)

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
    soft = proj.softening

    # Projectile ↔ jellium. À l'intérieur du fond, le champ croît linéairement
    # et ne dépend plus du nombre d'ions : seule compte la densité.
    r2 = sum(abs2, p)
    modf = r2 > jel.radius^2 ? jel.nions * q / r2^T(1.5) :
           q / WIGNER_SEITZ_NA^3
    force = modf .* p
    e_jellium = q * potential(jel, sqrt(r2))

    # Projectile ↔ pseudo-électrons. La forme de l'adoucissement est portée
    # par le type de `soft` : la boucle ne sait pas laquelle elle applique, et
    # la spécialisation se fait à la compilation.
    coef = -w * q
    e_electrons = zero(T)
    @inbounds for i in eachindex(cloud.positions)
        d = p .- cloud.positions[i]
        d2 = sum(abs2, d)
        f = (coef * force_kernel(soft, d2)) .* d
        force = force .+ f
        cloud.forces[i] = cloud.forces[i] .- f      # réaction
        e_electrons += w * pair_potential(soft, q, sqrt(d2))
    end
    (force, e_electrons, e_jellium)
end

"""
    enclosed_charge(cloud, proj, radius) -> T

Charge électronique contenue dans une boule de rayon `radius` autour du
projectile (le `capture` du Fortran, qui n'en faisait qu'un affichage).

Diagnostic : suivre cette quantité sur plusieurs rayons montre si le
projectile entraîne un cortège.
"""
function enclosed_charge(cloud::ParticleCloud{T}, proj::Projectile{T}, radius) where {T}
    r2 = radius^2
    n = count(p -> sum(abs2, p .- proj.position) < r2, cloud.positions)
    cloud.weight * n
end

"""
    capture!(cloud, proj; radius) -> (ncaptured, internal_energy)

Retire du nuage les pseudo-particules liées au projectile — celles à moins de
`radius` — et diminue d'autant sa charge : l'ion emporte des électrons.

⚠️ **L'adoucissement diffère ici de celui de [`projectile_forces!`](@ref)**, et
c'est le code d'origine qui en décide ainsi. `docapture` emploie
`2q/c − q·r²/c³` là où `incproj` emploie `1.5q/c − 0.5q·r²/c³`. Seule la
seconde est le potentiel d'une boule uniformément chargée ; la première est
continue au raccord mais vaut `4/3` de l'autre au centre. Reproduit tel quel,
consigné comme anomalie 9.

L'énergie rendue ne sert qu'au compte rendu, ce qui limite la portée de
l'écart.
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
