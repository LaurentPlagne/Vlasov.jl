"""
Tirage initial des pseudo-particules dans l'espace des phases (le
`initialise` du Fortran).

Modèle de Thomas-Fermi : la position suit la densité radiale de l'agrégat, et
l'impulsion est tirée uniformément dans la sphère de Fermi **locale**, dont le
rayon `p_F = (3π²ρ)^⅓` dépend de la densité au point tiré. C'est ce qui donne
à la distribution son caractère fermionique de départ — celui dont la thèse
montre qu'il survit à la dynamique.
"""

"Coefficient du moment de Fermi : `p_F = (3π²ρ)^⅓`."
const FERMI_COEFFICIENT = cbrt(3 * π^2)

"""
    PhaseSpaceProfile{T}

Ce qu'il faut savoir de l'agrégat au repos pour en tirer un état initial.

Deux réalisations, qui sont deux âges du code : [`RadialProfile`](@ref) inverse
une densité radiale tabulée (version portée, 1997), [`PotentialProfile`](@ref)
rejette dans l'espace des phases contre un potentiel auto-cohérent (version
cible, 1998). Elles décrivent le même état de Thomas-Fermi par deux chemins ;
[`sample_thomas_fermi`](@ref) les distingue, tout le reste les ignore.
"""
abstract type PhaseSpaceProfile{T<:AbstractFloat} end

"""
    RadialProfile(quantiles, density, rmax)

Le profil radial de l'agrégat, sous les deux formes dont le tirage a besoin :

  * `quantiles` — le rayon en fonction de la fraction cumulée de charge, soit
    la **réciproque** de la fonction de répartition. Tirer un rayon se réduit
    alors à y interpoler un nombre uniforme (`hm1.dat`).
  * `density` — la densité `ρ(r)` échantillonnée de 0 à `rmax`, qui fixe le
    moment de Fermi local (`rhoinit.dat`).
"""
struct RadialProfile{T<:AbstractFloat} <: PhaseSpaceProfile{T}
    quantiles::Vector{T}
    density::Vector{T}
    rmax::T
end

"""
    read_radial_profile(dir) -> RadialProfile

Lit `hm1.dat` et `rhoinit.dat` dans `dir`. Chaque fichier commence par son
nombre de points ; `rhoinit.dat` donne ensuite son rayon maximal.
"""
function read_radial_profile(dir::AbstractString)
    readvals(io, n) = [parse(Float64, strip(readline(io))) for _ in 1:n]
    quantiles = open(joinpath(dir, "hm1.dat")) do io
        readvals(io, parse(Int, strip(readline(io))))
    end
    density, rmax = open(joinpath(dir, "rhoinit.dat")) do io
        n = parse(Int, strip(readline(io)))
        rmax = parse(Float64, strip(readline(io)))
        (readvals(io, n), rmax)
    end
    RadialProfile(quantiles, density, rmax)
end

"""Interpolation linéaire de `v` sur une grille régulière, à la position réduite `u ∈ [0,1)`."""
@inline function _interp_regular(v::Vector{T}, u, n) where {T}
    j = min(floor(Int, u * (n - 1)) + 1, n - 1)
    # Forme littérale du Fortran : les deux bornes puis le rapport, et non
    # `u*(n-1) - (j-1)`, mathématiquement identique mais pas au bit près.
    a = (j - 1) / (n - 1)
    b = j / (n - 1)
    v[j] + (v[j+1] - v[j]) * (u - a) / (b - a)
end

"""
    sample_thomas_fermi(profile, npart, weight; rng) -> (positions, momenta)

Tire `npart` pseudo-particules de poids `weight` électrons chacune.

Six nombres uniformes par particule, dans l'ordre du Fortran : rayon, deux
angles de position, module de l'impulsion, deux angles d'impulsion. Cet ordre
compte — il faut consommer la suite de `rng` exactement comme l'oracle pour
pouvoir comparer particule par particule.

Le module de l'impulsion est tiré en `x^⅓·p_F`, ce qui peuple uniformément le
volume de la sphère de Fermi et non son rayon. Les impulsions rendues sont
celles des **pseudo-particules**, donc multipliées par `weight`.
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

        # Densité au rayon tiré, sur une grille régulière de 0 à rmax.
        ρ = _interp_regular(profile.density, r / rmax, nρ)
        p = cbrt(x[4]) * FERMI_COEFFICIENT * cbrt(ρ)
        momenta[i] = _on_sphere(p * weight, x[5], x[6])
    end
    positions, momenta
end

"""
    PotentialProfile(grid, potential, rmax, pmax, fermi)

Le potentiel radial auto-cohérent de l'agrégat, tel que le lit `initialise4`
(version 1998-01-05) dans `pot.dat` — voir [`read_potential_profile`](@ref).

Il remplace [`RadialProfile`](@ref) : au lieu d'inverser une densité tabulée,
la version tardive tire dans l'espace des phases et rejette ce qui dépasse le
niveau de Fermi. `rmax` et `pmax` bornent la boîte de tirage, `fermi` est le
critère d'acceptation.
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

Lit un `pot.dat` : le nombre d'intervalles, puis `rmax pmax E_F`, puis
`nbgrid+1` triplets dont on retient la première et la **troisième** colonne
— c'est ce que fait `initialise4`, la deuxième étant ignorée.

⚠️ Aucun `pot.dat` d'origine n'a survécu dans l'archive de la thèse ; celui de
`ref/fortran98/` est reconstruit. Voir `ref/fortran98/README.md`.
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
    sample_thomas_fermi(profile::PotentialProfile, npart, weight; rng)

Tirage par **rejet dans l'espace des phases**, le `initialise4` de 1998.

On tire un point uniformément dans la boule de rayon `rmax` et une impulsion
uniformément dans celle de rayon `pmax` (d'où les `∛u`, qui peuplent le volume
et non le rayon), et on garde le couple si `p²/2 + V(r) < E_F`. L'ensemble
accepté est exactement `{E < E_F}` : la distribution de Thomas-Fermi, sans
passer par une densité intermédiaire.

Les **six** uniformes sont retirés à chaque rejet, y compris les angles qui ne
servent pas au test. C'est ce que fait le Fortran, et c'est ce qui détermine
la consommation du flux de `rng` — donc la comparabilité particule à particule
avec l'oracle.
"""
function sample_thomas_fermi(profile::PotentialProfile{T}, npart::Integer,
                             weight::T; rng::Ran2 = Ran2(-1)) where {T}
    (; grid, potential, rmax, pmax, fermi) = profile
    n = length(grid) - 1              # nombre d'intervalles
    scale = n / rmax
    positions = Vector{NTuple{3,T}}(undef, npart)
    momenta = Vector{NTuple{3,T}}(undef, npart)

    for i in 1:npart
        local x, r, p
        while true
            x = ntuple(_ -> T(next!(rng)), 6)
            r = rmax * cbrt(x[1])
            p = pmax * cbrt(x[4])
            # Indice de cellule à la façon du Fortran : `int(scale*r - 1e-10)`,
            # décalé de 1 pour l'indexation Julia.
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

"""Point de norme `r` sur la sphère, à partir de deux uniformes — `2u−1` pour
le cosinus polaire, ce qui répartit uniformément en surface."""
@inline function _on_sphere(r, uφ, uμ)
    μ = 2 * uμ - 1
    s = sqrt(1 - μ^2)
    φ = 2 * π * uφ
    (r * cos(φ) * s, r * sin(φ) * s, r * μ)
end

"""
    initial_cloud(profile, npart, nbelec, dt; rng) -> ParticleCloud

Nuage prêt pour le schéma de Verlet : positions tirées, et positions
précédentes amorcées d'un demi-pas en arrière depuis les impulsions.

C'est l'enchaînement `makeinit` puis `moveback1` du Fortran. Le second pas
d'amorçage ([`full_step_back`](@ref)) demande les forces, donc un potentiel :
il n'a pas sa place ici.
"""
function initial_cloud(profile::PhaseSpaceProfile{T}, npart::Integer, nbelec::T,
                       dt::T; rng::Ran2 = Ran2(-1)) where {T}
    weight = nbelec / npart
    positions, momenta = sample_thomas_fermi(profile, npart, weight; rng)
    cloud = ParticleCloud(positions, weight)
    copyto!(cloud.previous, half_step_back(positions, momenta, ELECTRON_MASS * weight, dt))
    cloud
end
