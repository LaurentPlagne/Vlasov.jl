"""
Pseudo-particules et intégration temporelle.

La méthode échantillonne la densité d'espace des phases par des
pseudo-particules, chacune représentant `weight` électrons : sa masse et sa
charge sont celles de ces `weight` électrons réunis.

L'intégration est un Verlet en position : la vitesse n'est jamais stockée,
elle se lit dans l'écart entre deux positions successives.
"""

"Masse de l'électron, en unités atomiques (`mel` du Fortran)."
const ELECTRON_MASS = 1.0

"Charge de l'électron, en unités atomiques (`qel` du Fortran)."
const ELECTRON_CHARGE = -1.0

"""
    ParticleCloud(positions, previous, forces, weight)

Nuage de pseudo-particules pour un schéma de Verlet en position.

`previous` porte les positions au pas précédent — c'est ce qui tient lieu de
vitesse. `weight` est le nombre d'électrons que représente chaque
pseudo-particule.

Le Fortran stockait tout cela dans des tableaux `(3, npartmax)` en ordre
colonne, ce qui est exactement la disposition mémoire d'un
`Vector{NTuple{3,T}}` : le portage est une réinterprétation, pas une
conversion.
"""
struct ParticleCloud{T<:AbstractFloat}
    positions::Vector{NTuple{3,T}}
    previous::Vector{NTuple{3,T}}
    forces::Vector{NTuple{3,T}}
    weight::T
end

function ParticleCloud(positions::Vector{NTuple{3,T}}, weight::T) where {T}
    n = length(positions)
    zero3 = ntuple(_ -> zero(T), 3)
    ParticleCloud{T}(copy(positions), fill(zero3, n), fill(zero3, n), weight)
end

Base.length(c::ParticleCloud) = length(c.positions)

"""Masse d'une pseudo-particule : celle des `weight` électrons qu'elle porte."""
mass(c::ParticleCloud) = ELECTRON_MASS * c.weight

"""Charge d'une pseudo-particule."""
charge(c::ParticleCloud) = ELECTRON_CHARGE * c.weight

"""Produit vectoriel de deux triplets."""
@inline cross3(a, b) = (a[2] * b[3] - a[3] * b[2],
                        a[3] * b[1] - a[1] * b[3],
                        a[1] * b[2] - a[2] * b[1])

"""
    step!(cloud, dt; rcmax = Inf) -> (; kinetic, escaped, angular)

Avance le nuage d'un pas de temps par Verlet en position :

    q(t+dt) = 2q(t) − q(t−dt) + dt²·F/M

et renvoie les diagnostics que le Fortran calculait dans la même boucle —
énergie cinétique, moment cinétique total, et la part de l'énergie cinétique
portée par les particules **sorties**. Ils sont obtenus du moment centré
`p = M·(q(t+dt) − q(t−dt))/2dt`, qui n'existe qu'ici : le calculer après coup
demanderait de conserver un état de plus.

`rcmax` est le rayon au-delà duquel une particule compte comme sortie, mesuré
sur la position **avant** le pas — comme le `move` du Fortran, qui teste
`ract` sur `qp` et non sur la position nouvelle. Il valait `100.d0` en dur
jusqu'en 1997, et devient un paramètre d'entrée en 1998. Le défaut `Inf` ne
compte rien comme sorti, ce qui laisse `escaped` nul pour qui ne s'en sert pas.
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
        # Comparaison des carrés : une racine par particule pour un simple
        # seuil, c'est une racine de trop.
        q[1]^2 + q[2]^2 + q[3]^2 > r2max && (eout += e)
        angular = angular .+ cross3(q, p)
        cloud.previous[i] = q
        cloud.positions[i] = qnew
    end
    (; kinetic = ekin, escaped = eout, angular)
end

"""
    half_step_back(positions, momenta, M, dt)

Amorce du leapfrog, premier temps : `q(−dt/2) = q(0) − (dt/2M)·p`.

C'est le `moveback1` du Fortran, où le tableau des positions précédentes
portait encore les impulsions issues du tirage initial.
"""
half_step_back(positions, momenta, M, dt) =
    map((q, p) -> q .- (dt / 2M) .* p, positions, momenta)

"""
    full_step_back(positions, half, forces, M, dt)

Amorce du leapfrog, second temps : de `q(0)` et `q(−dt/2)` vers `q(−dt)`, en
utilisant les forces évaluées en `q(−dt/2)`.

⚠️ **Reproduit un coefficient douteux du code d'origine.** `moveback2` calcule

    coef2 = 0.5·dltt*2·npart/(mel·nbelec)

soit `dt/M`. Or `coef2·F` est alors une *vitesse*, ajoutée à des longueurs :
la formule n'est pas homogène. Un développement de Taylor donne `dt²/4M`, et
la routine voisine `move` écrit bien `dltt**2`. Tout indique une coquille
`dltt*2` pour `dltt**2` — mais elle est présente à l'identique dans les
**cinq** versions du code de la thèse, donc jamais corrigée.

L'effet est ponctuel : il ne fausse que l'amorçage, comme une erreur sur la
vitesse initiale. On le reproduit tel quel pour rester comparable à l'oracle ;
`consistent = true` sélectionne la variante homogène `dt²/4M`, pour mesurer
ce que la coquille coûte.
"""
function full_step_back(positions, half, forces, M, dt; consistent::Bool = false)
    coef = consistent ? dt^2 / 4M : dt / M
    map((q, h, f) -> .-q .+ 2 .* h .+ coef .* f, positions, half, forces)
end

# Pas de `kinetic_energy(cloud, dt)` autonome : le moment centré en t se lit
# entre q(t+dt) et q(t−dt), qui ne coexistent qu'à l'intérieur d'un pas. Une
# fonction d'après-coup ne verrait que q(t) et q(t−dt) et rendrait le moment
# en t−dt/2 — une autre grandeur, sous le même nom. `step!` la renvoie.
