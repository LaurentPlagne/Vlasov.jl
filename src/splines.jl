"""
Base de splines cubiques d'Hermite locales, en 1D.

Chaque nœud `k` de la grille porte **deux** fonctions de base :

  * `Value` : vaut 1 en `g[k]`, de dérivée nulle ;
  * `Slope` : vaut 0 en `g[k]`, de dérivée 1.

Le support d'une fonction de base est `[g[k-1], g[k+1]]` (deux intervalles).
Le code Fortran d'origine encodait ce couple dans un unique entier
`ido = 2k + isig`, d'où les `i/2` et `mod(i,2)` disséminés partout ; ici le
couple est réifié dans [`BasisIndex`](@ref), et l'indice linéaire n'apparaît
plus qu'au moment de remplir une matrice.
"""

"""Nature d'une fonction de base d'Hermite attachée à un nœud."""
@enum HermiteKind Value = 0 Slope = 1

"""
    BasisIndex(knot, kind)

Fonction de base d'Hermite : le nœud `knot` (indexé à partir de 1) et sa
nature `kind`. Type `isbits`, donc sans coût à la construction.
"""
struct BasisIndex
    knot::Int
    kind::HermiteKind
end

"""Indice linéaire (1-based) de la fonction de base, tel qu'utilisé en colonne."""
linearindex(b::BasisIndex) = 2 * (b.knot - 1) + Int(b.kind) + 1

"""Inverse de [`linearindex`](@ref)."""
function BasisIndex(lin::Integer)
    q, r = divrem(lin - 1, 2)
    BasisIndex(q + 1, HermiteKind(r))
end

"""
    SplineAxis(knots, colloc)

Un axe 1D : les nœuds de la grille et les points de collocation associés.

La base compte `2·length(knots)` fonctions, et il y a autant de points de
collocation, ce qui rend carrées les matrices de collocation.
"""
struct SplineAxis{T<:AbstractFloat}
    knots::Vector{T}
    colloc::Vector{T}

    function SplineAxis(knots::Vector{T}, colloc::Vector{T}) where {T}
        issorted(knots) || throw(ArgumentError("les nœuds doivent être croissants"))
        length(colloc) == 2length(knots) ||
            throw(DimensionMismatch("il faut 2 points de collocation par nœud"))
        new{T}(knots, colloc)
    end
end

"""Nombre de nœuds."""
nknots(ax::SplineAxis) = length(ax.knots)

"""Nombre de fonctions de base (= nombre de points de collocation)."""
nbasis(ax::SplineAxis) = 2nknots(ax)

Base.eachindex(ax::SplineAxis) = (BasisIndex(l) for l in 1:nbasis(ax))

# ---------------------------------------------------------------------------
# Formules d'Hermite de référence, sur la variable réduite a ∈ [0,1]
#
# `a` vaut 0 à l'extrémité du support et 1 au nœud porteur ; `σ = ±1` oriente
# le demi-support (droite/gauche). Ce sont les `formu`, `formu1` et `formu2`
# du Fortran.
# ---------------------------------------------------------------------------

@inline function hermite(kind::HermiteKind, a, h, σ)
    kind === Value ? 3a^2 - 2a^3 : h * (a^2 - a^3) * σ
end

@inline function hermite′(kind::HermiteKind, a, h, σ)
    kind === Value ? -6 * (a - a^2) * σ / h : -2a + 3a^2
end

@inline function hermite″(kind::HermiteKind, a, h, σ)
    kind === Value ? 6 * (1 - 2a) / h^2 : (2 - 6a) * σ / h
end

"""
    support(ax, b) -> (xmin, xmax)

Bornes du support de la fonction de base `b`.
"""
function support(ax::SplineAxis, b::BasisIndex)
    g, n = ax.knots, nknots(ax)
    lo = b.knot > 1 ? g[b.knot-1] : g[1]
    hi = b.knot < n ? g[b.knot+1] : g[n]
    (lo, hi)
end

"""
    localcoords(ax, b, x) -> (a, h, σ) ou `nothing`

Coordonnée réduite de `x` dans le support de `b`, ou `nothing` hors support.
"""
@inline function localcoords(ax::SplineAxis{T}, b::BasisIndex, x) where {T}
    g, n, k = ax.knots, nknots(ax), b.knot
    if k > 1 && x <= g[k]          # demi-support gauche, orienté σ = -1
        x < g[k-1] && return nothing
        h = g[k] - g[k-1]
        return ((x - g[k-1]) / h, h, -one(T))
    elseif k < n && x >= g[k]      # demi-support droit, orienté σ = +1
        x > g[k+1] && return nothing
        h = g[k+1] - g[k]
        return ((g[k+1] - x) / h, h, one(T))
    end
    nothing
end

"""
    evaluate(ax, b, x, Val(D)) -> NTuple{D+1}

Valeur de la fonction de base `b` en `x` et ses `D` premières dérivées.
Renvoie des zéros hors du support.
"""
@inline function evaluate(ax::SplineAxis{T}, b::BasisIndex, x, ::Val{D}) where {T,D}
    loc = localcoords(ax, b, x)
    loc === nothing && return ntuple(_ -> zero(T), Val(D + 1))
    a, h, σ = loc
    ntuple(Val(D + 1)) do d
        d == 1 ? hermite(b.kind, a, h, σ) :
        d == 2 ? hermite′(b.kind, a, h, σ) :
        hermite″(b.kind, a, h, σ)
    end
end

"""Valeur de la fonction de base `b` en `x`."""
@inline value(ax::SplineAxis, b::BasisIndex, x) = evaluate(ax, b, x, Val(0))[1]

# ---------------------------------------------------------------------------
# Construction d'un axe
# ---------------------------------------------------------------------------

"""
Nœuds de Gauss-Legendre à 2 points ramenés sur `[0,1]`.

C'est le choix de points de collocation du code d'origine : il rend la
collocation par splines cubiques superconvergente.
"""
const GAUSS2_NODES = ((1 - 1 / sqrt(3)) / 2, (1 + 1 / sqrt(3)) / 2)

"""
    collocation_points(knots) -> Vector

Points de collocation associés à des nœuds : les 2 points de Gauss de chaque
intervalle, encadrés par les deux extrémités du domaine.

Il y en a `2·length(knots)`, autant que de fonctions de base — c'est ce qui
rend carrées les matrices de collocation.
"""
collocation_points(knots::AbstractVector) =
    [knots[1];
     [knots[j] + (knots[j+1] - knots[j]) * u
      for j in 1:(length(knots)-1) for u in GAUSS2_NODES];
     knots[end]]

"""
    uniform_axis(x0, xn, nintervals)

Axe à pas constant sur `[x0, xn]` — la grille fine du code d'origine
(`mkgri` avec une raison géométrique de 1).
"""
function uniform_axis(x0::T, xn::T, nintervals::Integer) where {T<:AbstractFloat}
    knots = collect(range(x0, xn; length = nintervals + 1))
    SplineAxis(knots, collocation_points(knots))
end

"""
    stretch_ratio(h1, L, n) -> a

Raison géométrique `a > 1` telle que `n` pas de raison `a` couvrent la
longueur `L` en **démarrant** par un pas de longueur `h1` :

    L·(1 − a) / (1 − aⁿ) = h1

C'est ce qui raccorde la zone étirée à la zone à pas constant sans rupture de
pas. Résolu par dichotomie jusqu'à épuisement des flottants.

⚠️ Le Fortran (`findacc`) s'arrêtait à une tolérance de `1e-10` : la raison
obtenue ici en diffère d'autant, et les nœuds étirés avec elle. C'est un écart
*attendu*, pas une régression — voir `stretched_axis`.
"""
function stretch_ratio(h1::T, L::T, n::Integer) where {T<:AbstractFloat}
    f(a) = L * (1 - a) / (1 - a^n) - h1
    lo, hi = nextfloat(one(T)), T(10)
    flo = f(lo)
    signbit(flo) == signbit(f(hi)) && throw(ArgumentError(
        "pas de raison géométrique dans ]1, 10] pour h1=$h1, L=$L, n=$n"))
    # Dichotomie jusqu'à épuisement des flottants : l'encadrement est réduit
    # tant qu'il reste un flottant strictement entre les deux bornes.
    while nextfloat(lo) < hi
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        signbit(f(mid)) == signbit(flo) ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end

"""
    stretched_axis(xinner, xouter, n_inner, n_outer)

Axe symétrique à deux zones — la grille grossière du code d'origine
(`mkgri2`) : pas constant sur `[0, xinner]`, puis pas géométriquement
croissant jusqu'à `xouter`, le tout reflété autour de zéro.

Le premier pas étiré vaut exactement le pas constant, ce qui évite une rupture
de maillage à l'interface entre les deux zones.

⚠️ **Ne pas utiliser cet axe pour comparer l'aval à l'oracle.** La raison
géométrique est ici résolue à la précision machine, là où `findacc` s'arrêtait
à `1e-10` : les nœuds diffèrent d'autant, et tout calcul en aval hériterait de
cet écart, masquant les vraies régressions à `1e-15`. Pour valider un opérateur
sur la grille étirée, construire le `SplineAxis` **à partir des nœuds dumpés
par l'oracle**.
"""
function stretched_axis(xinner::T, xouter::T, n_inner::Integer,
                        n_outer::Integer) where {T<:AbstractFloat}
    h1 = xinner / (n_inner - 1)
    L = xouter - xinner
    a = stretch_ratio(h1, L, n_outer)
    step = L * (1 - a) / (1 - a^n_outer)

    # Demi-grille de 0 vers l'extérieur : la zone à pas constant, puis les pas
    # géométriques cumulés.
    half = [range(0, xinner; length = n_inner);
            xinner .+ cumsum(step .* a .^ (0:n_outer-1))]

    # Reflet autour de zéro, le nœud central n'étant pas repris deux fois.
    knots = [-reverse(half[2:end]); half]
    SplineAxis(knots, collocation_points(knots))
end

# ---------------------------------------------------------------------------
# Moments ∫ xᵏ φ(x) dx
#
# Ils portent les conditions aux limites multipolaires (monopôle, dipôle,
# quadrupôle) du solveur de Poisson.
#
# Le Fortran les obtenait par des primitives analytiques écrites à la main
# (`prim`, `primx`, `primx2` et leurs `formp*`, ~290 lignes). On les intègre
# ici par quadrature : `φ` est cubique, donc `x²φ` est de degré 5, et la
# quadrature de Gauss-Legendre à 3 points est **exacte** jusqu'au degré 5.
# Le résultat n'est donc pas une approximation.
# ---------------------------------------------------------------------------

const GAUSS3_NODES = (-sqrt(3 / 5), 0.0, sqrt(3 / 5))
const GAUSS3_WEIGHTS = (5 / 9, 8 / 9, 5 / 9)

"""Intègre `xᵏ φ_b(x)` sur `[a, c]` — exact car l'intégrande est de degré ≤ 5."""
@inline function _gauss3(ax::SplineAxis{T}, b::BasisIndex, a, c, ::Val{k}) where {T,k}
    mid, half = (a + c) / 2, (c - a) / 2
    # `map` sur deux tuples est déplié à la compilation : pas d'itérateur, pas
    # d'accumulateur boxé.
    half * sum(map(GAUSS3_NODES, GAUSS3_WEIGHTS) do ξ, w
        x = mid + half * ξ
        w * x^k * value(ax, b, x)
    end)
end

"""
    moment(ax, b, Val(k)) -> T

Moment d'ordre `k` de la fonction de base `b` : `∫ xᵏ φ_b(x) dx` sur tout son
support. Exact pour `k ≤ 2`.

Chaque demi-support est intégré séparément : `φ_b` y est un polynôme
*différent*, une quadrature unique sur le support entier serait fausse.
"""
function moment(ax::SplineAxis{T}, b::BasisIndex, ::Val{k}) where {T,k}
    g, n, kn = ax.knots, nknots(ax), b.knot
    total = zero(T)
    kn > 1 && (total += _gauss3(ax, b, g[kn-1], g[kn], Val(k)))
    kn < n && (total += _gauss3(ax, b, g[kn], g[kn+1], Val(k)))
    total
end

"""
    moments(ax, Val(k)) -> Vector

Moments d'ordre `k` de toutes les fonctions de base, dans l'ordre des indices
linéaires (les `psx`, `psxx` et `psx2` du Fortran pour `k = 0, 1, 2`).
"""
moments(ax::SplineAxis, ::Val{k}) where {k} =
    [moment(ax, BasisIndex(lin), Val(k)) for lin in 1:nbasis(ax)]

"""Dérivée première de `b` en `x`."""
@inline derivative(ax::SplineAxis, b::BasisIndex, x) = evaluate(ax, b, x, Val(1))[2]

"""Dérivée seconde de `b` en `x`."""
@inline curvature(ax::SplineAxis, b::BasisIndex, x) = evaluate(ax, b, x, Val(2))[3]
