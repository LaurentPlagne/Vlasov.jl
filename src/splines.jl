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

"""Dérivée première de `b` en `x`."""
@inline derivative(ax::SplineAxis, b::BasisIndex, x) = evaluate(ax, b, x, Val(1))[2]

"""Dérivée seconde de `b` en `x`."""
@inline curvature(ax::SplineAxis, b::BasisIndex, x) = evaluate(ax, b, x, Val(2))[3]
