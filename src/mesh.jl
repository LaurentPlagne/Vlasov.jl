"""
Maillage produit tensoriel et résolution de Poisson associée.

Le code d'origine dupliquait à la main tout l'appareillage d'une grille à
l'autre — 41 variables suffixées `big` dans le programme principal, et 47
arguments passés à `static`. Ici une seule structure, instanciée une fois par
niveau de grille.
"""

"""
    SplineMesh(axes...)

Maillage produit tensoriel de `N` axes de splines, avec tout ce que la
résolution de Poisson en tire : matrices de collocation par direction,
opérateur de dérivée seconde diagonalisé, et solveur tensoriel.

Construire un maillage fait le gros du travail (assemblage, factorisations,
diagonalisations) une fois pour toutes ; les résolutions qui suivent sont
ensuite peu coûteuses. C'est ce qui rend la méthode intéressante pour une
simulation où seul le second membre change à chaque pas de temps.
"""
struct SplineMesh{N,T,M}
    axes::NTuple{N,SplineAxis{T}}
    collocation::NTuple{N,CollocationMatrices{T,M}}
    "Opérateurs de dérivée seconde **complets**, bords compris : leurs colonnes
     extrêmes servent au relèvement des conditions de Dirichlet."
    laplacians::NTuple{N,Matrix{T}}
    "Moments `∫φ`, `∫xφ`, `∫x²φ` de chaque direction, **déjà transformés par
     `S⁻ᵀ`**. Ils permettent d'intégrer une densité donnée aux points de
     collocation sans jamais former ses coefficients spline — voir `multipole`."
    dual_moments::NTuple{N,NTuple{3,Vector{T}}}
    solver::TensorSolver{N,T}
    """Tampons de travail, pleine grille et grille intérieure.

    Les allocations comptent double dans une boucle en temps parallèle : elles
    ne coûtent pas que leur prix, elles déclenchent un ramasse-miettes qui met
    tous les fils à l'arrêt. Un maillage est donc **réutilisable mais pas
    partageable** entre fils."""
    scratch::NTuple{3,Array{T,N}}
    scratch_inner::Array{T,N}
end

function SplineMesh(axes::SplineAxis{T}...) where {T}
    cms = map(CollocationMatrices, axes)
    full = map(laplacian1d_full, cms)
    ops = map(D -> DiagonalizedOperator(D[2:end-1, 2:end-1]), full)
    # `S⁻ᵀ·m` une fois pour toutes : c'est ce qui dispense d'appliquer `S⁻¹`
    # au tableau 3D à chaque intégration.
    duals = map(cms) do cm
        ntuple(k -> transpose(cm.Sinv) * moments(cm.axis, Val(k - 1)), 3)
    end
    solver = TensorSolver(ops...)
    full_dims = map(nbasis, axes)
    SplineMesh(axes, cms, full, duals, solver,
               ntuple(_ -> Array{T,length(axes)}(undef, full_dims), 3),
               Array{T,length(axes)}(undef, size(solver)))
end

"""
    NestedMeshes(levels...)

Hiérarchie de grilles emboîtées, **de la plus fine à la plus grossière**.

Chaque niveau doit être strictement contenu dans le suivant : la grille fine
résout l'agrégat, la grossière porte les conditions au loin, et les valeurs de
bord de l'une sont lues dans la solution de l'autre.

C'est ce type qui remplace les 41 variables suffixées `big` du programme
principal. Le nombre de niveaux étant un paramètre, la version à trois grilles
du code d'origine n'exige aucune structure de plus.
"""
struct NestedMeshes{L,N,T,M}
    levels::NTuple{L,SplineMesh{N,T,M}}

    function NestedMeshes(levels::SplineMesh{N,T,M}...) where {N,T,M}
        L = length(levels)
        L >= 1 || throw(ArgumentError("il faut au moins une grille"))
        for l in 1:(L-1), d in 1:N
            inner, outer = levels[l].axes[d], levels[l+1].axes[d]
            outer.knots[1] <= inner.knots[1] && inner.knots[end] <= outer.knots[end] ||
                throw(ArgumentError(
                    "le niveau $l déborde du niveau $(l+1) dans la direction $d"))
        end
        new{L,N,T,M}(levels)
    end
end

Base.length(::NestedMeshes{L}) where {L} = L
Base.getindex(n::NestedMeshes, l::Integer) = n.levels[l]
Base.iterate(n::NestedMeshes, s = 1) = s > length(n) ? nothing : (n.levels[s], s + 1)

"""Le niveau le plus fin."""
finest(n::NestedMeshes) = n.levels[1]

"""Le niveau le plus grossier, celui qui porte les conditions au loin."""
coarsest(n::NestedMeshes) = n.levels[end]

"""Nombre de dimensions du maillage."""
Base.ndims(::SplineMesh{N}) where {N} = N

"""
Dimensions du problème **intérieur**, c'est-à-dire après retrait des fonctions
de base portant les conditions de Dirichlet. C'est la taille des tableaux que
[`solve!`](@ref) attend.
"""
Base.size(mesh::SplineMesh) = size(mesh.solver)
Base.size(mesh::SplineMesh, d::Integer) = size(mesh)[d]

"""
    collocation_axes(mesh)

Points de collocation de chaque direction, restreints à l'intérieur — les
coordonnées auxquelles un second membre doit être échantillonné.
"""
collocation_axes(mesh::SplineMesh) = map(ax -> ax.colloc[2:end-1], mesh.axes)

"""
    solve!(φ, ρ, mesh)

Résout `∇²φ = ρ` aux points de collocation intérieurs, avec conditions de
Dirichlet homogènes. `φ` et `ρ` peuvent être le même tableau.
"""
solve!(φ::Array{T,N}, ρ::Array{T,N}, mesh::SplineMesh{N,T}) where {T,N} =
    solve!(φ, ρ, mesh.solver)

"""Version allouante de [`solve!`](@ref)."""
solve(ρ::Array{T,N}, mesh::SplineMesh{N,T}) where {T,N} = solve!(similar(ρ), ρ, mesh)

"""
    laplacian!(dest, φ, mesh)

Applique l'opérateur `∇² = Σ_d D_d` — l'opération directe, dont
[`solve!`](@ref) est l'inverse. Sert à contrôler un résidu.
"""
function laplacian!(dest::Array{T,N}, φ::Array{T,N}, mesh::SplineMesh{N,T}) where {T,N}
    fill!(dest, zero(T))
    tmp = similar(dest)
    for d in 1:N
        @views apply_mode!(tmp, mesh.laplacians[d][2:end-1, 2:end-1], φ, d)
        dest .+= tmp
    end
    dest
end
