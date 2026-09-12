"""
Solveur tensoriel par diagonalisation rapide (méthode TBSCM).

Résout `(D₁⊗I⊗… + I⊗D₂⊗… + …) X = B` lorsque l'opérateur est **séparable**,
en diagonalisant chaque opérateur 1D :

    Dᵈ = Mᵈ Λᵈ Mᵈ⁻¹   ⟹   X = (⨂ Mᵈ) ∘ diag(1/Σλ) ∘ (⨂ Mᵈ⁻¹) B

Réf. : L. Plagne, J.-Y. Berthou, *Tensorial basis spline collocation method
for Poisson's equation*, J. Comput. Phys. **157**(2), 419-440 (2000).

Généralise `LidJul.PoissonTTSolver` sur deux points :

  * **dimension quelconque** `N` (le code Vlasov est 3D) via un produit
    mode-`d` générique, au lieu du jeu de `transpose` propre au 2D ;
  * **opérateurs non symétriques** — les matrices de collocation spline ne
    sont pas symétriques, contrairement aux laplaciens différences finies.
"""

"""
    DiagonalizedOperator(D)

Diagonalisation d'un opérateur 1D `D` dont le spectre est réel.

Le caractère réel est vérifié à la construction : c'est une propriété
*constatée* de l'opérateur spline, pas une propriété structurelle garantie
par le type, et une régression numérique doit être détectée ici plutôt que
de se propager silencieusement.
"""
struct DiagonalizedOperator{T<:AbstractFloat}
    M::Matrix{T}       # vecteurs propres
    Minv::Matrix{T}    # son inverse
    λ::Vector{T}       # valeurs propres

    function DiagonalizedOperator(D::AbstractMatrix{T}; atol = 1e-10) where {T}
        E = eigen(D)
        imagmax = maximum(abs, imag.(E.values))
        scale = maximum(abs, real.(E.values))
        imagmax <= atol * max(scale, one(T)) || throw(ArgumentError(
            "spectre non réel (max|Im λ| = $imagmax) : le solveur tensoriel " *
            "réel ne s'applique pas"))
        M = real.(E.vectors)
        new{T}(M, inv(M), real.(E.values))
    end
end

Base.size(d::DiagonalizedOperator) = length(d.λ)

"""
    TensorSolver(operators...)

Solveur pour la somme tensorielle des `operators` (un par dimension).

Précalcule `1/(λ¹ᵢ + λ²ⱼ + …)`, qui est le cœur de la méthode : l'inversion
d'un opérateur `N`-dimensionnel se ramène à une division terme à terme.
"""
struct TensorSolver{N,T<:AbstractFloat}
    ops::NTuple{N,DiagonalizedOperator{T}}
    invλsum::Array{T,N}
    work1::Array{T,N}
    work2::Array{T,N}
end

function TensorSolver(ops::DiagonalizedOperator{T}...) where {T}
    N = length(ops)
    dims = ntuple(d -> size(ops[d]), N)

    λsum = [sum(ops[d].λ[I[d]] for d in 1:N) for I in CartesianIndices(dims)]
    singulier = findfirst(iszero, λsum)
    singulier === nothing || throw(ArgumentError(
        "somme de valeurs propres nulle en $(Tuple(singulier)) : opérateur singulier"))

    TensorSolver{N,T}(ops, inv.(λsum),
                      Array{T,N}(undef, dims), Array{T,N}(undef, dims))
end

Base.size(s::TensorSolver) = size(s.invλsum)

"""
    apply_mode!(dest, A, src, d)

Produit mode-`d` : applique la matrice `A` le long de la dimension `d` du
tableau `src`, résultat dans `dest`. Sans allocation.

Le tableau est vu comme `(L, nᵈ, R)` ; pour `d = 1` une seule `mul!` suffit,
sinon on boucle sur les tranches — chacune reste contiguë en mémoire.
"""
function apply_mode!(dest::Array{T,N}, A::AbstractMatrix{T}, src::Array{T,N},
                     d::Integer) where {T,N}
    dims = size(src)
    L = prod(ntuple(i -> dims[i], d - 1))
    n = dims[d]
    R = prod(ntuple(i -> dims[d+i], N - d))

    if L == 1
        mul!(reshape(dest, n, R), A, reshape(src, n, R))
    else
        S = reshape(src, L, n, R)
        D = reshape(dest, L, n, R)
        @views for r in 1:R
            mul!(D[:, :, r], S[:, :, r], transpose(A))
        end
    end
    dest
end

"""
    apply_rotating!(dest, A, src, dims) -> dims permutées

Applique `A` le long de la **première** dimension, en un seul produit
matrice-matrice, et rend les dimensions permutées circulairement.

    mul!(C, Xᵀ, Aᵀ)  calcule  C = (A·X)ᵀ

La transposition n'est pas un détour : c'est elle qui fait la permutation. Le
résultat `(m, n)` se relit tel quel comme un tableau de dimensions
`(d₂, d₃, …, d₁)`, sans déplacer un octet. Appliquer `N` fois ramène donc les
dimensions dans leur ordre initial.

Deux conséquences : **une seule GEMM par dimension** au lieu d'une boucle de
tranches pour les dimensions du milieu, et une forme — un gros produit
matrice-matrice — qui est exactement ce qu'un GPU exécute le mieux.
"""
@inline function apply_rotating!(dest::Array{T,N}, A::AbstractMatrix{T},
                                 src::Array{T,N}, dims::NTuple{N,Int}) where {T,N}
    n = dims[1]
    m = length(src) ÷ n
    mul!(reshape(dest, m, n), transpose(reshape(src, n, m)), transpose(A))
    ntuple(i -> dims[mod1(i + 1, N)], N)
end

"""
    apply_all_rotating!(dest, mats, src, work) -> dest

Applique `mats[d]` le long de chaque dimension, par `N` rotations successives.

Après `N` rotations les dimensions ont retrouvé leur ordre : c'est le seul
motif dont le code ait besoin — le solveur tensoriel l'emploie deux fois, le
passage aux coefficients spline une fois. Un unique noyau BLAS pour toute la
boucle en temps, et un unique endroit à porter sur GPU le jour venu.

`work` est un tampon de la taille de `src`, fourni par l'appelant : cette
fonction n'alloue rien. `dest` doit être distinct de `src`.

⚠️ L'alternance entre `dest` et `work` se déduit de la **parité du nombre
d'étapes restantes**, et non d'un compteur ad hoc. Un ping-pong à un seul
tampon ferait coïncider source et destination dès la deuxième étape — la
lecture et l'écriture se marcheraient dessus, silencieusement.
"""
function apply_all_rotating!(dest::Array{T,N}, mats, src::Array{T,N},
                             work::Array{T,N}) where {T,N}
    dest === src && throw(ArgumentError("`dest` et `src` doivent être distincts"))
    dims = size(src)
    cur = src
    for d in 1:N
        out = iseven(N - d) ? dest : work
        dims = apply_rotating!(out, mats[d], cur, dims)
        cur = out
    end
    dest
end

"""
    solve!(X, B, solver)

Résout l'opérateur tensoriel pour le second membre `B`, résultat dans `X`.
`X` et `B` peuvent être le même tableau.

`2N` produits matrice-matrice et une division terme à terme, sans aucune
allocation : les deux tampons appartiennent au solveur. Les allocations
comptent double ici — elles ne coûtent pas que leur prix, elles déclenchent un
ramasse-miettes qui met les fils à l'arrêt.
"""
function solve!(X::Array{T,N}, B::Array{T,N}, s::TensorSolver{N,T}) where {T,N}
    size(X) == size(B) == size(s) ||
        throw(DimensionMismatch("dimensions incompatibles avec le solveur"))

    # Transformée directe : passage dans la base propre de chaque dimension.
    apply_all_rotating!(s.work1, map(o -> o.Minv, s.ops), B, s.work2)

    # Le cœur de la méthode : l'inversion devient une division terme à terme.
    s.work1 .*= s.invλsum

    # Transformée inverse. `X` peut être `B` : le second membre a déjà été
    # entièrement consommé par la transformée directe.
    apply_all_rotating!(X, map(o -> o.M, s.ops), s.work1, s.work2)
end

"""Version allouante de [`solve!`](@ref)."""
solve(B::Array{T,N}, s::TensorSolver{N,T}) where {T,N} = solve!(similar(B), B, s)
