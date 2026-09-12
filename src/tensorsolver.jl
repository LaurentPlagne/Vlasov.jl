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

    invλsum = Array{T,N}(undef, dims)
    for I in CartesianIndices(invλsum)
        s = sum(ops[d].λ[I[d]] for d in 1:N)
        iszero(s) && throw(ArgumentError(
            "somme de valeurs propres nulle en $(Tuple(I)) : opérateur singulier"))
        invλsum[I] = inv(s)
    end

    TensorSolver{N,T}(ops, invλsum,
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
    solve!(X, B, solver)

Résout l'opérateur tensoriel pour le second membre `B`, résultat dans `X`.
`X` et `B` peuvent être le même tableau.
"""
function solve!(X::Array{T,N}, B::Array{T,N}, s::TensorSolver{N,T}) where {T,N}
    size(X) == size(B) == size(s) ||
        throw(DimensionMismatch("dimensions incompatibles avec le solveur"))

    # Transformée directe : passage dans la base propre de chaque dimension.
    src, dst = B, s.work1
    for d in 1:N
        apply_mode!(dst, s.ops[d].Minv, src, d)
        src, dst = dst, (d == 1 ? s.work2 : src)
    end

    # Le cœur de la méthode : l'inversion devient une division terme à terme.
    src .*= s.invλsum

    # Transformée inverse ; la dernière étape écrit directement dans X.
    for d in 1:N
        out = (d == N) ? X : dst
        apply_mode!(out, s.ops[d].M, src, d)
        src, dst = out, src
    end
    X
end

"""Version allouante de [`solve!`](@ref)."""
solve(B::Array{T,N}, s::TensorSolver{N,T}) where {T,N} = solve!(similar(B), B, s)
