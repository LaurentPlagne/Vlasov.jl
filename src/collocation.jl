"""
Matrices de collocation d'un axe de splines.

`S[k, i] = φᵢ(τₖ)` où `τₖ` est le k-ième point de collocation et `φᵢ` la
i-ième fonction de base. Idem pour les dérivées première (`S′`) et seconde
(`S″`). Ces matrices sont **pentadiagonales** et **non symétriques** : les
lignes indexent des points, les colonnes des fonctions.
"""

"""Demi-largeur de bande des matrices de collocation d'Hermite."""
const COLLOCATION_BANDWIDTH = (2, 2)

"""
    CollocationMatrices(axis)

Assemble `S`, `S′` et `S″` pour un axe donné, sous forme de matrices bande.

Les deux points de collocation intérieurs à chaque intervalle ne « voient »
que les 4 fonctions de base des deux nœuds qui le bordent : la structure est
pentadiagonale, et le stockage bande la rend explicite plutôt que subie.
Les deux lignes extrêmes encodent les conditions au bord.
"""
struct CollocationMatrices{T,M<:AbstractMatrix{T}}
    axis::SplineAxis{T}
    S::M
    S′::M
    S″::M
    "Inverse de `S` : passage des valeurs aux points de collocation aux
     coefficients spline. Dense et matérialisé, car appliqué en produit
     tensoriel à chaque pas de temps. `cond(S) ≈ 4`, l'inversion est sûre."
    Sinv::Matrix{T}
end

function CollocationMatrices(ax::SplineAxis{T}) where {T}
    m = nbasis(ax)
    S, S′, S″ = (BandedMatrix(Zeros{T}(m, m), COLLOCATION_BANDWIDTH) for _ in 1:3)

    # Un intervalle `j` relie les nœuds j et j+1 ; il porte les points de
    # collocation 2j et 2j+1, et les 4 fonctions de base d'indices linéaires
    # 2j-1 … 2j+2. Tous ces couples (ligne, colonne) tiennent dans la bande.
    for j in 1:(nknots(ax)-1), lin in (2j-1):(2j+2), k in (2j):(2j+1)
        S[k, lin], S′[k, lin], S″[k, lin] =
            evaluate(ax, BasisIndex(lin), ax.colloc[k], Val(2))
    end

    # Conditions au bord : première et dernière ligne.
    first_b, last_b = BasisIndex(1), BasisIndex(m - 1)
    S[1, 1], S′[1, 1], S″[1, 1] = evaluate(ax, first_b, ax.colloc[1], Val(2))
    S[m, m-1], S′[m, m-1], S″[m, m-1] = evaluate(ax, last_b, ax.colloc[m], Val(2))

    # `inv(Matrix(S))` et non `inv(S)` : voir l'avertissement de `laplacian1d`
    # sur les divisions de `BandedMatrix`.
    CollocationMatrices{T,typeof(S)}(ax, S, S′, S″, inv(Matrix(S)))
end

"""
    laplacian1d(cm) -> Matrix

Opérateur de dérivée seconde exprimé **dans l'espace des valeurs aux points
de collocation** : `D = S″ · S⁻¹`.

La division exploite la structure bande de `S` (factorisation `O(n·b²)`),
mais le quotient est plein : `D` est rendu dense, ce qu'attend de toute façon
la diagonalisation en aval.

⚠️ La factorisation `lu` est explicite, et le numérateur est densifié
**avant** la division, pour deux raisons mesurées sur `BandedMatrices` v1.12.0 :

  * `S″ / S` entre deux `BandedMatrix` rend un résultat **faux** sans rien
    signaler — résidu `‖D·S − S″‖/‖S″‖ ≈ 0.3` alors que `cond(S) ≈ 4` ;
  * `S″ / lu(S)` avec un numérateur **bande** ne termine pas.

Seul `Matrix(S″) / lu(S)` est à la fois correct et terminant. La division à
gauche (`S \\ v`) est correcte, elle ; c'est la division à droite qui est en
cause.

Les conditions de Dirichlet sont appliquées en retirant la première et la
dernière fonction de base (c'était `extract` en Fortran).

`D` n'est pas symétrique, mais son spectre est réel et strictement négatif —
c'est ce qui rend licite la diagonalisation réelle utilisée par le solveur
tensoriel.
"""
laplacian1d(cm::CollocationMatrices) = laplacian1d_full(cm)[2:end-1, 2:end-1]

"""
    laplacian1d_full(cm) -> Matrix

L'opérateur avant retrait des fonctions de base du bord.

Ses deux colonnes extrêmes sont ce qui permet de **relever** des conditions de
Dirichlet non homogènes : elles disent ce que vaut l'opérateur appliqué aux
deux fonctions de base retirées, dont la valeur est imposée. Les deux lignes
extrêmes, elles, n'approchent aucune dérivée seconde.
"""
laplacian1d_full(cm::CollocationMatrices) = Matrix(cm.S″) / lu(cm.S)
