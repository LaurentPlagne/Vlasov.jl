"""
Dépôt des pseudo-particules sur la grille, et passage valeurs ↔ coefficients.

Deux représentations d'un même champ coexistent dans cette méthode, et les
confondre est *le* bug classique :

  * **valeurs aux points de collocation** — ce que produit le dépôt ;
  * **coefficients spline** — ce sur quoi s'intègrent les moments.

On passe des unes aux autres par `S⁻¹` appliqué dans chaque direction.
"""

"""
    dual_lengths(ax) -> Vector

Longueur de la cellule duale de chaque point de collocation : la part du
domaine qu'il « possède ». Sert à normaliser un dépôt en densité.

Les quatre points extrêmes sont traités à part — leur cellule est bornée par
le bord du domaine, pas par un point de collocation voisin.
"""
function dual_lengths(ax::SplineAxis)
    gt, g = ax.colloc, ax.knots
    m = length(gt)
    map(1:m) do j
        j == 1     ? (gt[2] - g[1]) / 2 :
        j == 2     ? (gt[3] - g[1]) / 2 :
        j == m     ? (g[end] - gt[m-1]) / 2 :
        j == m - 1 ? (g[end] - gt[m-2]) / 2 :
                     (gt[j+1] - gt[j-1]) / 2
    end
end

"""
    locate(ax, x) -> (cell, weight) ou `nothing`

Repère `x` entre deux points de collocation consécutifs : renvoie l'indice du
point de gauche et le poids qui lui revient (celui de droite reçoit
`1 - weight`). Renvoie `nothing` hors du domaine.
"""
@inline function locate(ax::SplineAxis{T}, x) where {T}
    gt = ax.colloc
    (x < gt[1] || x > gt[end]) && return nothing
    c = max(1, searchsortedfirst(gt, x) - 1)
    (c, (gt[c+1] - x) / (gt[c+1] - gt[c]))
end

"""
    deposit!(ρ, mesh, positions; charge) -> nout

Dépose des pseudo-particules de poids `charge` sur les points de collocation
du maillage, par interpolation trilinéaire (« cloud-in-cell »), et normalise
par le volume dual de chaque nœud pour obtenir une densité.

`ρ` a la taille **complète** de la grille de collocation (bords compris), pas
celle du problème intérieur. Renvoie le nombre de particules tombées hors
domaine, qui sont ignorées.
"""
function deposit!(ρ::Array{T,3}, mesh::SplineMesh{3,T},
                  positions; charge::T) where {T}
    mx, my, mz = mesh.axes
    size(ρ) == (nbasis(mx), nbasis(my), nbasis(mz)) ||
        throw(DimensionMismatch("ρ doit couvrir toute la grille de collocation"))
    fill!(ρ, zero(T))

    nout = 0
    for p in positions
        lx = locate(mx, p[1])
        ly = locate(my, p[2])
        lz = locate(mz, p[3])
        if lx === nothing || ly === nothing || lz === nothing
            nout += 1
            continue
        end
        (i, ax), (j, ay), (k, az) = lx, ly, lz
        bx, by, bz = 1 - ax, 1 - ay, 1 - az
        @inbounds begin
            ρ[i, j, k]         += ax * ay * az
            ρ[i, j, k+1]       += ax * ay * bz
            ρ[i, j+1, k]       += ax * by * az
            ρ[i, j+1, k+1]     += ax * by * bz
            ρ[i+1, j, k]       += bx * ay * az
            ρ[i+1, j, k+1]     += bx * ay * bz
            ρ[i+1, j+1, k]     += bx * by * az
            ρ[i+1, j+1, k+1]   += bx * by * bz
        end
    end

    # Normalisation en densité. Le volume dual est le produit extérieur des
    # trois longueurs duales : le broadcast le parcourt sans jamais le
    # matérialiser en 3D, là où le Fortran en gardait 58³ flottants (`volm1`).
    lx, ly, lz = map(dual_lengths, (mx, my, mz))
    ρ .*= charge ./ (lx .* ly' .* reshape(lz, 1, 1, :))
    nout
end

"""
    spline_coefficients!(c, ρ, mesh)

Passe des valeurs aux points de collocation aux coefficients spline, en
appliquant `S⁻¹` dans chaque direction (le `tensrus2` du Fortran).
"""
function spline_coefficients!(c::Array{T,N}, ρ::Array{T,N},
                              mesh::SplineMesh{N,T}) where {T,N}
    # Même noyau que le solveur tensoriel : `N` rotations, `N` produits
    # matrice-matrice, aucune tranche. Le tampon vient du maillage plutôt que
    # d'une allocation de 1,4 Mo à chaque appel.
    apply_all_rotating!(c, map(cm -> cm.Sinv, mesh.collocation), ρ, mesh.scratch[3])
end

"""Version allouante de [`spline_coefficients!`](@ref)."""
spline_coefficients(ρ::Array{T,N}, mesh::SplineMesh{N,T}) where {T,N} =
    spline_coefficients!(similar(ρ), ρ, mesh)

"""
    total_charge(ρ, mesh) -> T

Charge totale `∫ρ dV`, obtenue en contractant les coefficients spline avec les
moments d'ordre 0 de chaque direction.

C'est le contrôle de conservation du dépôt : déposer `N` électrons doit rendre
`N`, à la précision de l'interpolation près.
"""
function total_charge(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T}
    # Comme pour `multipole` : les moments duaux portent `S⁻ᵀ`, la densité se
    # contracte telle quelle et ses coefficients n'ont pas à être formés.
    px, py, pz = map(m -> m[1], mesh.dual_moments)
    s = zero(T)
    @inbounds for k in eachindex(pz), j in eachindex(py), i in eachindex(px)
        s += ρ[i, j, k] * px[i] * py[j] * pz[k]
    end
    s
end
