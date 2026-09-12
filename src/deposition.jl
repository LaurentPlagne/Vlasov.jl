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
    scatter!(kernel, ρ, mesh, positions, buffers) -> nout

Moteur commun aux deux dépôts. `kernel(dest, ax, ay, az, p)` dépose une
particule et rend `false` si elle est hors domaine.

Sans `buffers`, la boucle est séquentielle. Avec, chaque fil accumule dans son
propre tableau puis l'on somme : c'est la seule façon de paralléliser une
diffusion sans perdre de contributions ni payer d'atomiques.
"""
function scatter!(kernel, ρ::Array{T,3}, mesh::SplineMesh{3,T},
                  positions, buffers) where {T}
    mx, my, mz = mesh.axes
    size(ρ) == (nbasis(mx), nbasis(my), nbasis(mz)) ||
        throw(DimensionMismatch("ρ doit couvrir toute la grille de collocation"))

    if buffers === nothing
        fill!(ρ, zero(T))
        nout = 0
        for p in positions
            kernel(ρ, mx, my, mz, p) || (nout += 1)
        end
        return nout
    end

    parts = chunks(length(positions), min(length(buffers.slots), Threads.nthreads()))
    nouts = zeros(Int, length(parts))
    Threads.@threads for c in eachindex(parts)
        dest = buffers.slots[c]
        fill!(dest, zero(T))
        n = 0
        @inbounds for i in parts[c]
            kernel(dest, mx, my, mz, positions[i]) || (n += 1)
        end
        nouts[c] = n
    end
    scatter_reduce!(ρ, buffers, length(parts))
    sum(nouts)
end

"""
    ScatterBuffers(mesh)

Un tableau de densité **par fil**, pour paralléliser le dépôt.

Le dépôt est une *diffusion* : plusieurs particules écrivent dans la même
case, et découper naïvement la boucle ferait perdre des contributions. Chaque
fil accumule donc dans son propre tableau, et l'on somme à la fin — la somme
coûte `nfils × N³` additions, négligeable devant le dépôt lui-même.

⚠️ **Le coût mémoire croît comme le cube de la grille.** À 58³ et huit fils,
c'est 12 Mo ; à 128³ ce serait 134 Mo. Les tampons sont donc créés
explicitement par l'appelant, jamais en douce.

Piste connue pour aller plus loin quand les particules se désordonnent : les
**trier par maille** avant de déposer, pour que des particules voisines
écrivent dans des cases voisines. C'est ce que faisait le code de la thèse, en
parallèle (méthode PSRS).
"""
struct ScatterBuffers{T,N}
    slots::Vector{Array{T,N}}
end

function ScatterBuffers(mesh::SplineMesh{N,T}; nslots = Threads.nthreads()) where {N,T}
    dims = map(nbasis, mesh.axes)
    ScatterBuffers{T,N}([Array{T,N}(undef, dims) for _ in 1:nslots])
end

"""
    scatter_reduce!(ρ, buffers, nused) -> ρ

Somme les `nused` premiers tampons dans `ρ`.
"""
function scatter_reduce!(ρ::Array{T,N}, b::ScatterBuffers{T,N}, nused::Integer) where {T,N}
    copyto!(ρ, b.slots[1])
    for c in 2:nused
        ρ .+= b.slots[c]
    end
    ρ
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
                  positions; charge::T, buffers = nothing) where {T}
    mx, my, mz = mesh.axes
    size(ρ) == (nbasis(mx), nbasis(my), nbasis(mz)) ||
        throw(DimensionMismatch("ρ doit couvrir toute la grille de collocation"))
    nout = scatter!(ρ, mesh, positions, buffers) do dest, mx, my, mz, p
        lx = locate(mx, p[1])
        ly = locate(my, p[2])
        lz = locate(mz, p[3])
        (lx === nothing || ly === nothing || lz === nothing) && return false
        (i, ax), (j, ay), (k, az) = lx, ly, lz
        bx, by, bz = 1 - ax, 1 - ay, 1 - az
        @inbounds begin
            dest[i, j, k]       += ax * ay * az
            dest[i, j, k+1]     += ax * ay * bz
            dest[i, j+1, k]     += ax * by * az
            dest[i, j+1, k+1]   += ax * by * bz
            dest[i+1, j, k]     += bx * ay * az
            dest[i+1, j, k+1]   += bx * ay * bz
            dest[i+1, j+1, k]   += bx * by * az
            dest[i+1, j+1, k+1] += bx * by * bz
        end
        true
    end

    # Normalisation en densité. Le volume dual est le produit extérieur des
    # trois longueurs duales : le broadcast le parcourt sans jamais le
    # matérialiser en 3D, là où le Fortran en gardait 58³ flottants (`volm1`).
    #
    # ⚠️ Ces trois noms ne doivent PAS coïncider avec ceux du `do`-block
    # ci-dessus. Une variable assignée dans une fermeture ET dans la fonction
    # englobante n'en fait qu'une seule, que Julia boxe : les huit fils
    # écriraient alors dans la même case. Le symptôme était un dépôt
    # non déterministe, juste à un fil et faux à huit.
    wx, wy, wz = map(dual_lengths, (mx, my, mz))
    ρ .*= charge ./ (wx .* wy' .* reshape(wz, 1, 1, :))
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
