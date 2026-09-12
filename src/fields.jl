"""
Champ électrique vu par les pseudo-particules, et forces qui en découlent.

Une pseudo-particule n'est pas ponctuelle : c'est un paquet gaussien de
largeur `σ`. Le champ qu'elle ressent est donc le gradient du potentiel
**convolué** par cette gaussienne, ce qui adoucit les collisions proches que
la discrétisation rendrait autrement singulières.

Sur une grille à pas constant, cette convolution ne dépend que de la position
de la particule **dans sa maille** : on tabule une fois pour toutes les
recouvrements `∫φₐ(x')·G(x−x')dx'` et leurs dérivées, puis on interpole.
"""

"""
    GaussianSmoothing(axis; nbdt = 1000, quadrature = 1000)

Tables de convolution d'un axe à pas constant (`maketaint` du Fortran).

`overlap[a, i]` est le recouvrement de la a-ième des 10 fonctions de base
voisines avec une gaussienne centrée en `r`, et `gradient[a, i]` sa dérivée
en `r` — c'est elle qui donne le champ. L'indice `i` discrétise la position
de `r` dans une maille, en `nbdt + 1` valeurs.

La largeur vaut `σ = h/3`, liée au pas de grille : c'est le choix du code
d'origine, qui fixe le lissage à l'échelle de la résolution.

⚠️ **La quadrature est celle du Fortran** — trapèzes sur `quadrature`
intervalles — et non une règle d'ordre élevé. Ce n'est pas un oubli : c'est
elle qui définit les valeurs de l'oracle, et en changer ramollirait de `1e-14`
à `1e-9` toutes les comparaisons en aval, y compris celles des forces.

⚠️ **Le noyau tabulé n'est pas exactement normalisé** (mesuré) :

  * `Σ recouvrements` vaut 1 à `3.4e-6` près ;
  * la dérivée d'une fonction constante rend `1.3e-5` au lieu de 0.

Le champ lissé porte donc une erreur relative de l'ordre de `1e-5` — dont une
composante transverse, un potentiel ne dépendant que de `x` produisant un
champ en `y` non nul. C'est une limite de la **méthode d'origine**, pas du
portage : la fenêtre de 10 fonctions ne capte pas toute la gaussienne, et les
deux fonctions extrêmes sont intégrées sur un support tronqué. Le champ non
lissé, lui, est exact à l'arrondi près.
"""
struct GaussianSmoothing{T<:AbstractFloat}
    σ::T
    spacing::T
    nbdt::Int
    overlap::Matrix{T}
    gradient::Matrix{T}
end

"""
Bornes d'intégration des 10 fonctions de base voisines, en indices de nœuds
relatifs à la fenêtre de tabulation. Les deux extrêmes sont tronquées : leur
recouvrement avec la gaussienne y est de l'ordre de `e⁻¹⁸`.
"""
const SMOOTHING_SUPPORTS = ((1, 2), (1, 2), (1, 3), (1, 3), (2, 4),
                            (2, 4), (3, 5), (3, 5), (4, 5), (4, 5))

function GaussianSmoothing(ax::SplineAxis{T}; nbdt::Int = 1000,
                           quadrature::Int = 1000) where {T}
    g = ax.knots
    h = g[2] - g[1]
    σ = h / 3
    norm1d = inv(sqrt(2 * T(π)) * σ)   # normalisation 1D d'une gaussienne 3D

    # `r` balaie une maille centrée sur le nœud g[3] ; l'invariance par
    # translation de la grille régulière fait le reste.
    rmin, rmax = (g[3] + g[2]) / 2, (g[4] + g[3]) / 2

    overlap = Matrix{T}(undef, 10, nbdt + 1)
    gradient = Matrix{T}(undef, 10, nbdt + 1)

    for i in 0:nbdt
        r = rmin + i * (rmax - rmin) / nbdt
        for a in 1:10
            lo, hi = SMOOTHING_SUPPORTS[a]
            b = BasisIndex(a)
            gauss(x) = norm1d * exp(-(r - x)^2 / 2σ^2)
            overlap[a, i+1] = trapezoid(g[lo], g[hi], quadrature) do x
                value(ax, b, x) * gauss(x)
            end
            gradient[a, i+1] = trapezoid(g[lo], g[hi], quadrature) do x
                -(r - x) / σ^2 * value(ax, b, x) * gauss(x)
            end
        end
    end
    GaussianSmoothing{T}(σ, h, nbdt, overlap, gradient)
end

"""
    trapezoid(f, a, b, n)

Règle des trapèzes sur `n` intervalles. Les abscisses sont cumulées
(`x += pas`) comme dans le Fortran : reconstruire `a + i·pas` donnerait des
points très légèrement différents, et l'écart se verrait à la comparaison.
"""
function trapezoid(f, a::T, b::T, n::Integer) where {T}
    pas = (b - a) / n
    x = a
    fx = f(x)
    total = zero(T)
    for _ in 1:n
        xnext = x + pas
        fnext = f(xnext)
        total += (fx + fnext) * pas / 2
        x, fx = xnext, fnext
    end
    total
end

"""
    nearest_knot(knots, x) -> Int

Indice du nœud le plus proche de `x` (le `xig` du Fortran).
"""
@inline function nearest_knot(knots, x)
    i = clamp(searchsortedlast(knots, x), 1, length(knots) - 1)
    (knots[i+1] - x) < (x - knots[i]) ? i + 1 : i
end

"""
    cell_index(knots, x) -> Int ou `nothing`

Indice de la maille contenant `x` (le `xi` du Fortran), ou `nothing` hors
domaine.
"""
@inline function cell_index(knots, x)
    (x < knots[1] || x > knots[end]) && return nothing
    max(1, searchsortedfirst(knots, x) - 1)
end

"""Colonne de table correspondant à la position de `x` dans sa maille."""
@inline function table_column(sm::GaussianSmoothing, x, knot)
    # `floor(v + 1/2)` et non `round` : Julia arrondit au pair le plus proche,
    # le Fortran tranche vers le haut.
    floor(Int, (x - knot + sm.spacing / 2) / sm.spacing * sm.nbdt + 0.5) + 1
end

"""
    smoothed_field(axes, csol, sm, p) -> NTuple{3,T}

Champ électrique lissé au point `p` (le `champsg` du Fortran), à partir des
coefficients spline `csol` du potentiel.

`E = −∇(Φ ∗ G)` : chaque direction combine les recouvrements tabulés, la
direction dérivée prenant `gradient` là où les deux autres prennent
`overlap`.
"""
function smoothed_field(axes::NTuple{3,SplineAxis{T}}, csol::Array{T,3},
                        sm::GaussianSmoothing{T}, p) where {T}
    base = ntuple(3) do d
        2 * nearest_knot(axes[d].knots, p[d]) - 5
    end
    col = ntuple(d -> table_column(sm, p[d], axes[d].knots[nearest_knot(axes[d].knots, p[d])]), 3)

    ox = @view sm.overlap[:, col[1]]
    gx = @view sm.gradient[:, col[1]]
    oy = @view sm.overlap[:, col[2]]
    gy = @view sm.gradient[:, col[2]]
    oz = @view sm.overlap[:, col[3]]
    gz = @view sm.gradient[:, col[3]]

    fx = fy = fz = zero(T)
    @inbounds for kk in 1:10
        k = base[3] + kk - 1
        for jj in 1:10
            j = base[2] + jj - 1
            cxx = oy[jj] * oz[kk]
            cyy = gy[jj] * oz[kk]
            czz = oy[jj] * gz[kk]
            dxp = zero(T)   # avec la dérivée en x
            val = zero(T)   # sans
            for ii in 1:10
                c = csol[base[1]+ii-1, j, k]
                dxp += c * gx[ii]
                val += c * ox[ii]
            end
            fx += cxx * dxp
            fy += cyy * val
            fz += czz * val
        end
    end
    (-fx, -fy, -fz)
end

"""
    spline_field(axes, csol, p) -> NTuple{3,T} ou `nothing`

Champ électrique **non lissé**, gradient direct de l'interpolant spline (le
`champ` du Fortran). Renvoie `nothing` hors du domaine.

Employé sur la grille grossière, où les particules sont loin de la zone dense
et où le lissage n'a plus d'objet.
"""
function spline_field(axes::NTuple{3,SplineAxis{T}}, csol::Array{T,3}, p) where {T}
    cells = ntuple(d -> cell_index(axes[d].knots, p[d]), 3)
    any(isnothing, cells) && return nothing

    # Les 4 fonctions de base non nulles dans la maille, valeur et dérivée.
    vals = ntuple(3) do d
        c = cells[d]
        ntuple(4) do a
            evaluate(axes[d], BasisIndex(2 * (c - 1) + a), p[d], Val(1))
        end
    end

    ex = ey = ez = zero(T)
    @inbounds for ak in 1:4
        k = 2 * (cells[3] - 1) + ak
        vk, dk = vals[3][ak]
        for aj in 1:4
            j = 2 * (cells[2] - 1) + aj
            vj, dj = vals[2][aj]
            for ai in 1:4
                i = 2 * (cells[1] - 1) + ai
                vi, di = vals[1][ai]
                c = csol[i, j, k]
                ex -= c * di * vj * vk
                ey -= c * vi * dj * vk
                ez -= c * vi * vj * dk
            end
        end
    end
    (ex, ey, ez)
end

"""
    forces!(cloud, fine, csol_fine, coarse, csol_coarse, sm; escaped) -> Int

Remplit les forces du nuage (le `force2g` du Fortran), et renvoie le nombre
de particules traitées hors de la grille fine.

Trois régimes, du plus fin au plus grossier :

  1. bien à l'intérieur de la grille fine → champ **lissé** ;
  2. au-delà → gradient spline de la grille **grossière** ;
  3. hors des deux → monopôle coulombien de la charge enfermée, `escaped`
     particules manquant à l'appel.

La force vaut `w·E`, `w` étant le nombre d'électrons que porte la
pseudo-particule.
"""
function forces!(cloud::ParticleCloud{T},
                 fine::NTuple{3,SplineAxis{T}}, csol_fine::Array{T,3},
                 coarse::NTuple{3,SplineAxis{T}}, csol_coarse::Array{T,3},
                 sm::GaussianSmoothing{T}; escaped::Integer = 0) where {T}
    w = cloud.weight
    w2 = w * w
    # Marge de deux mailles : le champ lissé lit 10 fonctions de base autour
    # du point, il lui faut deux nœuds de chaque côté.
    lo = ntuple(d -> fine[d].knots[3], 3)
    hi = ntuple(d -> fine[d].knots[end-2], 3)

    outside = 0
    @inbounds for i in eachindex(cloud.positions)
        p = cloud.positions[i]
        if all(d -> lo[d] < p[d] < hi[d], 1:3)
            cloud.forces[i] = w .* smoothed_field(fine, csol_fine, sm, p)
        else
            outside += 1
            E = spline_field(coarse, csol_coarse, p)
            cloud.forces[i] = if E === nothing
                # Hors des deux grilles : tout ce qui reste est la charge
                # enfermée, vue de loin.
                r3 = (p[1]^2 + p[2]^2 + p[3]^2)^T(1.5)
                (-w2 * escaped / r3) .* p
            else
                w .* E
            end
        end
    end
    outside
end
