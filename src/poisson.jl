"""
Assemblage du second membre de l'équation de Poisson (`makerh2` du Fortran).

Le domaine de calcul est fini, mais le potentiel d'un agrégat ne l'est pas :
on impose sur les faces la valeur qu'aurait le potentiel à grande distance,
donnée par le **développement multipolaire** de la densité — monopôle et
quadrupôle, exprimés dans le repère du barycentre où le dipôle s'annule.

Ces valeurs de bord étant non nulles, elles sont **relevées** : on les fait
passer au second membre via les colonnes extrêmes de l'opérateur complet.
"""

"""
    Multipole(charge, center, quadrupole)

Développement multipolaire d'une distribution de charge.

`center` est le barycentre — s'y placer annule le terme dipolaire, ce qui
laisse monopôle et quadrupôle. `quadrupole` ne garde que les 6 composantes
indépendantes du tenseur symétrique, dans l'ordre `xx, yy, zz, xy, xz, yz`.
"""
struct Multipole{T}
    charge::T
    center::NTuple{3,T}
    quadrupole::NTuple{6,T}
end

"""
    contract(c, u, v, w) -> T

Contracte les coefficients spline avec un produit extérieur de moments 1D :
`Σ c[i,j,k]·u[i]·v[j]·w[k]`. Tous les moments multipolaires sont de cette
forme, à un choix de moments près.
"""
function contract(c::Array{T,3}, u, v, w) where {T}
    # ⚠️ Délibérément SÉQUENTIELLE. Une contraction coûte ~0,3 ms : la
    # découper sur huit fils la fait passer à 0,8 ms, l'orchestration
    # dominant le calcul. Le parallélisme est pris un cran au-dessus, dans
    # `multipole`, où les dix contractions sont indépendantes entre elles.
    s = zero(T)
    @inbounds for k in eachindex(w), j in eachindex(v), i in eachindex(u)
        s += c[i, j, k] * u[i] * v[j] * w[k]
    end
    s
end

"""
    all_moments(c, p0, p1, p2) -> NTuple{10,T}

Les dix contractions du développement multipolaire, en **une seule passe** sur
les coefficients.

Les calculer séparément relit `c` dix fois. Or l'opération est limitée par la
bande passante mémoire et non par le calcul — mesuré : les dix contractions
lancées en parallèle sur huit fils sont 2,6 fois plus LENTES que la même chose
en séquentiel, parce qu'elles se disputent la mémoire au lieu de se partager
du travail.

Une passe unique factorise tout : pour chaque couple `(j,k)`, trois sommes
partielles sur `i` suffisent à alimenter les dix moments.
"""
function all_moments(c::Array{T,3}, p0, p1, p2) where {T}
    p0x, p0y, p0z = p0
    p1x, p1y, p1z = p1
    p2x, p2y, p2z = p2
    q = d100 = d010 = d001 = m200 = m020 = m002 = mxy = mxz = myz = zero(T)

    @inbounds for k in eachindex(p0z), j in eachindex(p0y)
        a  = p0y[j] * p0z[k]
        b  = p1y[j] * p0z[k]
        cc = p0y[j] * p1z[k]
        d  = p2y[j] * p0z[k]
        e  = p0y[j] * p2z[k]
        f  = p1y[j] * p1z[k]

        # Les trois seules sommes sur `i` dont les dix moments ont besoin.
        s0 = s1 = s2 = zero(T)
        for i in eachindex(p0x)
            v = c[i, j, k]
            s0 += v * p0x[i]
            s1 += v * p1x[i]
            s2 += v * p2x[i]
        end

        q    += s0 * a
        d100 += s1 * a
        d010 += s0 * b
        d001 += s0 * cc
        m200 += s2 * a
        m020 += s0 * d
        m002 += s0 * e
        mxy  += s1 * b
        mxz  += s1 * cc
        myz  += s0 * f
    end
    (q, d100, d010, d001, m200, m020, m002, mxy, mxz, myz)
end

"""
    multipole(ρ, mesh) -> Multipole

Moments de la densité `ρ` donnée aux points de collocation.

L'intégration devrait passer par les coefficients spline — `∫f = Σ cᵦ ∫φᵦ` —
mais il n'est pas nécessaire de les former. Avec `c = S⁻¹ρ` :

    Σ c[i,j,k]·u[i]v[j]w[k] = Σ ρ[a,b,c]·ũ[a]ṽ[b]w̃[c]     où  ũ = S⁻ᵀu

Transformer les trois **vecteurs** de moments coûte trois produits
matrice-vecteur ; transformer le **tableau 3D** en coûtait trois produits
matrice-matrice, quatre ordres de grandeur de plus. Et comme les moments ne
dépendent que du maillage, `SplineMesh` les garde déjà transformés.
"""
function multipole(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T}
    # Les moments duaux portent déjà `S⁻ᵀ` : on contracte directement la
    # densité, sans former ses coefficients spline.
    p0 = map(m -> m[1], mesh.dual_moments)
    p1 = map(m -> m[2], mesh.dual_moments)
    p2 = map(m -> m[3], mesh.dual_moments)

    q, d100, d010, d001, m200, m020, m002, mxy, mxz, myz = all_moments(ρ, p0, p1, p2)

    # Dipôle, ramené en barycentre. Une densité de charge nulle n'en a pas.
    dip = (d100, d010, d001)
    center = iszero(q) ? ntuple(_ -> zero(T), 3) : dip ./ q

    # Quadrupôle sous forme sans trace : Qₗₗ = 2∫xₗ² − Σ_{m≠l} ∫xₘ², et
    # Qₗₘ = 3∫xₗxₘ hors diagonale.
    quad = (2m200 - m020 - m002,
            2m020 - m200 - m002,
            2m002 - m200 - m020,
            3mxy, 3mxz, 3myz)

    # Translation du tenseur au barycentre (théorème des axes parallèles).
    bx, by, bz = center
    b2 = bx^2 + by^2 + bz^2
    shift = (q * (3bx^2 - b2), q * (3by^2 - b2), q * (3bz^2 - b2),
             3q * bx * by, 3q * bx * bz, 3q * by * bz)

    Multipole{T}(q, center, quad .- shift)
end

"""
    potential(mp, x, y, z) -> T

Potentiel du développement multipolaire au point donné : `q/r` plus le terme
quadrupolaire en `1/r⁵`. Les composantes hors diagonale comptent double, le
tenseur étant symétrique.
"""
function potential(mp::Multipole{T}, x, y, z) where {T}
    px, py, pz = (x, y, z) .- mp.center
    r2 = px^2 + py^2 + pz^2
    invr = inv(sqrt(r2))
    qxx, qyy, qzz, qxy, qxz, qyz = mp.quadrupole
    quad = qxx * px^2 + qyy * py^2 + qzz * pz^2 +
           2 * (qxy * px * py + qxz * px * pz + qyz * py * pz)
    mp.charge * invr + quad * invr^5 / 2
end

"""
    foreach_face(f, nx, ny, nz)

Applique `f(i, j, k)` aux points de la **surface** d'une grille, et à eux
seuls.

Parcourir tout le volume en écartant l'intérieur par un test visite 195 000
points pour n'en traiter 19 500 : neuf dixièmes du temps passés à décider de
ne rien faire.

Les deux faces pleines sont traitées à part des parois latérales. Ce n'est pas
de la coquetterie : elles pèsent un tiers des points à elles deux, et les
répartir avec le reste déséquilibrerait les fils.
"""
function foreach_face(f, nx::Integer, ny::Integer, nz::Integer)
    tforeach(ny) do slice
        for j in slice, i in 1:nx
            f(i, j, 1)
            f(i, j, nz)
        end
    end
    tforeach(nz - 2) do slice
        for kk in slice
            k = kk + 1
            for i in 1:nx
                f(i, 1, k)
                f(i, ny, k)
            end
            for j in 2:(ny-1)
                f(1, j, k)
                f(nx, j, k)
            end
        end
    end
end

"""
    boundary_potential!(φ, mesh, mp) -> φ

Remplit `φ` (taille complète de la grille) avec le potentiel multipolaire sur
les **faces** du domaine. L'intérieur est laissé à zéro : il n'est jamais lu,
seules les faces servent au relèvement.
"""
function boundary_potential!(φ::Array{T,3}, mesh::SplineMesh{3,T},
                             mp::Multipole{T}) where {T}
    gx, gy, gz = map(ax -> ax.colloc, mesh.axes)
    nx, ny, nz = length(gx), length(gy), length(gz)
    fill!(φ, zero(T))
    foreach_face(nx, ny, nz) do i, j, k
        @inbounds φ[i, j, k] = potential(mp, gx[i], gy[j], gz[k])
    end
    φ
end

"""
    poisson_rhs!(rhs, ρ, mesh) -> rhs

Assemble le second membre de `∇²Φ = −4πρ` sur les points de collocation
**intérieurs**, conditions de bord multipolaires comprises.

`ρ` couvre toute la grille, `rhs` seulement l'intérieur — c'est ce qu'attend
[`solve!`](@ref).
"""
function poisson_rhs!(rhs::Array{T,3}, ρ::Array{T,3}, mesh::SplineMesh{3,T},
                      φ::Array{T,3}) where {T}
    size(rhs) == size(mesh) ||
        throw(DimensionMismatch("rhs doit avoir la taille du problème intérieur"))

    @views rhs .= -4 * T(π) .* ρ[2:end-1, 2:end-1, 2:end-1]

    # Relèvement : chaque face contribue par la colonne correspondante de
    # l'opérateur complet, vue depuis les lignes intérieures.
    Dx, Dy, Dz = mesh.laplacians
    nsx, nsy, nsz = size(mesh)
    @views begin
        rhs .-= reshape(Dx[2:end-1, 1], :, 1, 1) .*
                reshape(φ[1, 2:end-1, 2:end-1], 1, nsy, nsz)
        rhs .-= reshape(Dx[2:end-1, end], :, 1, 1) .*
                reshape(φ[end, 2:end-1, 2:end-1], 1, nsy, nsz)

        rhs .-= reshape(Dy[2:end-1, 1], 1, :, 1) .*
                reshape(φ[2:end-1, 1, 2:end-1], nsx, 1, nsz)
        rhs .-= reshape(Dy[2:end-1, end], 1, :, 1) .*
                reshape(φ[2:end-1, end, 2:end-1], nsx, 1, nsz)

        rhs .-= reshape(Dz[2:end-1, 1], 1, 1, :) .*
                reshape(φ[2:end-1, 2:end-1, 1], nsx, nsy, 1)
        rhs .-= reshape(Dz[2:end-1, end], 1, 1, :) .*
                reshape(φ[2:end-1, 2:end-1, end], nsx, nsy, 1)
    end
    rhs
end

"""
Variante qui calcule elle-même le potentiel de bord. À éviter dans une boucle
en temps : [`poisson!`](@ref) le réutilise au lieu de refaire les contractions
multipolaires.
"""
poisson_rhs!(rhs::Array{T,3}, ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T} =
    poisson_rhs!(rhs, ρ, mesh,
                 boundary_potential!(mesh.scratch[1], mesh, multipole(ρ, mesh)))

"""Version allouante de [`poisson_rhs!`](@ref)."""
poisson_rhs(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T} =
    poisson_rhs!(Array{T,3}(undef, size(mesh)), ρ, mesh)

"""
    poisson!(φ, ρ, mesh) -> φ

Résout `∇²Φ = −4πρ` sur toute la grille : les faces reçoivent le potentiel
multipolaire, l'intérieur la solution du système tensoriel.

`φ` a la taille complète de la grille de collocation, comme `ρ`. C'est la
chaîne complète densité → potentiel, et le `makerh2` + `solve` du Fortran.
"""
function poisson!(φ::Array{T,3}, ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T}
    boundary_potential!(φ, mesh, multipole(ρ, mesh))
    solve_interior!(φ, ρ, mesh)
end

"""
    solve_interior!(φ, ρ, mesh) -> φ

Résout l'intérieur en prenant pour conditions de Dirichlet les valeurs **déjà
présentes** sur les faces de `φ`.

C'est la moitié commune à [`poisson!`](@ref), qui pose ces valeurs par
développement multipolaire, et au raccord entre grilles, qui les lit dans la
solution du niveau plus grossier.
"""
function solve_interior!(φ::Array{T,3}, ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T}
    rhs = poisson_rhs!(mesh.scratch_inner, ρ, mesh, φ)
    solve!(rhs, rhs, mesh.solver)
    @views φ[2:end-1, 2:end-1, 2:end-1] .= rhs
    φ
end

"""
    boundary_from_coarse!(φ, mesh, coarse, csol_coarse) -> φ

Pose sur les faces de `φ` les valeurs lues dans la solution d'une grille plus
grossière (le `makerhsf` du Fortran).

C'est tout le raccord entre niveaux : la grille fine ne voit du monde
extérieur que ce que la grossière lui dit à sa frontière.
"""
function boundary_from_coarse!(φ::Array{T,3}, mesh::SplineMesh{3,T},
                               coarse::SplineMesh{3,T}, csol_coarse::Array{T,3}) where {T}
    gx, gy, gz = map(ax -> ax.colloc, mesh.axes)
    nx, ny, nz = length(gx), length(gy), length(gz)
    fill!(φ, zero(T))
    foreach_face(nx, ny, nz) do i, j, k
        p = spline_potential(coarse.axes, csol_coarse, (gx[i], gy[j], gz[k]))
        p === nothing && throw(ArgumentError(
            "le point de bord ($(gx[i]), $(gy[j]), $(gz[k])) sort de la grille " *
            "grossière : les niveaux ne sont pas emboîtés"))
        @inbounds φ[i, j, k] = p
    end
    φ
end

"""Version allouante de [`poisson!`](@ref)."""
poisson(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T} = poisson!(similar(ρ), ρ, mesh)

"""
    poisson!(φs, ρs, nested) -> φs

Résout Poisson sur une hiérarchie de grilles emboîtées, du plus grossier au
plus fin.

Le niveau le plus grossier prend ses conditions du développement multipolaire
de sa propre densité ; chaque niveau plus fin lit les siennes dans la solution
du niveau au-dessus. `φs` et `ρs` sont ordonnés comme les niveaux, du plus fin
au plus grossier.
"""
function poisson!(φs::NTuple{L,Array{T,3}}, ρs::NTuple{L,Array{T,3}},
                  nested::NestedMeshes{L,3,T}) where {L,T}
    poisson!(φs[L], ρs[L], nested[L])
    for l in (L-1):-1:1
        coefs = spline_coefficients!(nested[l+1].scratch[2], φs[l+1], nested[l+1])
        boundary_from_coarse!(φs[l], nested[l], nested[l+1], coefs)
        solve_interior!(φs[l], ρs[l], nested[l])
    end
    φs
end
