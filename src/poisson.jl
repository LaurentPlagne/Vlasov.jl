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
contract(c::Array{T,3}, u, v, w) where {T} =
    sum(c[i, j, k] * u[i] * v[j] * w[k]
        for i in eachindex(u), j in eachindex(v), k in eachindex(w))

"""
    multipole(ρ, mesh) -> Multipole

Moments de la densité `ρ` donnée aux points de collocation.

L'intégration passe par les **coefficients spline** : `∫f = Σ cᵦ ∫φᵦ`, ce que
les valeurs aux points de collocation ne permettent pas directement.
"""
function multipole(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T}
    c = spline_coefficients(ρ, mesh)
    p0 = map(ax -> moments(ax, Val(0)), mesh.axes)
    p1 = map(ax -> moments(ax, Val(1)), mesh.axes)
    p2 = map(ax -> moments(ax, Val(2)), mesh.axes)

    q = contract(c, p0[1], p0[2], p0[3])

    # Dipôle, ramené en barycentre. Une densité de charge nulle n'en a pas.
    dip = (contract(c, p1[1], p0[2], p0[3]),
           contract(c, p0[1], p1[2], p0[3]),
           contract(c, p0[1], p0[2], p1[3]))
    center = iszero(q) ? ntuple(_ -> zero(T), 3) : dip ./ q

    # Quadrupôle sous forme sans trace : Qₗₗ = 2∫xₗ² − Σ_{m≠l} ∫xₘ², et
    # Qₗₘ = 3∫xₗxₘ hors diagonale.
    m200 = contract(c, p2[1], p0[2], p0[3])
    m020 = contract(c, p0[1], p2[2], p0[3])
    m002 = contract(c, p0[1], p0[2], p2[3])
    quad = (2m200 - m020 - m002,
            2m020 - m200 - m002,
            2m002 - m200 - m020,
            3contract(c, p1[1], p1[2], p0[3]),
            3contract(c, p1[1], p0[2], p1[3]),
            3contract(c, p0[1], p1[2], p1[3]))

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
    for k in 1:nz, j in 1:ny, i in 1:nx
        surface = i == 1 || i == nx || j == 1 || j == ny || k == 1 || k == nz
        surface || continue
        φ[i, j, k] = potential(mp, gx[i], gy[j], gz[k])
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
                 boundary_potential!(similar(ρ), mesh, multipole(ρ, mesh)))

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
    rhs = poisson_rhs!(Array{T,3}(undef, size(mesh)), ρ, mesh, φ)
    solve!(rhs, rhs, mesh.solver)
    @views φ[2:end-1, 2:end-1, 2:end-1] .= rhs
    φ
end

"""Version allouante de [`poisson!`](@ref)."""
poisson(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T} = poisson!(similar(ρ), ρ, mesh)
