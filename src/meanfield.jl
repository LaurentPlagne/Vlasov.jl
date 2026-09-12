"""
Les deux termes de champ moyen qui, ajoutés au potentiel de Hartree, font
l'agrégat : l'échange-corrélation des électrons entre eux, et l'attraction du
fond ionique.

Sans eux il n'y a pas d'agrégat, seulement un gaz d'électrons qui se repousse
et se disperse. Le Fortran les ajoutait dans `pspech`, appelée à chaque pas
avant le calcul des forces — le nom suggère un diagnostic, c'en est loin.
"""

"Rayon de Wigner-Seitz du sodium, en unités atomiques (`rs` du Fortran)."
const WIGNER_SEITZ_NA = 4.0

"""
    Jellium(nions; rs = WIGNER_SEITZ_NA)

Fond ionique modélisé par une sphère uniformément chargée de `nions` charges
positives, de rayon `r₀ = rs·N^⅓`.

C'est ce fond qui retient les électrons ; sa densité est celle d'un métal de
rayon de Wigner-Seitz `rs`.

`rs` est un mot-clé, et pas un second argument positionnel, pour ne pas entrer
en concurrence avec le constructeur `Jellium(nions, radius)` engendré par la
déclaration du type — qui, portant la contrainte `T<:AbstractFloat`, est plus
spécifique qu'un `where {T}` et l'emporterait silencieusement.
"""
struct Jellium{T<:AbstractFloat}
    nions::T
    radius::T
end

function Jellium(nions::Real; rs::Real = WIGNER_SEITZ_NA)
    T = float(promote_type(typeof(nions), typeof(rs)))
    Jellium{T}(T(nions), T(rs) * cbrt(T(nions)))
end

"""
    potential(jel, r) -> T

Potentiel du fond de jellium à la distance `r` du centre.

À l'extérieur, c'est celui d'une charge ponctuelle `−N/r` ; à l'intérieur, la
parabole classique d'une boule uniformément chargée, qui reste finie au
centre.
"""
@inline function potential(jel::Jellium{T}, r) where {T}
    r0 = jel.radius
    r < r0 ? -jel.nions * (3 - (r / r0)^2) / 2r0 : -jel.nions / r
end

# Constantes du potentiel d'échange-corrélation LDA, en unités atomiques.
# Échange de Dirac : Vx = −(3/π)^⅓ ρ^⅓.
# Corrélation de Gunnarsson-Lundqvist : Vc = −0.0333·ln(1 + 11.4/rs),
# avec 1/rs = (4π/3)^⅓ ρ^⅓.
const XC_EXCHANGE = -cbrt(3 / π)
const XC_CORRELATION = -0.0333
const XC_CORRELATION_SCALE = 11.4 * cbrt(4π / 3)

"""
    xc_potential(ρ) -> T

Potentiel d'échange-corrélation LDA pour une densité électronique `ρ`.

L'échange est la forme de Dirac, la corrélation celle de Gunnarsson-Lundqvist.
Tous deux ne dépendent de la densité que par `ρ^⅓`, calculé une fois.
"""
@inline function xc_potential(ρ::T) where {T}
    c = cbrt(ρ)
    T(XC_EXCHANGE) * c + T(XC_CORRELATION) * log1p(T(XC_CORRELATION_SCALE) * c)
end

"""
    effective_potential!(csol, ρ, mesh, jellium) -> csol

Ajoute au potentiel de Hartree — déjà présent dans `csol` sous forme de
coefficients spline — l'échange-corrélation et le fond de jellium (le `pspech`
du Fortran).

Les deux termes sont évalués aux points de collocation, puis convertis en
coefficients avant d'être ajoutés : c'est dans cet espace-là que vit `csol`,
et les mélanger serait le bug silencieux que la distinction de types existe
pour empêcher.
"""
function effective_potential!(csol::Array{T,3}, ρ::Array{T,3},
                              mesh::SplineMesh{3,T}, jel::Jellium{T}) where {T}
    gx, gy, gz = map(ax -> ax.colloc, mesh.axes)
    extra = similar(ρ)
    @inbounds for k in eachindex(gz), j in eachindex(gy), i in eachindex(gx)
        r = sqrt(gx[i]^2 + gy[j]^2 + gz[k]^2)
        extra[i, j, k] = xc_potential(ρ[i, j, k]) + potential(jel, r)
    end
    csol .+= spline_coefficients(extra, mesh)
end
