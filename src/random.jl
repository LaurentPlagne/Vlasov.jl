"""
Le générateur pseudo-aléatoire du code d'origine, reproduit à l'identique.

Le portage ne peut pas se contenter d'un générateur « équivalent » : le tirage
initial des pseudo-particules est la seule entrée non déterministe de la
simulation. Reproduire la suite exacte permet de comparer l'initialisation
**particule par particule** à l'oracle, au lieu de constater des statistiques
compatibles — ce qui ne distingue pas un bug d'un bruit d'échantillonnage.
"""

"""
    Ran2(seed)

Générateur `ran2` de *Numerical Recipes* (L'Ecuyer combiné, brassage de
Bays-Durham), **tel qu'il figure dans le code de la thèse**.

⚠️ **Il s'écarte de la version publiée, et ce n'est pas anodin.** La constante
`IQ1` y vaut `3668` au lieu de `53668` — un chiffre perdu. La méthode de
Schrage, qui évite le débordement, exige `IR1 < IQ1` ; avec `12211 > 3668`
elle ne tient plus, et `k*IR1` déborde l'entier 32 bits (jusqu'à `7.1e9` pour
une limite à `2.1e9`).

Le générateur produit donc bien une suite, mais ce n'est **pas** celle de
L'Ecuyer : c'est une variante dont la période et les corrélations n'ont
jamais été vérifiées. La coquille est présente à l'identique dans les cinq
versions du code de la thèse.

On la reproduit — c'est la seule façon de retrouver le tirage de l'oracle —
en laissant l'arithmétique déborder comme en Fortran. `consistent = true`
rétablit `IQ1 = 53668`, pour comparer.
"""
mutable struct Ran2
    idum::Int32
    idum2::Int32
    iy::Int32
    const iv::Vector{Int32}
    const iq1::Int32
end

const RAN2_IM1 = Int32(2147483563)
const RAN2_IM2 = Int32(2147483399)
const RAN2_IMM1 = RAN2_IM1 - Int32(1)
const RAN2_IA1 = Int32(40014)
const RAN2_IA2 = Int32(40692)
const RAN2_IQ2 = Int32(52774)
const RAN2_IR1 = Int32(12211)
const RAN2_IR2 = Int32(3791)
const RAN2_NTAB = 32
const RAN2_NDIV = Int32(1 + (RAN2_IMM1 - Int32(1)) ÷ Int32(RAN2_NTAB))

"`AM` et `RNMX` sont calculés en simple précision, comme le `REAL` du Fortran."
const RAN2_AM = Float32(1) / Float32(RAN2_IM1)
const RAN2_RNMX = Float32(1) - Float32(1.2e-7)

function Ran2(seed::Integer = -1; consistent::Bool = false)
    iq1 = consistent ? Int32(53668) : Int32(3668)
    idum = Int32(max(-seed, 1))
    idum2 = idum
    iv = zeros(Int32, RAN2_NTAB)
    # Rodage : on jette les 8 premières valeurs, puis on remplit la table.
    for j in (RAN2_NTAB+8):-1:1
        idum = _ran2_step(idum, iq1, RAN2_IR1, RAN2_IA1, RAN2_IM1)
        j <= RAN2_NTAB && (iv[j] = idum)
    end
    Ran2(idum, idum2, iv[1], iv, iq1)
end

"""Un pas de Schrage. ⚠️ Déborde volontairement : voir [`Ran2`](@ref)."""
@inline function _ran2_step(x::Int32, q::Int32, r::Int32, a::Int32, m::Int32)
    k = x ÷ q
    x = a * (x - k * q) - k * r
    x < 0 ? x + m : x
end

"""
    next!(rng) -> Float32

Valeur suivante dans `[0, 1)`. Le résultat est en **simple précision** : le
Fortran déclarait `REAL ran2`, et arrondir autrement changerait la suite.
"""
function next!(rng::Ran2)
    rng.idum = _ran2_step(rng.idum, rng.iq1, RAN2_IR1, RAN2_IA1, RAN2_IM1)
    rng.idum2 = _ran2_step(rng.idum2, RAN2_IQ2, RAN2_IR2, RAN2_IA2, RAN2_IM2)

    j = 1 + rng.iy ÷ RAN2_NDIV
    rng.iy = rng.iv[j] - rng.idum2
    rng.iv[j] = rng.idum
    rng.iy < 1 && (rng.iy += RAN2_IMM1)

    min(RAN2_AM * rng.iy, RAN2_RNMX)
end
