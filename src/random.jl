"""
The original code's pseudo-random generator, reproduced exactly.

The port cannot settle for an "equivalent" generator: the initial sampling of
the pseudo-particles is the simulation's only non-deterministic input.
Reproducing the exact sequence makes it possible to compare the initialisation
**particle by particle** against the oracle, instead of merely observing
compatible statistics — which cannot tell a bug from sampling noise.
"""

"""
    Ran2(seed)

The `ran2` generator from *Numerical Recipes* (combined L'Ecuyer, Bays-Durham
shuffle), **as it appears in the thesis code**.

⚠️ **It departs from the published version, and not harmlessly.** The constant
`IQ1` reads `3668` instead of `53668` — one digit lost. Schrage's method, which
avoids overflow, requires `IR1 < IQ1`; with `12211 > 3668` it no longer holds,
and `k*IR1` overflows the 32-bit integer (up to `7.1e9` against a limit of
`2.1e9`).

The generator therefore does produce a sequence, but **not** L'Ecuyer's: it is a
variant whose period and correlations were never checked. The typo is present
identically in all five versions of the thesis code.

We reproduce it — the only way to recover the oracle's sampling — by letting the
arithmetic overflow as it does in Fortran. `consistent = true` restores
`IQ1 = 53668`, for comparison.
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

"`AM` and `RNMX` are computed in single precision, like the Fortran `REAL`."
const RAN2_AM = Float32(1) / Float32(RAN2_IM1)
const RAN2_RNMX = Float32(1) - Float32(1.2e-7)

function Ran2(seed::Integer = -1; consistent::Bool = false)
    iq1 = consistent ? Int32(53668) : Int32(3668)
    idum = Int32(max(-seed, 1))
    idum2 = idum
    iv = zeros(Int32, RAN2_NTAB)
    # Warm-up: discard the first 8 values, then fill the table.
    for j in (RAN2_NTAB+8):-1:1
        idum = _ran2_step(idum, iq1, RAN2_IR1, RAN2_IA1, RAN2_IM1)
        j <= RAN2_NTAB && (iv[j] = idum)
    end
    Ran2(idum, idum2, iv[1], iv, iq1)
end

"""One Schrage step. ⚠️ Overflows on purpose: see [`Ran2`](@ref)."""
@inline function _ran2_step(x::Int32, q::Int32, r::Int32, a::Int32, m::Int32)
    k = x ÷ q
    x = a * (x - k * q) - k * r
    x < 0 ? x + m : x
end

"""
    next!(rng) -> Float32

Next value in `[0, 1)`. The result is in **single precision**: the Fortran
declared `REAL ran2`, and rounding otherwise would change the sequence.
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
