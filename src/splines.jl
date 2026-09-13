"""
Local cubic Hermite spline basis, in 1D.

Each knot `k` of the grid carries **two** basis functions:

  * `Value`: equals 1 at `g[k]`, with vanishing derivative;
  * `Slope`: equals 0 at `g[k]`, with derivative 1.

The support of a basis function is `[g[k-1], g[k+1]]` (two intervals). The
original Fortran encoded this pair in a single integer `ido = 2k + isig`, whence
the `i/2` and `mod(i,2)` scattered everywhere; here the pair is reified in
[`BasisIndex`](@ref), and the linear index appears only when filling a matrix.
"""

"""Nature of a Hermite basis function attached to a knot."""
@enum HermiteKind Value = 0 Slope = 1

"""
    BasisIndex(knot, kind)

Hermite basis function: the knot `knot` (indexed from 1) and its nature `kind`.
An `isbits` type, hence free to construct.
"""
struct BasisIndex
    knot::Int
    kind::HermiteKind
end

"""Linear (1-based) index of the basis function, as used for a column."""
linearindex(b::BasisIndex) = 2 * (b.knot - 1) + Int(b.kind) + 1

"""Inverse of [`linearindex`](@ref)."""
function BasisIndex(lin::Integer)
    q, r = divrem(lin - 1, 2)
    BasisIndex(q + 1, HermiteKind(r))
end

"""
    SplineAxis(knots, colloc)

A 1D axis: the grid knots and the associated collocation points.

The basis counts `2·length(knots)` functions, and there are as many collocation
points, which is what makes the collocation matrices square.
"""
struct SplineAxis{T<:AbstractFloat}
    knots::Vector{T}
    colloc::Vector{T}

    function SplineAxis(knots::Vector{T}, colloc::Vector{T}) where {T}
        issorted(knots) || throw(ArgumentError("knots must be increasing"))
        length(colloc) == 2length(knots) ||
            throw(DimensionMismatch("2 collocation points per knot are required"))
        new{T}(knots, colloc)
    end
end

"""Number of knots."""
nknots(ax::SplineAxis) = length(ax.knots)

"""Number of basis functions (= number of collocation points)."""
nbasis(ax::SplineAxis) = 2nknots(ax)

Base.eachindex(ax::SplineAxis) = (BasisIndex(l) for l in 1:nbasis(ax))

# ---------------------------------------------------------------------------
# Reference Hermite formulas, on the reduced variable a ∈ [0,1]
#
# `a` equals 0 at the end of the support and 1 at the carrying knot; `σ = ±1`
# orients the half-support (right/left). These are the Fortran's `formu`,
# `formu1` and `formu2`.
# ---------------------------------------------------------------------------

@inline function hermite(kind::HermiteKind, a, h, σ)
    kind === Value ? 3a^2 - 2a^3 : h * (a^2 - a^3) * σ
end

@inline function hermite′(kind::HermiteKind, a, h, σ)
    kind === Value ? -6 * (a - a^2) * σ / h : -2a + 3a^2
end

@inline function hermite″(kind::HermiteKind, a, h, σ)
    kind === Value ? 6 * (1 - 2a) / h^2 : (2 - 6a) * σ / h
end

"""
    support(ax, b) -> (xmin, xmax)

Bounds of the support of the basis function `b`.
"""
function support(ax::SplineAxis, b::BasisIndex)
    g, n = ax.knots, nknots(ax)
    lo = b.knot > 1 ? g[b.knot-1] : g[1]
    hi = b.knot < n ? g[b.knot+1] : g[n]
    (lo, hi)
end

"""
    localcoords(ax, b, x) -> (a, h, σ) or `nothing`

Reduced coordinate of `x` within the support of `b`, or `nothing` outside it.
"""
@inline function localcoords(ax::SplineAxis{T}, b::BasisIndex, x) where {T}
    g, n, k = ax.knots, nknots(ax), b.knot
    if k > 1 && x <= g[k]          # left half-support, oriented σ = -1
        x < g[k-1] && return nothing
        h = g[k] - g[k-1]
        return ((x - g[k-1]) / h, h, -one(T))
    elseif k < n && x >= g[k]      # right half-support, oriented σ = +1
        x > g[k+1] && return nothing
        h = g[k+1] - g[k]
        return ((g[k+1] - x) / h, h, one(T))
    end
    nothing
end

"""
    evaluate(ax, b, x, Val(D)) -> NTuple{D+1}

Value of the basis function `b` at `x` and its first `D` derivatives. Returns
zeros outside the support.
"""
@inline function evaluate(ax::SplineAxis{T}, b::BasisIndex, x, ::Val{D}) where {T,D}
    loc = localcoords(ax, b, x)
    loc === nothing && return ntuple(_ -> zero(T), Val(D + 1))
    a, h, σ = loc
    ntuple(Val(D + 1)) do d
        d == 1 ? hermite(b.kind, a, h, σ) :
        d == 2 ? hermite′(b.kind, a, h, σ) :
        hermite″(b.kind, a, h, σ)
    end
end

"""Value of the basis function `b` at `x`."""
@inline value(ax::SplineAxis, b::BasisIndex, x) = evaluate(ax, b, x, Val(0))[1]

# ---------------------------------------------------------------------------
# Building an axis
# ---------------------------------------------------------------------------

"""
Two-point Gauss-Legendre nodes mapped onto `[0,1]`.

This is the original code's choice of collocation points: it makes cubic spline
collocation superconvergent.
"""
const GAUSS2_NODES = ((1 - 1 / sqrt(3)) / 2, (1 + 1 / sqrt(3)) / 2)

"""
    collocation_points(knots) -> Vector

Collocation points associated with a set of knots: the 2 Gauss points of each
interval, bracketed by the two ends of the domain.

There are `2·length(knots)` of them, as many as basis functions — which is what
makes the collocation matrices square.
"""
collocation_points(knots::AbstractVector) =
    [knots[1];
     [knots[j] + (knots[j+1] - knots[j]) * u
      for j in 1:(length(knots)-1) for u in GAUSS2_NODES];
     knots[end]]

"""
    knots_from_collocation(colloc) -> Vector

Reconstructs the knots from the collocation points alone.

The two Gauss points of an interval determine it unambiguously: from
`c₁ = a + h·u₁` and `c₂ = a + h·u₂` one gets `h = (c₂−c₁)/(u₂−u₁)` then
`a = c₁ − h·u₁`.

Useful for replaying an oracle dump that carries only its collocation grid: the
axis follows from it, with no need to guess which other routine to borrow the
knots from.
"""
function knots_from_collocation(colloc::AbstractVector{T}) where {T}
    n = length(colloc) ÷ 2 - 1                     # number of intervals
    u1, u2 = GAUSS2_NODES
    knots = Vector{T}(undef, n + 1)
    knots[1] = colloc[1]
    for j in 1:n
        c1, c2 = colloc[2j], colloc[2j+1]
        h = (c2 - c1) / (u2 - u1)
        knots[j] = c1 - h * u1
        knots[j+1] = knots[j] + h
    end
    knots[1] = colloc[1]                            # exact bound of the domain
    knots[end] = colloc[end]
    knots
end

"""
    axis_from_collocation(colloc) -> SplineAxis

Axis reconstructed from its collocation points alone.
"""
axis_from_collocation(colloc::AbstractVector) =
    SplineAxis(knots_from_collocation(colloc), collect(colloc))

"""
    uniform_axis(x0, xn, nintervals)

Constant-step axis over `[x0, xn]` — the original code's fine grid (`mkgri`
with a geometric ratio of 1).
"""
function uniform_axis(x0::T, xn::T, nintervals::Integer) where {T<:AbstractFloat}
    knots = collect(range(x0, xn; length = nintervals + 1))
    SplineAxis(knots, collocation_points(knots))
end

"""
    stretch_ratio(h1, L, n) -> a

Geometric ratio `a > 1` such that `n` steps of ratio `a` cover the length `L`
**starting** with a step of length `h1`:

    L·(1 − a) / (1 − aⁿ) = h1

This is what joins the stretched zone to the constant-step zone without a break
in step size. Solved by bisection until the floating-point numbers run out.

⚠️ The Fortran (`findacc`) stopped at a tolerance of `1e-10`: the ratio obtained
here differs by that much, and so do the knots stretched with it. That is an
*expected* discrepancy, not a regression — see `stretched_axis`.
"""
function stretch_ratio(h1::T, L::T, n::Integer) where {T<:AbstractFloat}
    f(a) = L * (1 - a) / (1 - a^n) - h1
    lo, hi = nextfloat(one(T)), T(10)
    flo = f(lo)
    signbit(flo) == signbit(f(hi)) && throw(ArgumentError(
        "no geometric ratio in ]1, 10] for h1=$h1, L=$L, n=$n"))
    # Bisection until the floats run out: the bracket is narrowed as long as a
    # float remains strictly between the two bounds.
    while nextfloat(lo) < hi
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        signbit(f(mid)) == signbit(flo) ? (lo = mid) : (hi = mid)
    end
    (lo + hi) / 2
end

"""
    stretched_axis(xinner, xouter, n_inner, n_outer)

Symmetric two-zone axis — the original code's coarse grid (`mkgri2`): constant
step over `[0, xinner]`, then geometrically growing steps up to `xouter`, the
whole thing mirrored about zero.

The first stretched step equals the constant step exactly, which avoids a break
in the mesh at the interface between the two zones.

⚠️ **Do not use this axis to compare downstream results against the oracle.**
The geometric ratio is solved here to machine precision, where `findacc` stopped
at `1e-10`: the knots differ by that much, and any downstream computation would
inherit the discrepancy, masking genuine regressions at `1e-15`. To validate an
operator on the stretched grid, build the `SplineAxis` **from the knots dumped
by the oracle**.
"""
function stretched_axis(xinner::T, xouter::T, n_inner::Integer,
                        n_outer::Integer) where {T<:AbstractFloat}
    h1 = xinner / (n_inner - 1)
    L = xouter - xinner
    a = stretch_ratio(h1, L, n_outer)
    step = L * (1 - a) / (1 - a^n_outer)

    # Half-grid from 0 outwards: the constant-step zone, then the accumulated
    # geometric steps.
    half = [range(0, xinner; length = n_inner);
            xinner .+ cumsum(step .* a .^ (0:n_outer-1))]

    # Mirrored about zero, the central knot not being taken twice.
    knots = [-reverse(half[2:end]); half]
    SplineAxis(knots, collocation_points(knots))
end

# ---------------------------------------------------------------------------
# Moments ∫ xᵏ φ(x) dx
#
# They carry the multipole boundary conditions (monopole, dipole, quadrupole) of
# the Poisson solver.
#
# The Fortran obtained them from hand-written analytic antiderivatives (`prim`,
# `primx`, `primx2` and their `formp*`, ~290 lines). Here they are integrated by
# quadrature: `φ` is cubic, so `x²φ` is of degree 5, and three-point
# Gauss-Legendre quadrature is **exact** up to degree 5. The result is therefore
# not an approximation.
# ---------------------------------------------------------------------------

const GAUSS3_NODES = (-sqrt(3 / 5), 0.0, sqrt(3 / 5))
const GAUSS3_WEIGHTS = (5 / 9, 8 / 9, 5 / 9)

"""Integrates `xᵏ φ_b(x)` over `[a, c]` — exact, the integrand being of degree ≤ 5."""
@inline function _gauss3(ax::SplineAxis{T}, b::BasisIndex, a, c, ::Val{k}) where {T,k}
    mid, half = (a + c) / 2, (c - a) / 2
    # `map` over two tuples is unrolled at compile time: no iterator, no boxed
    # accumulator.
    half * sum(map(GAUSS3_NODES, GAUSS3_WEIGHTS) do ξ, w
        x = mid + half * ξ
        w * x^k * value(ax, b, x)
    end)
end

"""
    moment(ax, b, Val(k)) -> T

Moment of order `k` of the basis function `b`: `∫ xᵏ φ_b(x) dx` over its whole
support. Exact for `k ≤ 2`.

Each half-support is integrated separately: `φ_b` is a *different* polynomial on
each, and a single quadrature over the whole support would be wrong.
"""
function moment(ax::SplineAxis{T}, b::BasisIndex, ::Val{k}) where {T,k}
    g, n, kn = ax.knots, nknots(ax), b.knot
    total = zero(T)
    kn > 1 && (total += _gauss3(ax, b, g[kn-1], g[kn], Val(k)))
    kn < n && (total += _gauss3(ax, b, g[kn], g[kn+1], Val(k)))
    total
end

"""
    moments(ax, Val(k)) -> Vector

Moments of order `k` of every basis function, in linear index order (the
Fortran's `psx`, `psxx` and `psx2` for `k = 0, 1, 2`).
"""
moments(ax::SplineAxis, ::Val{k}) where {k} =
    [moment(ax, BasisIndex(lin), Val(k)) for lin in 1:nbasis(ax)]

"""First derivative of `b` at `x`."""
@inline derivative(ax::SplineAxis, b::BasisIndex, x) = evaluate(ax, b, x, Val(1))[2]

"""Second derivative of `b` at `x`."""
@inline curvature(ax::SplineAxis, b::BasisIndex, x) = evaluate(ax, b, x, Val(2))[3]

"""
    LocateTable(ax)

Replaces the bisection in [`locate`](@ref) by a table lookup.

A **uniform** subdivision of the axis, fine enough that no collocation interval
contains fewer than one bin, gives a candidate cell directly; at most one step of
correction is then needed.

It pays off because the coarse grid, though **stretched**, is not stretched by
much: a factor of three between its smallest and largest step, hence a table of
a few hundred entries. Measured at 800 000 particles: **×14.6** on `locate`,
which accounted for half the coarse deposition.
"""
struct LocateTable{T<:AbstractFloat}
    x0::T
    invwidth::T
    cell::Vector{Int32}
end

function LocateTable(ax::SplineAxis{T}) where {T}
    gt = ax.colloc
    n = length(gt)
    width = minimum(diff(gt)) / 2
    nbin = ceil(Int, (gt[end] - gt[1]) / width) + 2
    cell = [Int32(clamp(searchsortedlast(gt, gt[1] + (b - 1) * width), 1, n - 1))
            for b in 1:nbin]
    LocateTable{T}(gt[1], inv(width), cell)
end

"""
    locate(tbl, ax, x) -> (cell, weight) or `nothing`

Same contract as [`locate`](@ref), by table. Returns **exactly** the same result
— verified, and that is a test.
"""
@inline function locate(tbl::LocateTable{T}, ax::SplineAxis{T}, x) where {T}
    gt = ax.colloc
    (x < gt[1] || x > gt[end]) && return nothing
    @inbounds begin
        b = min(floor(Int, (x - tbl.x0) * tbl.invwidth) + 1, length(tbl.cell))
        c = Int(tbl.cell[b])
        last = length(gt) - 1
        while c < last && gt[c+1] < x
            c += 1
        end
        (c, (gt[c+1] - x) / (gt[c+1] - gt[c]))
    end
end
