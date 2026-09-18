"""
Electric field as seen by the pseudo-particles, and the forces that follow.

A pseudo-particle is not point-like: it is a Gaussian packet of width `σ`. The
field it feels is therefore the gradient of the potential **convolved** with
that Gaussian, which softens the close collisions that discretisation would
otherwise make singular.

On a constant-step grid this convolution depends only on the particle's position
**within its cell**: the overlaps `∫φₐ(x')·G(x−x')dx'` and their derivatives are
tabulated once and for all, then interpolated.
"""

"""
    GaussianSmoothing(axis; nbdt = 1000, quadrature = 1000)

Convolution tables of a constant-step axis (the Fortran's `maketaint`).

`overlap[a, i]` is the overlap of the a-th of the 10 neighbouring basis
functions with a Gaussian centred at `r`, and `gradient[a, i]` its derivative
with respect to `r` — that is what gives the field. The index `i` discretises
the position of `r` within a cell, in `nbdt + 1` values.

The width is `σ = h/3`, tied to the grid step: the original code's choice, which
sets the smoothing at the scale of the resolution.

⚠️ **The quadrature is the Fortran's** — trapezoids over `quadrature` intervals
— and not a high-order rule. This is no oversight: it is what defines the
oracle's values, and changing it would soften every downstream comparison from
`1e-14` to `1e-9`, the forces included.

⚠️ **The tabulated kernel is not exactly normalised** (measured):

  * `Σ overlaps` equals 1 to within `3.4e-6`;
  * the derivative of a constant function returns `1.3e-5` instead of 0.

The smoothed field therefore carries a relative error of order `1e-5` — part of
it transverse, a potential depending on `x` alone producing a non-zero field
along `y`. This is a limit of the **original method**, not of the port: the
window of 10 functions does not capture the whole Gaussian, and the two outermost
functions are integrated over a truncated support. The unsmoothed field, for its
part, is exact up to rounding.
"""
struct GaussianSmoothing{T<:AbstractFloat}
    σ::T
    spacing::T
    nbdt::Int
    overlap::Matrix{T}
    gradient::Matrix{T}
    "Gaussian evaluated at the 8 neighbouring collocation points (the Fortran's
     `gausstab`), for the smoothed charge deposition."
    nodes::Matrix{T}
end

"""
Integration bounds of the 10 neighbouring basis functions, in knot indices
relative to the tabulation window. The two outermost are truncated: their
overlap with the Gaussian there is of order `e⁻¹⁸`.
"""
const SMOOTHING_SUPPORTS = ((1, 2), (1, 2), (1, 3), (1, 3), (2, 4),
                            (2, 4), (3, 5), (3, 5), (4, 5), (4, 5))

function GaussianSmoothing(ax::SplineAxis{T}; nbdt::Int = 1000,
                           quadrature::Int = 1000) where {T}
    g = ax.knots
    h = g[2] - g[1]
    σ = h / 3
    norm1d = inv(sqrt(2 * T(π)) * σ)   # 1D normalisation of a 3D Gaussian

    # `r` sweeps a cell centred on the knot g[3]; the translation invariance of
    # the regular grid does the rest.
    rmin, rmax = (g[3] + g[2]) / 2, (g[4] + g[3]) / 2

    overlap = Matrix{T}(undef, 10, nbdt + 1)
    gradient = Matrix{T}(undef, 10, nbdt + 1)
    nodes = Matrix{T}(undef, 8, nbdt + 1)

    for i in 0:nbdt
        r = rmin + i * (rmax - rmin) / nbdt
        gauss(x) = norm1d * exp(-(r - x)^2 / 2σ^2)
        for a in 1:10
            lo, hi = SMOOTHING_SUPPORTS[a]
            b = BasisIndex(a)
            overlap[a, i+1] = trapezoid(g[lo], g[hi], quadrature) do x
                value(ax, b, x) * gauss(x)
            end
            gradient[a, i+1] = trapezoid(g[lo], g[hi], quadrature) do x
                -(r - x) / σ^2 * value(ax, b, x) * gauss(x)
            end
        end
        # The 8 collocation points of the window, for the deposition.
        for a in 1:8
            nodes[a, i+1] = gauss(ax.colloc[a+1])
        end
    end
    GaussianSmoothing{T}(σ, h, nbdt, overlap, gradient, nodes)
end

"""
    trapezoid(f, a, b, n)

Trapezoidal rule over `n` intervals. The abscissae are accumulated (`x += step`)
as in the Fortran: rebuilding `a + i·step` would give very slightly different
points, and the discrepancy would show up in the comparison.
"""
function trapezoid(f, a::T, b::T, n::Integer) where {T}
    step = (b - a) / n
    x = a
    fx = f(x)
    total = zero(T)
    for _ in 1:n
        xnext = x + step
        fnext = f(xnext)
        total += (fx + fnext) * step / 2
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

"""Table column corresponding to the position of `x` within its cell."""
@inline function table_column(sm::GaussianSmoothing, x, knot)
    # `floor(v + 1/2)` and not `round`: Julia rounds to the nearest even, the
    # Fortran breaks ties upwards.
    floor(Int, (x - knot + sm.spacing / 2) / sm.spacing * sm.nbdt + 0.5) + 1
end

"""
    deposit_smoothed!(ρ, mesh, sm, positions; charge) -> nout

**Smoothed** charge deposition on the fine grid (the Fortran's `makerhog`).

Each pseudo-particle spreads its weight over the 8³ neighbouring collocation
points according to the tabulated Gaussian, instead of the 2³ of the trilinear
interpolation in [`deposit!`](@ref).

The density is then **renormalised** so that the total charge equals exactly
that of the deposited particles. This is no fussiness: the tabulated kernel is
normalised only to within `3e-6`, and without this correction the error would
enter the potential.

A particle is rejected if its 8-point stencil would overflow the grid — whence a
margin of one and a half cells at the boundary.
"""
function deposit_smoothed!(ρ::Array{T,3}, mesh::SplineMesh{3,T},
                           sm::GaussianSmoothing{T}, positions;
                           charge::T, buffers = nothing) where {T}
    knots = map(a -> a.knots, mesh.axes)
    half = sm.spacing / 2
    bounds = map(g -> (g[2] + half, g[end-1] - half), knots)

    nout = scatter!(ρ, mesh, positions, buffers) do dest, _, _, _, p
        all(d -> bounds[d][1] <= p[d] <= bounds[d][2], 1:3) || return false
        ci = ntuple(d -> nearest_knot(knots[d], p[d]), 3)
        col = ntuple(d -> table_column(sm, p[d], knots[d][ci[d]]), 3)
        # `nodes[a]` is the Gaussian at collocation point `colloc[a+1]` of the
        # reference window, whose central knot is the 3rd: the stencil
        # therefore covers `2·ci−4 … 2·ci+3`.
        base = ntuple(d -> 2 * ci[d] - 5, 3)

        gx = @view sm.nodes[:, col[1]]
        gy = @view sm.nodes[:, col[2]]
        gz = @view sm.nodes[:, col[3]]
        @inbounds for kk in 1:8, jj in 1:8
            c = gy[jj] * gz[kk]
            j, k = base[2] + jj, base[3] + kk
            for ii in 1:8
                dest[base[1]+ii, j, k] += gx[ii] * c
            end
        end
        true
    end

    ρ .*= charge
    # Renormalisation: the deposited charge must be that of the particles.
    ρ .*= (length(positions) - nout) * charge / total_charge(ρ, mesh)
    nout
end

"""
    contract_spline_10(csol, ovl, grad, bx, by, bz, cx, cy, cz)

Evaluates the 10x10x10 cubic Hermite spline contraction with Gaussian smoothing tables.
Pure scalar kernel: shared identically between CPU (`Float64`) and Metal GPU (`Float32`).
"""
@inline function contract_spline_10(csol, ovl, grad, bx, by, bz, cx, cy, cz)
    T = eltype(csol)
    fx = zero(T); fy = zero(T); fz = zero(T)
    @inbounds for kk in 1:10
        k = bz + kk - 1
        oz = ovl[kk, cz]; gz = grad[kk, cz]
        for jj in 1:10
            j = by + jj - 1
            oy = ovl[jj, cy]; gy = grad[jj, cy]
            dxp = zero(T); val = zero(T)
            for ii in 1:10
                c = csol[bx + ii - 1, j, k]
                dxp = fma(c, grad[ii, cx], dxp)
                val = fma(c, ovl[ii, cx], val)
            end
            fx = fma(oy * oz, dxp, fx)
            fy = fma(gy * oz, val, fy)
            fz = fma(oy * gz, val, fz)
        end
    end
    (fx, fy, fz)
end

"""
    smoothed_field(axes, csol, sm, p) -> NTuple{3,T}

Smoothed electric field at the point `p` (the Fortran's `champsg`), from the
potential's spline coefficients `csol`.

`E = −∇(Φ ∗ G)`: each direction combines the tabulated overlaps, the
differentiated direction taking `gradient` where the other two take `overlap`.
"""
function smoothed_field(axes::NTuple{3,SplineAxis{T}}, csol::Array{T,3},
                        sm::GaussianSmoothing{T}, p) where {T}
    base = ntuple(3) do d
        2 * nearest_knot(axes[d].knots, p[d]) - 5
    end
    col = ntuple(d -> table_column(sm, p[d], axes[d].knots[nearest_knot(axes[d].knots, p[d])]), 3)
    fx, fy, fz = contract_spline_10(csol, sm.overlap, sm.gradient,
                                    base[1], base[2], base[3],
                                    col[1], col[2], col[3])
    (-fx, -fy, -fz)
end

"""
    spline_field(axes, csol, p) -> NTuple{3,T} or `nothing`

**Unsmoothed** electric field, the direct gradient of the spline interpolant
(the Fortran's `champ`). Returns `nothing` outside the domain.

Used on the coarse grid, where the particles are far from the dense region and
smoothing no longer serves any purpose.
"""
function spline_field(axes::NTuple{3,SplineAxis{T}}, csol::Array{T,3}, p) where {T}
    # Same precaution as in `spline_potential`: rule out the `nothing` before
    # building anything, on pain of type instability.
    cx = cell_index(axes[1].knots, p[1])
    cy = cell_index(axes[2].knots, p[2])
    cz = cell_index(axes[3].knots, p[3])
    (cx === nothing || cy === nothing || cz === nothing) && return nothing
    cells = (cx, cy, cz)

    # The 4 basis functions non-zero in the cell, value and derivative.
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
    smoothed_potential(axes, csol, sm, p) -> T

**Smoothed** potential at the point `p` (the Fortran's `potensg`): the value of
the potential convolved with the pseudo-particle's Gaussian.

It is to [`smoothed_field`](@ref) what the value is to the gradient — the same
tables, but `overlap` in all three directions instead of differentiating one.
Used by the energy budget, which must see the same potential as the forces.
"""
function smoothed_potential(axes::NTuple{3,SplineAxis{T}}, csol::Array{T,3},
                            sm::GaussianSmoothing{T}, p) where {T}
    ci = ntuple(d -> nearest_knot(axes[d].knots, p[d]), 3)
    base = ntuple(d -> 2 * ci[d] - 5, 3)
    col = ntuple(d -> table_column(sm, p[d], axes[d].knots[ci[d]]), 3)

    ox = @view sm.overlap[:, col[1]]
    oy = @view sm.overlap[:, col[2]]
    oz = @view sm.overlap[:, col[3]]

    φ = zero(T)
    @inbounds for kk in 1:10
        k = base[3] + kk - 1
        for jj in 1:10
            j = base[2] + jj - 1
            w = oy[jj] * oz[kk]
            s = zero(T)
            for ii in 1:10
                s += csol[base[1]+ii-1, j, k] * ox[ii]
            end
            φ += s * w
        end
    end
    φ
end

"""
    spline_potential(axes, csol, p) -> T or `nothing`

Value of the spline interpolant at the point `p` (the Fortran's `potentiel`).
Returns `nothing` outside the domain.

Used at the inter-grid junction: the fine grid's boundary values are read from
the coarse grid's solution.
"""
function spline_potential(knots::NTuple{3,<:AbstractVector}, csol, p)
    T = eltype(csol)
    # ⚠️ The three out-of-domain tests come BEFORE any construction:
    # `cell_index` returns `Union{Nothing,Int}`, and letting that union into an
    # `ntuple` propagates it to everything downstream. Inference fails, the
    # tuples are boxed, and evaluating one point goes from 20 ns to 20 µs.
    cx = cell_index(knots[1], p[1])
    cy = cell_index(knots[2], p[2])
    cz = cell_index(knots[3], p[3])
    (cx === nothing || cy === nothing || cz === nothing) && return nothing
    cells = (cx, cy, cz)

    vals = ntuple(3) do d
        c = cells[d]
        ntuple(a -> value(knots[d], BasisIndex(2 * (c - 1) + a), p[d]), 4)
    end

    φ = zero(T)
    @inbounds for ak in 1:4
        k = 2 * (cells[3] - 1) + ak
        for aj in 1:4
            j = 2 * (cells[2] - 1) + aj
            w = vals[2][aj] * vals[3][ak]
            for ai in 1:4
                φ += csol[2*(cells[1]-1)+ai, j, k] * vals[1][ai] * w
            end
        end
    end
    φ
end

spline_potential(axes::NTuple{3,SplineAxis{T}}, csol::AbstractArray{T,3}, p) where {T} =
    spline_potential(map(a -> a.knots, axes), csol, p)

"""
    forces!(cloud, fine, csol_fine, coarse, csol_coarse, sm; escaped) -> Int

Fills the cloud's forces (the Fortran's `force2g`), and returns the number of
particles handled outside the fine grid.

Three regimes, from finest to coarsest:

  1. well inside the fine grid → **smoothed** field;
  2. beyond it → spline gradient of the **coarse** grid;
  3. outside both → Coulomb monopole of the enclosed charge, `escaped` particles
     being unaccounted for.

The force is `w·E`, `w` being the number of electrons the pseudo-particle
carries.
"""
function forces!(cloud::ParticleCloud{T},
                 fine::NTuple{3,SplineAxis{T}}, csol_fine::Array{T,3},
                 coarse::NTuple{3,SplineAxis{T}}, csol_coarse::Array{T,3},
                 sm::GaussianSmoothing{T}; escaped::Integer = 0) where {T}
    w = cloud.weight
    w2 = w * w
    # A two-cell margin: the smoothed field reads 10 basis functions around the
    # point, so it needs two knots on each side.
    lo = ntuple(d -> fine[d].knots[3], 3)
    hi = ntuple(d -> fine[d].knots[end-2], 3)

    # Each particle writes only its own force: the loop splits with no special
    # care. Only the counter calls for a reduction.
    tmapreduce(length(cloud.positions)) do slice
        n = 0
        @inbounds for i in slice
            p = cloud.positions[i]
            if all(d -> lo[d] < p[d] < hi[d], 1:3)
                cloud.forces[i] = w .* smoothed_field(fine, csol_fine, sm, p)
            else
                n += 1
                E = spline_field(coarse, csol_coarse, p)
                cloud.forces[i] = if E === nothing
                    # Outside both grids: all that remains is the enclosed
                    # charge, seen from afar.
                    r3 = (p[1]^2 + p[2]^2 + p[3]^2)^T(1.5)
                    (-w2 * escaped / r3) .* p
                else
                    w .* E
                end
            end
        end
        n
    end
end
