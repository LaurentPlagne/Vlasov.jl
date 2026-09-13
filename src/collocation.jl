"""
Collocation matrices of a spline axis.

`S[k, i] = φᵢ(τₖ)`, where `τₖ` is the k-th collocation point and `φᵢ` the i-th
basis function. Likewise for the first (`S′`) and second (`S″`) derivatives.
These matrices are **pentadiagonal** and **non-symmetric**: rows index points,
columns index functions.
"""

"""Half-bandwidth of the Hermite collocation matrices."""
const COLLOCATION_BANDWIDTH = (2, 2)

"""
    CollocationMatrices(axis)

Assembles `S`, `S′` and `S″` for a given axis, as banded matrices.

The two collocation points interior to each interval only "see" the 4 basis
functions of the two knots bounding it: the structure is pentadiagonal, and
banded storage makes that explicit rather than merely implied. The two extreme
rows encode the boundary conditions.
"""
struct CollocationMatrices{T,M<:AbstractMatrix{T}}
    axis::SplineAxis{T}
    S::M
    S′::M
    S″::M
    "Inverse of `S`: maps values at the collocation points to spline
     coefficients. Dense and materialised, since it is applied as a tensor
     product at every time step. `cond(S) ≈ 4`, so inversion is safe."
    Sinv::Matrix{T}
end

function CollocationMatrices(ax::SplineAxis{T}) where {T}
    m = nbasis(ax)
    S, S′, S″ = (BandedMatrix(Zeros{T}(m, m), COLLOCATION_BANDWIDTH) for _ in 1:3)

    # An interval `j` joins knots j and j+1; it carries collocation points 2j
    # and 2j+1, and the 4 basis functions with linear indices 2j-1 … 2j+2. All
    # those (row, column) pairs fit inside the band.
    for j in 1:(nknots(ax)-1), lin in (2j-1):(2j+2), k in (2j):(2j+1)
        S[k, lin], S′[k, lin], S″[k, lin] =
            evaluate(ax, BasisIndex(lin), ax.colloc[k], Val(2))
    end

    # Boundary conditions: first and last rows.
    first_b, last_b = BasisIndex(1), BasisIndex(m - 1)
    S[1, 1], S′[1, 1], S″[1, 1] = evaluate(ax, first_b, ax.colloc[1], Val(2))
    S[m, m-1], S′[m, m-1], S″[m, m-1] = evaluate(ax, last_b, ax.colloc[m], Val(2))

    # `inv(Matrix(S))` and not `inv(S)`: see the warning in `laplacian1d` about
    # `BandedMatrix` divisions.
    CollocationMatrices{T,typeof(S)}(ax, S, S′, S″, inv(Matrix(S)))
end

"""
    laplacian1d(cm) -> Matrix

Second-derivative operator expressed **in the space of values at the collocation
points**: `D = S″ · S⁻¹`.

The division exploits the banded structure of `S` (an `O(n·b²)` factorisation),
but the quotient is full: `D` is returned dense, which is what the downstream
diagonalisation expects anyway.

⚠️ The `lu` factorisation is explicit, and the numerator is densified **before**
the division, for two reasons measured on `BandedMatrices` v1.12.0:

  * `S″ / S` between two `BandedMatrix` returns a **wrong** result without
    signalling anything — residual `‖D·S − S″‖/‖S″‖ ≈ 0.3` while `cond(S) ≈ 4`;
  * `S″ / lu(S)` with a **banded** numerator does not terminate.

Only `Matrix(S″) / lu(S)` is both correct and terminating. Left division
(`S \\ v`) is fine; it is right division that is at fault.

Dirichlet conditions are applied by dropping the first and last basis functions
(this was `extract` in the Fortran).

`D` is not symmetric, but its spectrum is real and strictly negative — which is
what makes the real diagonalisation used by the tensor solver legitimate.
"""
laplacian1d(cm::CollocationMatrices) = laplacian1d_full(cm)[2:end-1, 2:end-1]

"""
    laplacian1d_full(cm) -> Matrix

The operator before the boundary basis functions are dropped.

Its two outermost columns are what makes it possible to **lift** inhomogeneous
Dirichlet conditions: they say what the operator applied to the two dropped
basis functions amounts to, those being the ones whose value is imposed. Its two
outermost rows, by contrast, approximate no second derivative at all.
"""
laplacian1d_full(cm::CollocationMatrices) = Matrix(cm.S″) / lu(cm.S)
