"""
Tensor solver by fast diagonalisation (the TBSCM method).

Solves `(D₁⊗I⊗… + I⊗D₂⊗… + …) X = B` when the operator is **separable**, by
diagonalising each 1D operator:

    Dᵈ = Mᵈ Λᵈ Mᵈ⁻¹   ⟹   X = (⨂ Mᵈ) ∘ diag(1/Σλ) ∘ (⨂ Mᵈ⁻¹) B

Ref.: L. Plagne, J.-Y. Berthou, *Tensorial basis spline collocation method for
Poisson's equation*, J. Comput. Phys. **157**(2), 419-440 (2000).

Generalises `LidJul.PoissonTTSolver` on two points:

  * **arbitrary dimension** `N` (the Vlasov code is 3D) through a generic
    mode-`d` product, instead of the `transpose` interplay specific to 2D;
  * **non-symmetric operators** — the spline collocation matrices are not
    symmetric, unlike finite-difference Laplacians.
"""

"""
    DiagonalizedOperator(D)

Diagonalisation of a 1D operator `D` whose spectrum is real.

Realness is checked at construction: it is an *observed* property of the spline
operator, not a structural one guaranteed by the type, and a numerical
regression ought to be caught here rather than propagate silently.
"""
struct DiagonalizedOperator{T<:AbstractFloat}
    M::Matrix{T}       # eigenvectors
    Minv::Matrix{T}    # its inverse
    λ::Vector{T}       # eigenvalues

    function DiagonalizedOperator(D::AbstractMatrix{T}; atol = 1e-10) where {T}
        E = eigen(D)
        imagmax = maximum(abs, imag.(E.values))
        scale = maximum(abs, real.(E.values))
        imagmax <= atol * max(scale, one(T)) || throw(ArgumentError(
            "non-real spectrum (max|Im λ| = $imagmax): the real tensor solver " *
            "does not apply"))
        M = real.(E.vectors)
        new{T}(M, inv(M), real.(E.values))
    end
end

Base.size(d::DiagonalizedOperator) = length(d.λ)

"""
    TensorSolver(operators...)

Solver for the tensor sum of the `operators` (one per dimension).

Precomputes `1/(λ¹ᵢ + λ²ⱼ + …)`, which is the heart of the method: inverting an
`N`-dimensional operator reduces to an element-wise division.
"""
struct TensorSolver{N,T<:AbstractFloat}
    ops::NTuple{N,DiagonalizedOperator{T}}
    invλsum::Array{T,N}
    work1::Array{T,N}
    work2::Array{T,N}
end

function TensorSolver(ops::DiagonalizedOperator{T}...) where {T}
    N = length(ops)
    dims = ntuple(d -> size(ops[d]), N)

    λsum = [sum(ops[d].λ[I[d]] for d in 1:N) for I in CartesianIndices(dims)]
    singular = findfirst(iszero, λsum)
    singular === nothing || throw(ArgumentError(
        "eigenvalue sum vanishes at $(Tuple(singular)): singular operator"))

    TensorSolver{N,T}(ops, inv.(λsum),
                      Array{T,N}(undef, dims), Array{T,N}(undef, dims))
end

Base.size(s::TensorSolver) = size(s.invλsum)

"""
    apply_mode!(dest, A, src, d)

Mode-`d` product: applies the matrix `A` along dimension `d` of the array `src`,
result in `dest`. Allocation-free.

The array is viewed as `(L, nᵈ, R)`; for `d = 1` a single `mul!` suffices,
otherwise we loop over slices — each of which stays contiguous in memory.
"""
function apply_mode!(dest::AbstractArray{T,N}, A::AbstractMatrix{T},
                     src::AbstractArray{T,N}, d::Integer) where {T,N}
    dims = size(src)
    L = prod(ntuple(i -> dims[i], d - 1))
    n = dims[d]
    R = prod(ntuple(i -> dims[d+i], N - d))

    if L == 1
        mul!(reshape(dest, n, R), A, reshape(src, n, R))
    else
        S = reshape(src, L, n, R)
        D = reshape(dest, L, n, R)
        @views for r in 1:R
            mul!(D[:, :, r], S[:, :, r], transpose(A))
        end
    end
    dest
end

"""
    apply_rotating!(dest, A, src, dims) -> permuted dims

Applies `A` along the **first** dimension, in a single matrix-matrix product,
and returns the dimensions cyclically permuted.

    mul!(C, Xᵀ, Aᵀ)  computes  C = (A·X)ᵀ

The transposition is not a detour: it is what performs the permutation. The
`(m, n)` result reads back as is as an array of dimensions `(d₂, d₃, …, d₁)`,
without moving a single byte. Applying it `N` times therefore restores the
dimensions to their original order.

Two consequences: **one GEMM per dimension** instead of a loop over slices for
the middle dimensions, and a shape — one large matrix-matrix product — which is
exactly what a GPU runs best.
"""
@inline function apply_rotating!(dest::AbstractArray{T,N}, A::AbstractMatrix{T},
                                 src::AbstractArray{T,N}, dims::NTuple{N,Int}) where {T,N}
    n = dims[1]
    m = length(src) ÷ n
    mul!(reshape(dest, m, n), transpose(reshape(src, n, m)), transpose(A))
    ntuple(i -> dims[mod1(i + 1, N)], N)
end

"""
    apply_all_rotating!(dest, mats, src, work) -> dest

Applies `mats[d]` along each dimension, by `N` successive rotations.

After `N` rotations the dimensions have regained their order: this is the only
pattern the code needs — the tensor solver uses it twice, the conversion to
spline coefficients once. A single BLAS kernel for the whole time loop, and a
single place to port to a GPU when the day comes.

`work` is a buffer the size of `src`, supplied by the caller: this function
allocates nothing. `dest` must be distinct from `src`.

⚠️ The alternation between `dest` and `work` follows from the **parity of the
number of remaining steps**, not from an ad-hoc counter. A single-buffer
ping-pong would make source and destination coincide from the second step on —
reading and writing would tread on each other, silently.
"""
function apply_all_rotating!(dest::AbstractArray{T,N}, mats, src::AbstractArray{T,N},
                             work::AbstractArray{T,N}) where {T,N}
    dest === src && throw(ArgumentError("`dest` and `src` must be distinct"))
    dims = size(src)
    cur = src
    for d in 1:N
        out = iseven(N - d) ? dest : work
        dims = apply_rotating!(out, mats[d], cur, dims)
        cur = out
    end
    dest
end

"""
    solve!(X, B, solver)

Solves the tensor operator for the right-hand side `B`, result in `X`. `X` and
`B` may be the same array.

`2N` matrix-matrix products and one element-wise division, with no allocation at
all: both buffers belong to the solver. Allocations count double here — they do
not merely cost their price, they trigger a garbage collection that brings the
threads to a halt.
"""
function _solve!(X, B, Minvs, Ms, invλsum, work1, work2)
    # Forward transform: into the eigenbasis of each dimension.
    apply_all_rotating!(work1, Minvs, B, work2)

    # The heart of the method: inversion becomes an element-wise division.
    # ⚠️ `vec`, and it is not cosmetic: a Metal broadcast over a **3-D** array
    # is slower than the same one flattened, for the same bytes and the same
    # buffer underneath. Measured on `a .+= b`, freshly allocated each time:
    #
    #   n     3-D      vec     ratio
    #   256   131.0    258.8   1.98
    #   257   164.0    262.6   1.60
    #   258   106.9    267.5   2.50      <- our grid
    #   260   166.3    266.0   1.60
    #   512   261.1    358.5   1.37
    #
    # The cause is **not** established. What can be said is that the cartesian
    # path turns a linear work-item index into `(i,j,k)` with two integer
    # divisions per element and the flat path does not, and that the penalty is
    # there at every size tried — 1.4× to 2.5×. It is *not* a matter of the
    # leading dimension being a power of two or a multiple of four: 256 is both
    # and is among the slowest, 257 is neither and does better.
    #
    # Both operands are dense and the same shape, so the flattening is exact.
    vec(work1) .*= vec(invλsum)

    # Inverse transform. `X` may be `B`: the right-hand side has already been
    # fully consumed by the forward transform.
    apply_all_rotating!(X, Ms, work1, work2)
end

function solve!(X::AbstractArray{T,N}, B::AbstractArray{T,N}, s::TensorSolver{N,T}) where {T,N}
    size(X) == size(B) == size(s) ||
        throw(DimensionMismatch("dimensions incompatible with the solver"))
    _solve!(X, B, map(o -> o.Minv, s.ops), map(o -> o.M, s.ops),
            s.invλsum, s.work1, s.work2)
end

"""Allocating version of [`solve!`](@ref)."""
solve(B::AbstractArray{T,N}, s::TensorSolver{N,T}) where {T,N} = solve!(similar(B), B, s)
