"""
Tensor-product mesh and the associated Poisson solve.

The original code duplicated the whole apparatus by hand from one grid to the
next — 41 variables suffixed `big` in the main program, and 47 arguments passed
to `static`. Here a single structure, instantiated once per grid level.
"""

"""
    SplineMesh(axes...)

Tensor-product mesh of `N` spline axes, with everything the Poisson solve draws
from it: collocation matrices per direction, diagonalised second-derivative
operator, and tensor solver.

Building a mesh does the bulk of the work (assembly, factorisations,
diagonalisations) once and for all; the solves that follow are then cheap. That
is what makes the method attractive for a simulation where only the right-hand
side changes at each time step.
"""
struct SplineMesh{N,T,M}
    axes::NTuple{N,SplineAxis{T}}
    collocation::NTuple{N,CollocationMatrices{T,M}}
    "**Complete** second-derivative operators, boundaries included: their
     outermost columns serve to lift the Dirichlet conditions."
    laplacians::NTuple{N,Matrix{T}}
    "Moments `∫φ`, `∫xφ`, `∫x²φ` of each direction, **already transformed by
     `S⁻ᵀ`**. They allow a density given at the collocation points to be
     integrated without ever forming its spline coefficients — see `multipole`."
    dual_moments::NTuple{N,NTuple{3,Vector{T}}}
    solver::TensorSolver{N,T}
    """Work buffers, full grid and interior grid.

    Allocations count double inside a parallel time loop: they do not merely
    cost their price, they trigger a garbage collection that brings every thread
    to a halt. A mesh is therefore **reusable but not shareable** between
    threads."""
    scratch::NTuple{3,Array{T,N}}
    scratch_inner::Array{T,N}
    """Lookup tables, one per direction — see [`LocateTable`](@ref).

    Deposition on the **coarse** grid searched for the cell by bisection, which
    made up half its cost. The table is built once, here."""
    locators::NTuple{N,LocateTable{T}}
end

function SplineMesh(axes::SplineAxis{T}...) where {T}
    cms = map(CollocationMatrices, axes)
    full = map(laplacian1d_full, cms)
    ops = map(D -> DiagonalizedOperator(D[2:end-1, 2:end-1]), full)
    # `S⁻ᵀ·m` once and for all: this is what spares us applying `S⁻¹` to the 3D
    # array at every integration.
    duals = map(cms) do cm
        ntuple(k -> transpose(cm.Sinv) * moments(cm.axis, Val(k - 1)), 3)
    end
    solver = TensorSolver(ops...)
    full_dims = map(nbasis, axes)
    SplineMesh(axes, cms, full, duals, solver,
               ntuple(_ -> Array{T,length(axes)}(undef, full_dims), 3),
               Array{T,length(axes)}(undef, size(solver)),
               map(LocateTable, axes))
end

"""
    NestedMeshes(levels...)

Hierarchy of nested grids, **from the finest to the coarsest**.

Each level must be strictly contained in the next: the fine grid resolves the
cluster, the coarse one carries the conditions at large distance, and the
boundary values of the former are read from the solution of the latter.

This is the type that replaces the 41 `big`-suffixed variables of the main
program. The number of levels being a parameter, the original code's three-grid
version calls for no additional structure.
"""
struct NestedMeshes{L,N,T,M}
    levels::NTuple{L,SplineMesh{N,T,M}}

    function NestedMeshes(levels::SplineMesh{N,T,M}...) where {N,T,M}
        L = length(levels)
        L >= 1 || throw(ArgumentError("at least one grid is required"))
        for l in 1:(L-1), d in 1:N
            inner, outer = levels[l].axes[d], levels[l+1].axes[d]
            outer.knots[1] <= inner.knots[1] && inner.knots[end] <= outer.knots[end] ||
                throw(ArgumentError(
                    "level $l sticks out of level $(l+1) along direction $d"))
        end
        new{L,N,T,M}(levels)
    end
end

Base.length(::NestedMeshes{L}) where {L} = L
Base.getindex(n::NestedMeshes, l::Integer) = n.levels[l]
Base.iterate(n::NestedMeshes, s = 1) = s > length(n) ? nothing : (n.levels[s], s + 1)

"""The finest level."""
finest(n::NestedMeshes) = n.levels[1]

"""The coarsest level, the one carrying the conditions at large distance."""
coarsest(n::NestedMeshes) = n.levels[end]

"""Number of mesh dimensions."""
Base.ndims(::SplineMesh{N}) where {N} = N

"""
Dimensions of the **interior** problem, that is, after removing the basis
functions carrying the Dirichlet conditions. This is the size of the arrays
[`solve!`](@ref) expects.
"""
Base.size(mesh::SplineMesh) = size(mesh.solver)
Base.size(mesh::SplineMesh, d::Integer) = size(mesh)[d]

"""
    collocation_axes(mesh)

Collocation points of each direction, restricted to the interior — the
coordinates at which a right-hand side must be sampled.
"""
collocation_axes(mesh::SplineMesh) = map(ax -> ax.colloc[2:end-1], mesh.axes)

"""
    solve!(φ, ρ, mesh)

Solves `∇²φ = ρ` at the interior collocation points, with homogeneous Dirichlet
conditions. `φ` and `ρ` may be the same array.
"""
solve!(φ::AbstractArray{T,N}, ρ::AbstractArray{T,N}, mesh::SplineMesh{N,T}) where {T,N} =
    solve!(φ, ρ, mesh.solver)

"""Allocating version of [`solve!`](@ref)."""
solve(ρ::AbstractArray{T,N}, mesh::SplineMesh{N,T}) where {T,N} = solve!(similar(ρ), ρ, mesh)

"""
    laplacian!(dest, φ, mesh)

Applies the operator `∇² = Σ_d D_d` — the forward operation, of which
[`solve!`](@ref) is the inverse. Used to check a residual.
"""
function laplacian!(dest::AbstractArray{T,N}, φ::AbstractArray{T,N}, mesh::SplineMesh{N,T}) where {T,N}
    fill!(dest, zero(T))
    tmp = similar(dest)
    for d in 1:N
        @views apply_mode!(tmp, mesh.laplacians[d][2:end-1, 2:end-1], φ, d)
        dest .+= tmp
    end
    dest
end
