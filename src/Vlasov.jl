module Vlasov

using LinearAlgebra
using BandedMatrices

include("splines.jl")
include("collocation.jl")
include("tensorsolver.jl")
include("mesh.jl")
include("deposition.jl")
include("poisson.jl")

export HermiteKind, Value, Slope, BasisIndex, linearindex
export SplineAxis, nknots, nbasis, evaluate, value, derivative, curvature, support
export uniform_axis, stretched_axis, stretch_ratio, collocation_points, moment, moments
export CollocationMatrices, laplacian1d, laplacian1d_full, COLLOCATION_BANDWIDTH
export DiagonalizedOperator, TensorSolver, apply_mode!, solve!, solve
export SplineMesh, collocation_axes, laplacian!
export dual_lengths, locate, deposit!, spline_coefficients, spline_coefficients!, total_charge
export Multipole, multipole, potential, boundary_potential!, poisson_rhs, poisson_rhs!
export poisson, poisson!

end # module
