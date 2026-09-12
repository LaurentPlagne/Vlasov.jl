module Vlasov

using LinearAlgebra
using BandedMatrices

include("splines.jl")
include("collocation.jl")
include("tensorsolver.jl")

export HermiteKind, Value, Slope, BasisIndex, linearindex
export SplineAxis, nknots, nbasis, evaluate, value, derivative, curvature, support
export CollocationMatrices, laplacian1d, COLLOCATION_BANDWIDTH
export DiagonalizedOperator, TensorSolver, apply_mode!, solve!, solve

end # module
