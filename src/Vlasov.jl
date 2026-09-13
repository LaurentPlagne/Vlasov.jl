module Vlasov

using LinearAlgebra
using BandedMatrices

include("threading.jl")
include("random.jl")
include("splines.jl")
include("collocation.jl")
include("tensorsolver.jl")
include("mesh.jl")
include("deposition.jl")
include("poisson.jl")
include("particles.jl")
include("fields.jl")
include("meanfield.jl")
include("initial.jl")
include("energy.jl")
include("projectile.jl")
include("simulation.jl")

export Ran2, next!
export configure_blas!
export HermiteKind, Value, Slope, BasisIndex, linearindex
export SplineAxis, nknots, nbasis, evaluate, value, derivative, curvature, support
export uniform_axis, stretched_axis, stretch_ratio, collocation_points, moment, moments
export knots_from_collocation, axis_from_collocation
export CollocationMatrices, laplacian1d, laplacian1d_full, COLLOCATION_BANDWIDTH
export DiagonalizedOperator, TensorSolver, apply_mode!, apply_rotating!, apply_all_rotating!, solve!, solve
export SplineMesh, NestedMeshes, finest, coarsest, collocation_axes, laplacian!
export ScatterBuffers, scatter!, scatter_reduce!
export dual_lengths, locate, deposit!, spline_coefficients, spline_coefficients!, total_charge
export Multipole, multipole, potential, boundary_potential!, poisson_rhs, poisson_rhs!
export poisson, poisson!, solve_interior!, boundary_from_coarse!
export ParticleCloud, ELECTRON_MASS, ELECTRON_CHARGE, mass, charge
export step!, half_step_back, full_step_back
export GaussianSmoothing, smoothed_field, smoothed_potential, spline_field, spline_potential, forces!, nearest_knot, cell_index
export deposit_smoothed!
export Jellium, WIGNER_SEITZ_NA, xc_potential, xc_energy_density
export effective_potential!, uniform_sphere_potential
export PhaseSpaceProfile, RadialProfile, read_radial_profile, sample_thomas_fermi
export PotentialProfile, read_potential_profile, initial_cloud, FERMI_COEFFICIENT
export EnergyBudget, energy_budget, interaction_energy, hartree_energy, ion_self_energy
export SimulationParameters, read_parameters, Simulation, run!
export Projectile, projectile_forces!, energy_loss, kinetic_energy, HARTREE_TO_EV
export enclosed_charge, capture!, advance_projectile!

end # module
