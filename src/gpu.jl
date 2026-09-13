"""
Interface for force evaluation on an accelerator.

The GPU port targets the **particle loops**, which account for 79 % of a step at
production scale — not the tensor solver, which accounts for only 14 % and would
therefore cap out at ×1.16.

Nothing here depends on a backend: the methods are supplied by a package
extension (`ext/VlasovMetalExt.jl` for Metal), loaded only if the user loads
`Metal`. Without a backend these functions raise an explicit error and the CPU
path remains the only one.

⚠️ **The GPU works in `Float32`.** Apple GPUs have no double precision — Metal
has no `double` type. The `Float64` CPU path therefore remains the reference,
the one compared against the Fortran oracle at `1e-13`; the GPU path is
validated against it, at the level the physics demands.
"""

"""
    ForceAccelerator

What a backend must provide to take over evaluation of the smoothed field. The
type is declared here so that [`forces!`](@ref) can name it; the realisations
live in the extensions.

    ForceAccelerator(MtlArray, fine_axes, smoothing, npart, n)

Prepares the computation for a given grid and particle count: tables and buffers
are allocated **once**, not at every step. Allocations kill parallelism, and on
a GPU more so.
"""
abstract type ForceAccelerator end

"""
    forces!(cloud, acc, csol_fine, coarse, csol_coarse, sm; escaped) -> Int

Same contract as the CPU method of [`forces!`](@ref), minus the smoothed-field
evaluation: that goes to the accelerator `acc`.

Particles too close to the boundary for the 10³ stencil to fit inside the grid
are **handed back to the CPU**: they are rare, and handling them on the GPU
would call for branches exactly where the point is to have none.
"""
function forces! end

"""
    deposit_smoothed!(ρ, acc, mesh, sm, positions; charge) -> nout

Same contract as the CPU method of [`deposit_smoothed!`](@ref), minus the
*scatter*: that goes to the accelerator.

Deposition is the hard part to port — each particle writes into 8³ points, and
neighbouring particles write into the same ones. The naive route, one atomic add
per point per particle, is **three times slower than the CPU**: 410 million
contending atomics, which the GPU serialises. The route taken here first orders
the particles by cell ([`CellSort`](@ref)) and gives one cell to one thread
group, each thread owning one stencil point — one atomic per point per **cell**,
a hundred times fewer.

See `docs/gpu.md` for the measurements.
"""
function deposit_smoothed! end
