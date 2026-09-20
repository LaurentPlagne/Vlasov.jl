```@meta
CurrentModule = Vlasov
```

# The device path

How the same source runs a time step on ten CPU cores or on a GPU, what had to
change in the data to make that possible, and why each piece is shaped the way
it is.

This page is about **structure**. The numbers that justify it are in
[Performance](@ref); where a decision rests on a measurement, the measurement is
named and the figure quoted, but the log lives there.

## The problem

A time step is two kinds of work in alternation.

*Grid work* — solve Poisson on `n³` points, add exchange-correlation — costs
`O(n⁴)` in the solver and is dominated by dense linear algebra. *Particle work* —
deposit `N` particles onto the grid, read the field back at their positions,
advance them — costs `O(N)` with a large constant.

At the thesis's scale (20 000 particles, 33³ points) the grid dominated. At the
scale this code now runs (8×10⁷ particles, 222³ points) the particles are **79 %
of a step**, and the grid is almost free. Everything below follows from that one
inversion: the target is the particle loops, and the tensor solver is not worth
moving.

## One kernel, every backend

The particle kernels are written once, in
[`KernelAbstractions`](https://juliagpu.github.io/KernelAbstractions.jl/), and
run wherever that package has a backend:

```julia
_deposit_sorted_kernel!(backend, groupsize)(args...; ndrange = ...)
```

`backend` is `CPU()`, `MetalBackend()`, and in principle CUDA, ROCm or oneAPI —
those three are untested here for want of the hardware. The payoff is not
elegance but **arithmetic that cannot drift apart**: the CPU reference and the
GPU path execute the same expression in the same order, so a discrepancy between
them is a precision effect and never a second implementation of the physics.

What that replaced was four hundred lines of hand-written Metal. Measured
against it at 4×10⁶ particles, the portable field kernel is ×0.999 and **bit for
bit identical**, the deposition ×0.961. The extension that remains is 56 lines,
and it holds exactly one thing.

### The one thing that is genuinely Apple

Where the buffers live.

```julia
function Vlasov.dual_buffer(::Metal.MetalBackend, ::Type{T}, dims::Integer...)
    mtl = MtlArray{T,length(dims),Metal.SharedStorage}(undef, dims...)
    DualBuffer(mtl, unsafe_wrap(Array, mtl), true)   # same bytes, two views
end
```

On Apple Silicon the CPU and the GPU address one pool of memory, so a
[`DualBuffer`](@ref)'s `host` and `device` faces are *the same bytes* and the
transfers are free. On a discrete GPU they are two allocations and the transfers
are real. Call sites do not change; only the price does.

![One allocation, or two](assets/diagrams/dual-buffer.svg)

Flattening this into a uniform "always copy" would cost Apple Silicon a copy of
every buffer at every step, on the very machine the campaigns run on.

!!! danger "A no-op copy still has to order"
    [`download!`](@ref) **synchronises unconditionally**, then copies only where
    the memories differ. A version that merely copied would order nothing on
    unified memory, and the host would read buffers the device had not finished
    writing. The bug is silent and looks like wrong physics — it was caught as a
    density that integrated to the raw particle count, the scaling kernel not
    having run yet.

## Splitting every function in two

Nothing in `src/` branches on a backend. Instead each function of the chain was
split: one half takes the **tables**, one takes the **object** that holds them.

```julia
# the half that does the work — it never asks where its arrays live
_poisson_rhs!(rhs, ρ, φ, laplacians, interior)

# the two half-line methods that supply the tables
poisson_rhs!(rhs, ρ, mesh::SplineMesh, φ) = _poisson_rhs!(rhs, ρ, φ, mesh.laplacians, size(mesh))
poisson_rhs!(rhs, ρ, dm::DeviceMesh,  φ) = _poisson_rhs!(rhs, ρ, φ, dm.laplacians, dm.interior)
```

![One worker, two suppliers of tables](assets/diagrams/split-in-two.svg)

[`DeviceMesh`](@ref) is then a *mirror*, not a reimplementation: it copies, once,
the short list of tables the time loop actually reads, in the kernels'
precision. The mesh itself keeps everything that has no business on a GPU — the
banded collocation factorisations, the eigenbasis as `eigen` returned it, the
locator tables.

The result is that `poisson!` on a host mesh and `poisson!` on a device mirror
**are the same code**, and `_update_forces_resident!` is line for line its host
twin with different arrays.

## Residency: what stays where

A step touches three kinds of state, and each has one right home.

| State | Lives | Why |
|---|---|---|
| Grids: `ρ`, `φ`, `csol` | device | written and read only by kernels |
| Mesh tables | device mirror | constant for the run, copied once |
| The cloud | **device**, as `(k, δ)` | see below |
| Sort counters | device, scan on host | the scan is `O(cells)` and sequential |

Two readbacks survive, and both are deliberate: the coarse coefficients come
back for the handful of particles that left the fine grid, and the energy budget
— a diagnostic taken one step in ten — reads the potential particle by particle
on the host. On unified memory neither is a copy at all.

Drawn out, one step looks like this. Everything inside the shaded band runs as a
kernel; only the three dashed arrows cross to the host, and two of them are
diagnostics:

![What a step crosses](assets/diagrams/step-boundary.svg)

Before the cloud was held as `(k, δ)`, two more arrows crossed that boundary on
**every** step and carried the whole cloud with them: the packing read 8×10⁷
`Float64` triples to produce `(k, δ)`, and the forces were converted back to
host `Float64` for the integrator. Both are gone.

## The cloud, and why it changed shape

This is the part worth reading slowly, because the reason is numerical rather
than architectural.

### The obstacle

Every particle kernel wants the same two quantities: the index `k` of the
nearest knot, and the offset `δ = x − knot` from it. Deriving them from an
absolute coordinate means subtracting two nearby numbers, and in floating point
that **cancels the leading digits**: the result keeps only the resolution the
original magnitude allowed.

At 78 a₀ the `Float32` ULP is `7.6e-6`. A smoothing-table column is `1.42e-3`
wide. So the error is **0.54 % of a column** — and 0.05 % of particles land on
the wrong side of a boundary and get the wrong Gaussian sample. That is a wrong
*discrete choice*, not a rounding error that averages out.

![Cancellation, and the form that avoids it](assets/diagrams/packed-positions.svg)

For a long time the conclusion drawn from this was "the packing must run on the
host, in `Float64`". It is the wrong conclusion. The obstacle is not the width
of the type; it is **forming the difference at all**.

### The fix: hold `(k, δ)`, never re-derive it

[`PackedPositions`](@ref) stores the pair and rebuilds the absolute triple on
access:

```
         x = x0 + (knode − 1)·h + delta
             └──────┬──────┘   └──┬──┘
                exact, Int32      bounded by h/2
```

`δ` is small, so `Float32` resolves it to `6e-8` — a hundred times finer than
the same type on an absolute coordinate. And because the type is an
`AbstractVector{NTuple{3,T}}` that reconstructs on `getindex`, every existing
reader of `cloud.positions` keeps working untouched.

!!! note "`knode` is not clamped"
    A particle that has left the fine mesh keeps a *virtual* cell index —
    negative, or past the last knot — so the pair stays an exact description of
    where it is, with `δ` still small. Clamping belongs to the kernels that
    index a stencil, not to the representation. [`_cell_key`](@ref) is where it
    happens.

### What it buys the integrator

The scheme is a position Verlet, and **both of its observables are differences
of positions**:

```
  q(t+dt) = 2q(t) − q(t−dt) + dt²·F/M        ← a difference
  p       = M·(q(t+dt) − q(t−dt))/2dt        ← a difference, and the energy is p²
```

In the packed form the integer part `k(t+dt) = 2k(t) − k(t−dt)` is **exact**,
and every rounding that remains falls on `δ`. Measured over 101 steps on the
production cloud, against the same integrator on absolute `Float32`
coordinates: `2.55e-4` a₀ of drift against `1.03e-2`, a factor **40** — and the
gap widens with the step count.

### And what it buys the step

The packing stage does not move to the device. It **ceases to exist**: the cloud
is already in the form the kernels consume, held in the accelerator's own
buffers, which the kernels already read. Three things go with it — the packing
itself, the recomputation of the cell key by the sort, and the per-step
conversion of 8×10⁷ force triples to host `Float64`.

## The sort

Deposition is a *scatter*: each particle writes into 8³ = 512 grid points, and
neighbours write into the same ones. Ordering the particles by cell turns that
into one thread group per occupied cell, each thread owning one stencil point,
walking the cell's particles in a register and performing **one** atomic at the
end. It divides the atomics by the particles per cell — several hundred, at
production scale.

The sort is a counting sort, `O(N)` and insensitive to the starting order. The
thesis used PSRS instead, and for a reason that no longer applies: on a
distributed machine, sorting particles is a *communication* problem and regular
sampling balances the exchange. In shared memory there is nothing to balance.

### Why the placement is in two stages

The counting sort's last pass writes each particle's index to its final slot —
eighty million scattered 4-byte writes into a 305 MB array. That is the whole
cost of the sort, and it is not the atomics: separated by ablation, the atomics
are 16.9 ms and the scattered writes 107.2.

The cure is locality, and the lever is enormous. Confining the same writes to a
window of destinations:

| window | 0.25 MB | 1 MB | 4 MB | 64 MB | 305 MB | sequential |
|---|---:|---:|---:|---:|---:|---:|
| ms | 5.7 | 6.0 | 15.7 | 62.0 | 106.2 | **1.84** |

So the placement is split. [`_place_coarse_kernel!`](@ref) first bins the
particles into buckets of `2^shift` neighbouring cells — few enough destinations
that each bucket's write front advances nearly sequentially — and
[`_place_fine_kernel!`](@ref) then performs the exact sort **walking the order
the first stage produced**, so neighbouring work-items carry particles of
neighbouring cells and their destinations share pages.

![Scatter, and the locality two stages buy](assets/diagrams/two-stage-sort.svg)

The first stage does not sort anything anyone wants; it only buys the second one
its locality.

Both extremes of that choice are bad, and symmetrically so:

| buckets | 1 | 334 | 1 336 | **2 672** | 10 685 | 1 367 631 |
|---|---:|---:|---:|---:|---:|---:|
| ms | 122.0 | 63.0 | 26.4 | **17.5** | 29.3 | 118.1 |

One bucket is pure atomic contention; one cell per bucket is pure scatter. The
floor sits at a few tens of thousands of particles per bucket, which is what
[`sort_shift`](@ref) targets. The `perm` the two stages produce is identical to
the host sort's, cell for cell.

!!! warning "Two traps in the counting sort, both paid for"
    The counters **must be reset** — a repeated call otherwise accumulates them
    and the placement writes out of bounds. And the totals must go into their
    **own** buffer: writing them into `partial[1]` destroys the first chunk's
    counters, which the offset pass still needs. The price of forgetting is a
    segmentation fault.

## Precision, stated plainly

The CPU `Float64` path is the reference, and the only one comparable to the
Fortran oracle at `1e-13`. Apple GPUs have no double precision, so the device
path works in `Float32` and is validated against the reference rather than
against the oracle.

| Quantity | Discrepancy, device vs reference |
|---|---|
| forces | `8.3e-06` in norm |
| density | `6.3e-07` |
| projectile energy loss over 20 steps | `3.9e-07` |

All far below the physical dispersion, which is 2 to 4 %. `precision = Float64`
on a backend that supports it makes the whole chain bit-comparable with the host
again, and the test suite uses exactly that to pin the device path down.

## Reading the code

| File | Lines | What lives there |
|---|---:|---|
| `kernels.jl` | 946 | every portable kernel: deposition, field, packing, sort, Verlet |
| `accelerator.jl` | 632 | [`DeviceAccelerator`](@ref), its buffers, the step's device half |
| `devicemesh.jl` | 178 | [`DeviceMesh`](@ref), the mirror, and the chain's device methods |
| `gpu.jl` | 62 | the [`ForceAccelerator`](@ref) interface, and nothing else |
| `ext/VlasovMetalExt.jl` | 56 | where the buffers live, and the Metal entry point |

A good order to read them in: `gpu.jl` for the interface, `devicemesh.jl` to see
the split-in-two idea at its clearest, then `accelerator.jl` for the step, and
`kernels.jl` last — it is the longest, but by then every one of its arguments
has a home.
