```@meta
CurrentModule = Vlasov
```

# Performance

A time step at production scale — Na₁₀₀₀, 800 000 pseudo-particles, grid
`nfine = 44` — went from **307.9 ms to 77.9 ms**, a factor **3.95**. This page
says how, and what had to be unlearned along the way.

The full measurement log is in
[`docs/gpu.md`](https://github.com/laurentplagne/Vlasov.jl/blob/master/docs/gpu.md).

## Measuring first

Four disciplines, each paid for by a wrong conclusion.

**Profile in sequence, not in loops.** Timing each stage inside its own
repetition leaves that stage's data hot, while the real sequence overflows the
cache every turn. An early profile built that way lost **42 %** of the step:
the parts summed to 238 ms out of 412. Measuring each stage *in its place* in
the sequence closes the sum at 99.8 %. That is what `scripts/profil_pas.jl`
does.

**Warm up before timing.** The first GPU call pays the Metal kernel
compilation. Without a warm-up the same measurement read 614 ms/step — a GPU
*slower* than the CPU.

**Validate the state before benchmarking.** A bench that reuses one `Simulation`
ages it. After a LAPACK mishap, 98 % of the particles sat outside the fine grid;
they took the much cheaper coarse path, and the CPU/GPU ranking came out
**inverted**. [`forces!`](@ref) returns the number of out-of-grid particles: 635
out of 800 000 is healthy, 784 579 is not.

**Compare A and B alternately, in the same process.** Machine load varies. The
BLAS switch is reversible, so the whole comparison table below was taken by
interleaving configurations inside one process rather than comparing the moods
of two.

## Where the time goes

Na₁₀₀₀, 800 000 particles, one step. The last column says how each stage
scales — with the particle count `N`, or with the grid.

| Stage | CPU only | with forces on GPU | scales as |
|---|---:|---:|---|
| **interaction energy (Hartree)** | 77.6 | 76.9 | N |
| **interaction energy (total)** | 71.8 | 69.5 | N |
| forces on the cloud | 127.8 | **29.9** | N |
| smoothed deposit (fine) | 68.6 | 67.0 | N |
| **Poisson (GEMM + boundaries)** | **42.6** | 30.9 | grid |
| deposit (coarse) | 15.5 | 28.0 | N |
| projectile forces | 14.8 | 14.9 | N |
| mean field (XC + jellium) | 12.6 | 13.2 | grid |
| spline coefficients | 3.5 | 2.8 | grid |
| Verlet | 2.7 | 2.5 | N |
| **total** | **438.6** | **336.6** | |

Two lessons, both of which reversed the plan.

**The tensor solver was the wrong target.** The GEMMs are **10 %** of a step.
Making them free would cap at ×1.11 — Amdahl. Yet that was the starting plan,
because on Na₁₉₆ at 20 000 particles the grid did dominate: grid cost grows as
`n⁴`, particle cost as `N`, and at production scale `N` wins.

**The energy budget cost more than the forces.** Its two calls together are
149 ms — 34 % of the CPU step, 43 % once the forces moved to the GPU.

## What was done

### Apple Accelerate — one line, ×1.31

```julia
using AppleAccelerate
AppleAccelerate.load_accelerate()
```

Not only on the GEMMs: the particle loops themselves gain 15–25 %, because
OpenBLAS's thread pool stops competing with them for the cores.

!!! danger "Never `BLAS.lbt_forward(libacc)` raw"
    That binds Accelerate's *old* LAPACK. `inv` then returns garbage —
    `InexactError: Int64(1.0e-323)` — and the cluster explodes to a median
    radius of 1475 a₀. Use `AppleAccelerate.load_accelerate()`, which forwards
    with `suffix_hint = "\$NEWLAPACK\$ILP64"`.

### A periodic energy budget

The budget **observes**; it feeds back into nothing. The Fortran computed
`enertot2g` one step in ten, and so can we:

```julia
run!(sim; nsteps, energy_every = 10)     # the Fortran's choice
step!(sim; energy = false)               # a single step without a budget
```

This is a tested property, not an assumption: twelve steps with
`energy_every = 1` and with `energy_every = 4` leave **rigorously equal**
positions.

### The three levers, composed

Three interleaved rounds in the same process, 800 000 particles, minimum of
three measurements:

| Configuration | ms/step | speed-up |
|---|---:|---:|
| OpenBLAS, budget every step, CPU | 307.9 | — |
| OpenBLAS + GPU | 262.4 | ×1.17 |
| Accelerate alone | 234.8 | ×1.31 |
| OpenBLAS + budget 1/10 | 234.1 | ×1.32 |
| Accelerate + GPU | 200.9 | ×1.53 |
| OpenBLAS + GPU + budget 1/10 | 188.3 | ×1.64 |
| Accelerate + budget 1/10 | 181.3 | ×1.70 |
| **Accelerate + GPU + budget 1/10** | **140.2** | **×2.20** |

They compose almost multiplicatively, and the single best one costs one line.

### Four local wins

| Change | Gain | Note |
|---|---|---|
| `poisson_rhs!` fused into one threaded pass | **×16** (4.63 → 0.29 ms) | bit-for-bit identical result |
| `LocateTable` replacing bisection | **×14.6** on `locate` | coarse deposit 13.9 → 6.74 ms |
| threading the mean-field loop | **×5** | see the warning below |
| particles sorted by cell before depositing | ×1.34 on the CPU deposit | the thesis's sort, for the 1997 reason |

!!! warning "A performance decision is only valid where it was measured"
    A comment in the original said not to parallelise the mean field. That was
    true under OpenBLAS — and false under Accelerate, where it is worth ×5. Any
    such note has to carry the environment it was measured in.

## The GPU port

The target is the **particle loops**, 79 % of a step at production scale — not
the tensor solver.

Nothing in `src/` depends on a backend: `gpu.jl` declares
[`ForceAccelerator`](@ref), and `ext/VlasovMetalExt.jl` implements it, loaded
only if the user loads `Metal`.

### The constraint: no double precision

Apple GPUs have no `double`. The `Float64` CPU path therefore stays the
reference — the only one comparable to the Fortran oracle at `1e-13` — and the
GPU path is validated against it:

| Quantity | Discrepancy GPU vs CPU |
|---|---|
| forces | `8.3e-06` in norm |
| density | `6.3e-07` |
| projectile energy loss over 20 steps | `3.9e-07` |

Far below the physical dispersion (2 to 4 %), but no longer the oracle.

!!! note "Where `Float32` actually bites"
    Computing the smoothing-table **column indices** on the GPU tips 0.05 % of
    them onto their neighbour, taking the density discrepancy from `1.5e-07` to
    `9.0e-05`. The cause is structural: the `Float32` ULP at 78 a₀ is `7.6e-06`,
    which is 0.22 % of a column's width. One particle in five hundred sits less
    than one ULP from a boundary and gets the wrong Gaussian sample — a wrong
    *discrete choice*, not a rounding error that averages out.

    The fix is to upload `(k, δ)` computed in `Float64` on the host rather than
    absolute positions. Accuracy, not speed, was the reason.

### Deposition: two routes, one of them useless

Deposition is a **scatter**: each particle writes into 8³ = 512 grid points, and
neighbouring particles write into the same ones.

* **Atomic** — one particle per thread, 512 atomic additions. Simple, no
  preparation, and **three times slower than the CPU**: 800 000 × 512 = 410
  million contended atomics.
* **Sorted** — particles ranged by cell, then one group of 512 threads per cell,
  **each thread owning one stencil point**. It walks every particle of the cell
  accumulating in a register and performs **one** atomic at the end. Reversing
  the loops divides the atomics by the particles per cell — 108 here.

The sort is a parallel counting sort ([`CellSort`](@ref)) with everything
preallocated, restricted to the **occupied** cells: 7 413 of 91 125, because a
cluster of radius 40 fills little of a ±78 box. Sweeping all of them cost
1.86 ms; restricted, 0.13 ms plus 0.06 to build the list.

!!! warning "Two traps in the counting sort, both paid"
    The counters **must be reset** — a repeated call otherwise accumulates them
    and the sort produces out-of-bounds indices. And the totals must go into
    their **own buffer**: writing them into `partial[1]` destroys the first
    chunk's counters, which the offset pass still needs. The price of forgetting
    is a segmentation fault.

### Other GPU items

**The projectile fused into the force kernel** — computing it in the same sweep
saved 16 ms, since the particles are already resident.

**Shared storage** — `SharedStorage` instead of `PrivateStorage` for the
transfer buffers, ×6 on transfers on unified memory.

## Where it stands

| Configuration | ms/step |
|---|---:|
| CPU, budget every step | 225.2 |
| GPU, budget every step | 148.6 |
| CPU, budget one step in ten | 161.8 |
| **GPU, budget one step in ten** | **77.9** |

From the starting point — OpenBLAS, all CPU, budget every step, 307.9 ms —
**×3.95**.

Profile of that optimal configuration (step = 133.3 ms at the time of
measurement):

| Stage | ms | % |
|---|---:|---:|
| **smoothed deposit (fine)** | **39.1** | **29.4 %** |
| forces (GPU) | 28.4 | 21.3 % |
| `poisson!` | 21.5 | 16.1 % |
| projectile forces | 14.3 | 10.7 % |
| deposit (coarse) | 13.2 | 9.9 % |
| mean field | 11.2 | 8.4 % |
| Verlet | 2.8 | 2.1 % |
| spline coefficients | 2.5 | 1.9 % |

⚠️ **This ranking depends on the configuration**, and quoting it without saying
so misleads. On the **CPU** path (Accelerate, no GPU) the forces still lead by a
wide margin — 66.1 ms against 38.0 for the deposit. It is the GPU port that
brings them to 28.4 and puts deposition in front.

## What remains

1. **The force kernel (≈13 ms)** — a 10³ contraction bound by memory. The
   particles are **already sorted** for the deposit; having them read `csol` in
   that order would give the same locality that is worth ×1.34 on the CPU. The
   sort is there; it only needs using. This is the most promising avenue.
2. Writing straight into the shared buffers (`unsafe_wrap`) — a thin gain, but
   it removes the host buffers and half the transfer code.
3. The coarse deposit and the mean field, still on the CPU.
4. A **component-wise particle layout**. `ParticleCloud` stores positions as
   `Vector{NTuple{3,T}}`, which must be repacked into a `3×N` matrix for every
   GPU call. Storing by component would remove the repacking *and* help
   vectorise the CPU path.

Not on the list: **GEMMs on the GPU**. Accelerate already runs them in `Float64`
at 400–470 GFLOPS, which the GPU cannot do at all, and they are 1.7 ms per
solve.

## Reproducing the measurements

```
julia --project=. -t auto scripts/profil_pas.jl     # stage-by-stage, in sequence
julia --project=gpu -t auto scripts/bench_gpu.jl    # CPU vs GPU: speed and accuracy
julia --project=gpu -t auto scripts/depot_gpu.jl    # the two deposition routes
```
