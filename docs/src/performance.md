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

**A profile made of stages you cut yourself only shows what you thought to
measure.** The ten stages below close at 100 % of the step and look complete.
The driver's own per-kernel enumeration — `Metal.@profile` — found, in the same
step, that the **GPU was idle 61 % of the time**, and that 49.6 ms went to eight
broadcast kernels no stage of mine accounted for separately. Neither is visible
to a sum of stage durations, because such a sum never asks what the GPU was
doing while the host worked. See [Profiling a kernel](@ref) below.

### Roofs of the machine, measured

Percentages of "peak" mean nothing against a datasheet. These are what the
hardware actually delivered on an M1 Max, and the denominators every ratio in
this page uses:

| | measured | datasheet |
|---|---:|---:|
| device stream (`c = a + b`, 256 MB arrays) | **336 GB/s** | 400 |
| host stream (threaded copy, 10 cores) | **171 GB/s** | — |
| scattered `Float32` atomics | **4.0 G/s** | — |
| square GEMM, 4096³ | **8849 GFLOP/s** | 10 400 |
| GEMM in the tensor solver's shape, 65536×256×256 | **2833 GFLOP/s** | — |

Two of these are results in themselves. The solver's rectangular shape costs
**×3.1** against a square GEMM on the same machine — and only 12 % of that is
the transposition, the rest is the shape. And the stream roof must be measured
on a **large** array: taken on a 258³ cube (68 MB) it reads 127 GB/s, low enough
that a kernel of this code appeared to exceed it by 17 %. A roof you go through
is not a roof.

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

1. A **component-wise particle layout**. `ParticleCloud` stores positions as
   `Vector{NTuple{3,T}}`; storing by component would help vectorise the host
   packing, which runs at 19 % of the machine's memory bandwidth.
2. The **energy budget**, still particle-by-particle on the host. It is a
   diagnostic taken one step in ten, and has never been profiled at 8×10⁷.

Not on this list any more: **overlapping the two depositions**, which looked
like it needed nothing but not draining the queue between them. Measured and
dropped — see "Dead ends, measured" below.

Not on the list: **GEMMs on the GPU**. Accelerate already runs them in `Float64`
at 400–470 GFLOPS, which the GPU cannot do at all, and they are 1.7 ms per
solve.

### Dead ends, measured

Each of these looks obviously worth doing, and each was tried and abandoned on
numbers. They are written down so that the two hours go unpaid a second time.

**Moving the sort to the device: ×1.17, not worth it.** The cell key is exactly
`knode`, which is already there, so the `Float64` that keeps the packing on the
host does not apply. A counting sort in three parts measures, at 8×10⁷:

| | ms |
|---|---:|
| histogram (80 M atomics into 2.1 M bins) | 14.7 |
| scan and compaction (host, 2.1 M cells) | 1.2 |
| **placement** | **115.3** |
| total | 131.2 |
| *host `cellsort!`* | *153.4* |

The placement is the whole of it, and it is **not** the atomics: those alone
cost 15.1 ms. It is the 80 million scattered 4-byte writes into a 320 MB array —
the same volume written sequentially costs **1.4 ms**, a factor of 84.

**Keeping the cloud physically sorted: net −234 ms.** If the particle arrays
were held in cell order, the placement's writes would be local. They are: the
scatter drops to 29.9 ms, and one step of drift leaves a median displacement of
2230 positions out of 8×10⁷. But permuting `positions`, `previous` and `forces`
costs **328 ms** even preallocated and threaded, against 85 ms saved on the
scatter and **9 ms** on the two big kernels — forces 480.0 → 471.5, deposit
203.7 → 203.3.

That 9 ms is the interesting part: **the sorted traversal had already bought the
locality.** Neighbouring work-items read neighbouring `perm` entries, so their
`delta` reads land in the same cache lines whether or not the array is
physically ordered.

**Tracking only the particles that changed cell: 38.3 % change every step.**
Measured over four consecutive steps (38.26 – 38.29 %), and it follows from the
physics rather than from the code: the mean displacement is 0.356 a₀ against a
cell of 1.219, so 0.29 of a cell per step, which over three dimensions gives a
crossing probability near 40 %. There is no temporal coherence to exploit at
this `dt`.

**The smoothing columns in closed form: 697.1 ms against 505.9.** The force
kernel's table reads were worth three quarters of it, so
[`smoothing_columns`](@ref) was written to compute those columns instead —
five `erf` and five `exp` per direction cover all ten basis functions and both
tables, and a micro-benchmark of that arithmetic alone measured 90 ms per step
against the 350 the reads cost. Grafted into the kernel, with the sixty values
staged in threadgroup memory, it measures **697.1 ms** — worse than the tables
it replaces, and more than three times the 208.5 ms that staging the
*tabulated* columns reaches. The micro-benchmark was right about the
arithmetic and silent about what it does to the contraction around it.

The function stays, tested and exact: it is the reference the tables are
checked against. It is simply not how this kernel should be fed.

**Confining the coarse deposition's scatter to a window: 154.8 ms against
89.7.** Once the counters had shown that kernel to be bound on address
translation, the obvious cure was to bound the working set — scatter inside a
window of consecutive particles and sweep the windows in order, so the pages in
flight are `W·48` bytes whatever the cloud. It is worse at every window size
tried: 154.8 ms for 4×10⁶ particles, 707 for 2.6×10⁵, 1009 for 6.5×10⁴, against
89.7 for the plain stride. Inside a window the same coarse cells come round
again and again, and the atomic collisions the scatter exists to break come
back with them. The stride *is* the compromise, and its only knob is its length.

**Overlapping the two depositions on separate Metal queues: no gain.** The fine
and coarse depositions write disjoint grids and looked like the one item on the
list that didn't depend on understanding a kernel — just stop draining the
queue between them. Metal.jl 1.11's `global_queue` is **task-local**, so two
`Threads.@spawn` around independent kernel launches do get two distinct
`MTLCommandQueue`s — confirmed with `Metal.@profile`, which counted
`[MTLDevice newCommandQueue]` twice. That was the mechanism this idea needed,
and it does not help: two launches of the same deposit kernel (80 M particles,
122 636 occupied cells) measured **634 → 624 → 593 ms** in three consecutive
A-B-A runs, no significant difference. The profile of the concurrent run says
why — the GPU was busy 614.7 ms of a 689.1 ms wall clock, and the two kernel
calls logged 306.89 ms ± 8.82 **each**, summing to the device-busy time almost
exactly: they ran back to back, not together. Two command queues do not buy
concurrent *compute* dispatch on this GPU — the M1 Max's single compute engine
serializes them regardless, and the deposit kernels are atomic-scatter bound in
the first place, so there was no spare throughput for a second one to use even
if it had overlapped.

## Profiling a kernel

Stage timing says *how long*; it never says *why*. Two levels go further, and
both work from the command line.

**Per-kernel, no Xcode.** The driver enumerates its own dispatches:

```julia
show(stdout, MIME("text/plain"), Metal.@profile onestep!())
```

The `show` is not optional outside a REPL — a bare expression prints nothing and
yields an empty table without saying so. And **MPS does not appear**: the GEMMs
submit their own command buffers, so Poisson and the spline coefficients are
missing from the table by construction, not by accident.

**Hardware counters.** `xctrace` records Apple's *Performance Limiters* set —
`Compute Occupancy`, `ALU Limiter`, `Buffer Read/Write Limiter`,
`Threadgroup/Imageblock Load/Store Limiter`, `GPU Last Level Cache Limiter`,
`MMU TLB Miss Rate`, and the read/write bandwidths.

This used to be an afternoon's work. It is now a minute, and the difference is
entirely in **what you record**: one kernel, on a fraction of its work, for a
tenth of a second.

### The recipe

**1. Shrink the work, not the problem.** Keep the production cloud and give the
kernel fewer cells — the regime is what must be preserved, not the size. At
8×10⁷ particles, 20 000 occupied cells out of 109 054 run at 3.93 µs per cell
against 4.22 for the whole grid: **93 % of the regime for 17 % of the time**.

!!! warning "Shrinking the grid instead does not work"
    Halving `nfine` doubles the cell width, the 10³ stencil then overruns the
    grid, and the forces come back `NaN`. The particles-per-cell count and the
    stencil geometry are the regime; the number of groups is not.

**2. Loop the kernel alone, for much longer than the recording.** Then every
sample belongs to it, with no attribution to do afterwards — and the window
cannot fall outside the run. A first attempt recorded a **GPU at rest**, because
a 31 s loop had ended before `xctrace` attached: the counters read zero at the
median with absurd values in the tail, which is what that failure looks like.

**3. Record 100 ms.** The counters are stable to 1 % — mean equals median — so
nothing is bought by recording longer:

| | 4 s | **0.1 s** |
|---|---:|---:|
| Buffer Read Limiter | 98.6 % | **99.0 %** |
| Last Level Cache Limiter | 93.1 % | **93.0 %** |
| ALU Limiter | 13.7 % | **14.0 %** |
| trace | 2.4 GB of XML | **248 MB** |
| export | minutes | 38 s |

```
xcrun xctrace record --template "Metal System Trace" \
     --instrument "Metal GPU Counters" --no-prompt \
     --output t.trace --attach <pid> --time-limit 100ms
scripts/compteurs_gpu.py t.trace
```

**4. Parse by position, in a stream.** The table's columns are `timestamp,
counter-id, value, accelerator-id, sample-index, ring-buffer-index`, and the
export **compresses by reference**: a value appears once as `id="N"` and every
repetition is `ref="N"`. Resolve each row's elements in order, keeping a
dictionary of definitions — matching on names instead gives plausible-looking
nonsense, such as occupancies above 100 %. And the counter is an integer id
whose name lives in a second table, `gpu-counter-info`.

`scripts/compteurs_gpu.py` does all of that, exports included — it was written
twice from this paragraph before being kept.

⚠️ Its medians are **rates over the recorded window**, not per unit of work.
Two runs of the same kernel at different speeds do not compare line by line: a
rate that holds while the kernel does twice the work means the absolute
activity doubled.

`--attach`, never `--launch`: recording from launch captures the 50 s of
construction, which at 8×10⁷ particles is 50 million samples and a **7.6 GB**
export, none of it about the kernel.

### What the counters say about the force kernel

Measured this way, on `_smoothed_field_kernel!` alone:

| | |
|---|---:|
| **Buffer Read Limiter** | **99 %** |
| **GPU Last Level Cache Limiter** | **93 %** |
| Buffer Load Utilization | 26 % |
| ALU Limiter | 14 % |
| Compute Occupancy | 12 % |
| Threadgroup Load Limiter | 7 % |
| GPU Read Bandwidth | **2.3 GB/s** |

The kernel is saturated on reads that **never leave the cache**: 2.3 GB/s to
external memory on a machine that sustains 336, and threadgroup memory — where
the 10³ tile lives — at 7 %. Whatever it is waiting for, it is a buffer load
served by the last level cache.

!!! note "Resolved: the loads were in the middle of the loop nest"
    "The compiler was already hoisting `cx`, so the loads are elsewhere" was
    the right conclusion, and *elsewhere* is the other two directions. The
    inner loop reads `ovl[ii, cx]` a thousand times per particle and gets it
    for free; the middle loop reads `ovl[jj, cy]` two hundred times and the
    outer one `ovl[kk, cz]` twenty, and **those** the compiler will not hoist —
    carrying them would cost twenty registers live across the whole
    contraction.

    Staging them in threadgroup memory, 40 values per particle, is the whole
    fix:

    | staged | ms |
    |---|---:|
    | nothing | 505.9 |
    | `x` only | 505.6 |
    | all three directions | 260.8 |
    | **`y` and `z`** | **208.5** |

    Bit for bit identical over the 13.6 million slots checked — the same
    values, read from somewhere else. The kernel loses 297 ms and the step
    goes from 1291.2 to 988.1 ms (A-B-A, the closing A at 1294.7), for a change
    that touches no arithmetic.

    Staging all three directions is worse than staging two: the `x` values then
    make a round trip through threadgroup memory that the register file was
    doing for free.

    Two lessons, both already on this page and both paid again: a counter names
    the limiter but not the line, and an ablation gives the gain but not the
    cause. What closed it was an A-B over *which* reads were staged.

### And about the coarse deposition: the MMU

The same recipe on `_deposit_cic_kernel!`, which had never been measured this
way, named a limiter nobody had proposed:

| | stride 7919 | **stride 509** |
|---|---:|---:|
| **MMU Limiter** | **70.7 %** | **61.5 %** |
| GPU Last Level Cache Limiter | 55.9 % | 52.0 % |
| Buffer Read Limiter | 40.7 % | 15.0 % |
| MMU TLB Miss Rate | 15.1 % | 11.9 % |
| Compute Occupancy | 26.7 % | 26.6 % |
| ALU Limiter | 7.4 % | 14.4 % |
| Buffer Write Limiter | 0.0 % | 0.0 % |
| GPU Read Bandwidth | 42.3 GB/s | 35.8 GB/s |

Not the atomics — the write limiter reads zero — but **address translation**.
The kernel walks the cloud by a stride coprime with the particle count, to keep
its eight atomics off the same coarse cell; at a stride of 7919 that is 380 KB
between neighbouring work-items, so each one reads its 48-byte particle from a
page of its own.

Shortening the stride to 509 measures **195.0 → 89.7 ms** at 8×10⁷ particles,
and the step 841.9 → 745.9 (A-B-A). The two columns above are the same window
of wall clock, and the right-hand one does 2.2× the work in it: per particle it
reads 2.6× fewer bytes. [`scatter_stride`](@ref) carries the whole curve, which
is flat-bottomed between 251 and 1009 and steep on both sides — too near and
the atomics collide, too far and the MMU does.

The kernel is still MMU-bound at 61.5 %, so there may be more; it would take a
different layout of the cloud, not another constant.

Ablation — remove a piece, measure again — gives the *gain* but never the
*cause*. It is what found that the projectile's tree reduction was worth 227 ms
of [`_smoothed_field_kernel!`](@ref); the conclusion that occupancy was
responsible was an inference from the fact that a smaller group won, not a
measurement. Counters name the limiter instead of guessing it.

## Reproducing the measurements

```
julia --project=. -t auto scripts/profil_pas.jl     # stage-by-stage, in sequence
julia --project=gpu -t auto scripts/bench_gpu.jl    # CPU vs GPU: speed and accuracy
julia --project=gpu -t auto scripts/depot_gpu.jl    # the two deposition routes
```
