# Vlasov.jl

The electron dynamics of a sodium cluster struck by a fast ion, by the
pseudo-particle (Vlasov) method — a Julia port of the Fortran 77 code written
for L. Plagne's PhD thesis, prepared at **CEA-Grenoble** under the supervision
of **C. Guet** (1996–1998). The original still compiles and runs, and serves as
the oracle the port is kept honest against.

![Na₁₉₆ + Xe²⁵⁺, 500 keV, b = 45 a₀](docs/src/assets/film_xenon_80M.gif)

**Na₁₉₆ + Xe²⁵⁺, 500 keV, impact parameter 45 a₀, 80 million pseudo-particles.**
The ion grazes the cluster; its field tears an electron bridge out of the
valence cloud and carries part of it away. That is the collision the first run
below computes — the 1997 Springer study, at two hundred times the particle
count the machines of the day allowed.

---

## Running your first collision

### 1. Install Julia

This is a Julia package, so the language comes first. The official installer
takes one line and needs no privileges:

```sh
curl -fsSL https://install.julialang.org | sh          # macOS, Linux
winget install julia -s msstore                        # Windows
```

Close the terminal, open a new one, and check it answers:

```sh
julia --version        # developed and tested on 1.12 and 1.13
```

### 2. Get the code and its dependencies

```sh
git clone https://github.com/LaurentPlagne/Vlasov.jl.git
cd Vlasov.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

`instantiate` fetches the exact package versions this was tested with. A few
minutes the first time, nothing afterwards.

### 3. Run it on the processor

```sh
julia --project=. -t auto scripts/xenon.jl
```

`-t auto` hands Julia every core; without it the particle loops run on one. The
whole output of that command, on a ten-core M1 Max:

```
path: CPU, 10 threads
Na196 + Xe25+, 500 keV, b = 45 a0 | 5.0e+05 particles, grid 90^3
built in 3.3 s

   step     t (fs)    x_ion (a0)   q(<8 a0)    ms/step
    100       1.21        -50.0      0.000      177.2
    200       2.42        -30.0      0.089      176.7
    300       3.63        -10.0      0.546      171.0
    400       4.84         10.0      0.600      173.0
    500       6.05         30.0      1.010      185.9
    600       7.26         50.0      1.086      189.5
    700       8.47         70.0      0.920      188.9

700 steps in 126.2 s — 180.3 ms/step
charge carried away by the ion: 0.920 electrons
```

`q(<8 a0)` is the charge sitting within 8 a₀ of the ion, in electrons. It starts
at zero, rises as the ion passes the cluster, peaks above one, and settles at
what the ion actually takes with it. **That curve is the physics** — everything
else on this page is about computing it faster or more finely.

Four options, and nothing else to learn:

| | | default |
|---|---|---|
| `--particules=N` | pseudo-particles | 5×10⁵ on CPU, 8×10⁶ on GPU |
| `--nfine=n` | intervals of the fine grid — **even** | 44 on CPU, 64 on GPU |
| `--pas=N` | time steps; 700 is the whole crossing | 700 |
| `--tous=N` | print one line every N steps | 50 |

```sh
julia --project=. -t auto scripts/xenon.jl --particules=200000 --nfine=28
```

is the quickest whole crossing — **43 seconds** on the ten cores above, and the
capture still comes out at 1.0 electron. Coarser than the run before it, and it
tells the same story.

### 4. Run it on the GPU

The device path is portable by construction — the kernels are written once, in
`KernelAbstractions` — but only **Apple Metal** has hardware here to be tested
on. On a Mac:

```sh
julia --project=gpu -e 'using Pkg; Pkg.instantiate()'
julia --project=gpu -t auto scripts/xenon.jl
```

Sixteen times the particles, and it finishes sooner:

```
path: GPU (Metal) + AppleAccelerate, 10 threads
Na196 + Xe25+, 500 keV, b = 45 a0 | 8.0e+06 particles, grid 130^3
built in 13.3 s
    ...
    700       8.47         70.0      0.842      163.4

700 steps in 86.4 s — 123.5 ms/step
charge carried away by the ion: 0.842 electrons
```

The `gpu/` environment carries `Metal` and `AppleAccelerate`. It has no
manifest of its own, so that `instantiate` *resolves* the two packages rather
than replaying pinned versions — unlike the main project, whose `Manifest.toml`
is tracked.

The script names the path it took on its first line, and that line is worth
reading: a missing package is otherwise silent, and the same command then
measures something else.

On an NVIDIA card there is an environment ready, and a script that will pick it
up on its own:

```sh
julia --project=cuda -e 'using Pkg; Pkg.instantiate()'
julia --project=cuda -t auto scripts/xenon.jl
```

> [!WARNING]
> **CUDA, ROCm and oneAPI have never been run here** — there is no such card on
> the machine this was written on. Nothing in the kernels is Apple-specific and
> the accelerator takes any `KernelAbstractions` backend, so it is *expected* to
> work; expected is not measured. The first attempt on a Linux box with an
> NVIDIA card found three real bugs in ten seconds — a data file read from a
> path that only exists after the Fortran is built, a GPU probe that mistook
> *loading* Metal for *having* Metal, and a host-side diagnostic that on a
> discrete card would have read a stale copy of the cloud. All three are fixed
> above. Please report the next one.

### Making the film

The run above prints numbers; the film at the top of this page is a separate
script, because rendering needs a plotting stack the simulation itself does not:

```sh
julia --project=gpu -t auto scripts/film_xenon.jl        # 6×10⁵ particles
```

It writes `film_xenon.mp4`, `film_xenon.gif` and `xenon_snapshots.png` **at the
root of the repository** (all three are gitignored), and caches its run in
`xenon_data_cache.jls` — delete that file to recompute rather than redraw. On
the machine above: 31 s of physics, 31 s of drawing, 176 frames.

`scripts/render_xenon_glmakie.jl` redraws the cached run **six times faster**,
and the reason is worth knowing: it is not the backend, it is the primitive.

| encoding 176 frames | CairoMakie | GLMakie |
|---|---:|---:|
| `contourf`, 45 levels, on a 350×350 resample | 30.9 s | 31.3 s |
| `heatmap`, `interpolate = true`, raw 130³ slice | *refused* | **4.6 s** |

Swapping the backend under `contourf` buys nothing — the tessellation is CPU
work inside Makie either way. A `heatmap` is a texture, which is what a GPU can
actually draw, and its sampler does the smoothing the 350×350 pre-pass was
doing by hand. Cairo declines that combination outright: the collocation points
are not equally spaced, and it does not interpolate non-regular grids.

The film on this page is the same script at production scale,
`scripts/film_xenon_80M.jl`: 80 million particles, some eleven gigabytes of
device memory, and it copies its gif into `docs/src/assets/`.

---

## How long, and how big

Measured with the command above on one machine — **Apple M1 Max, 32-core GPU,
64 GB unified, 10 CPU threads** — so read the shape of it, not the absolute
numbers:

| path | particles | grid | ms/step | whole crossing |
|---|---:|---|---:|---:|
| CPU, 10 threads | 5×10⁵ | 90³ | 180 | **126 s** |
| GPU | 2×10⁶ | 90³ | 51 | ≈ 36 s |
| **GPU** | **8×10⁶** | **130³** | **124** | **86 s** |
| GPU | 3.2×10⁷ | 130³ | 466 | ≈ 5 min |
| GPU | 8×10⁷ | 222³ | 824 | ≈ 10 min |

The two bold rows are complete 700-step runs; the others are 200-step
measurements, and the crossing takes a little more than the multiplication
suggests — the step gets slower as the cloud spreads over more cells.

**What fits in a given graphics memory** follows from two numbers measured on
the live objects, one per pseudo-particle and one per grid point:

```
device memory ≈ 128 bytes × particles  +  91 bytes × (2·nfine + 2)³
```

| graphics memory | particles at 130³ | at 222³ |
|---|---:|---:|
| 4 GB | 22 M | 15 M |
| 8 GB | 53 M | 46 M |
| 16 GB | 115 M | 109 M |
| 24 GB | 178 M | 172 M |
| 64 GB unified (Apple) | ≈ 350 M | ≈ 340 M |

A gigabyte is left to the driver and the runtime. **8 million is the safe first
number**: one gigabyte of device memory, a minute and a half of wall clock, and
it fits everywhere. The Apple row is bounded by the *construction* rather than
by the run, which peaks 48 bytes a particle higher — that, and the rest of the
accounting, is in [the device path](docs/src/device.md).

> [!NOTE]
> Times on other cards are **not measured** — there is no such hardware here.
> The step is close to linear in the particle count, and the kernels are bound
> by memory bandwidth and by address translation rather than by arithmetic, so
> a card's bandwidth is the first thing to scale by.

### Choosing a grid

⚠️ **The grid comes in pairs.** The coarse mesh must carry exactly as many basis
functions as the fine one, so `--nfine` must be **even**, and the two coarse
parameters are derived from it rather than given:

| `--nfine` | grid | what it is for |
|---|---|---|
| 28 | 58³ | a quick look, a minute of laptop CPU |
| 44 | 90³ | the first run above |
| 64 | 130³ | the film at the top of this page |
| 110 | 222³ | production, 8×10⁷ particles |

An odd `nfine` leaves the two meshes one basis function apart, and the run stops
on `DimensionMismatch: ρ must cover the whole collocation grid`.

---

## Documentation

**[laurentplagne.github.io/Vlasov.jl](https://laurentplagne.github.io/Vlasov.jl/)**
— fourteen pages, and the figures on them are *computed* when the site is
built, not checked in.

| page | what it answers |
|---|---|
| [Principles](docs/src/principles.md) | what equations are being solved, and why this model |
| [Numerics](docs/src/numerics.md) | splines, collocation, the tensor Poisson solver, Verlet |
| [Architecture](docs/src/architecture.md) | how the code is laid out, and what one step does |
| [The device path](docs/src/device.md) | what runs on the GPU, what it costs in memory, and why the cloud changed shape |
| [Validation](docs/src/validation.md) | how we know it is right — and where the original was not |
| [Performance](docs/src/performance.md) | where the time goes, and what was done about it |
| [The original code](docs/src/history.md) | the 43 Fortran versions, and which one is the target |

Those links go to the Markdown sources, which GitHub renders directly; the
built site is prettier and has the cross-references. To build it yourself:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl        # a few minutes: it runs the physics
open docs/build/index.html
```

## What else is here

| | |
|---|---|
| `scripts/` | the thesis figures, the films, the profiles and the benchmarks |
| `ref/` | the original Fortran, its data, and the thesis's published curves |
| [`lpthese2.pdf`](lpthese2.pdf) | the thesis itself (CEA-Grenoble, 1998) — the physics this code is the instrument of |
| `test/` | 4824 tests — `julia --project=. -e 'include("test/runtests.jl")'` |

⚠️ The measurement log under `docs/*.md` — as opposed to `docs/src/` — is in
French and is **not** part of the site: it is the lab notebook, it keeps the raw
numbers and the reasoning behind every decision, and the site quotes from it
what matters.

## Licence

MIT — see [LICENSE.md](LICENSE.md).
