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

### 2. Get the code, and the environment for your machine

```sh
git clone https://github.com/LaurentPlagne/Vlasov.jl.git
cd Vlasov.jl
```

Then **one** of these, whichever describes the machine:

```sh
julia --project=gpu  -e 'using Pkg; Pkg.instantiate()'    # Apple Silicon
julia --project=cuda -e 'using Pkg; Pkg.instantiate()'    # an NVIDIA card
julia --project=viz  -e 'using Pkg; Pkg.instantiate()'    # no GPU at all
```

A few minutes the first time, nothing afterwards. There are three because each
carries what its path needs and nothing more — `Metal` and `AppleAccelerate`,
or `CUDA`, or neither — so instantiating one does not drag in another vendor's
stack. The package itself (`--project=.`) carries no plotting stack at all: a
simulation package has no business depending on Makie.

### 3. Run it

One command. It computes the collision, prints what is happening, and writes the
film:

```sh
julia --project=gpu -t auto scripts/xenon.jl
```

`-t auto` hands Julia every core; without it the host-side loops run on one.

```
path: GPU (Metal) + AppleAccelerate, 10 threads | renderer: GLMakie, heatmap
Na196 + Xe25+, 500 keV, b = 45 a0 | 8.0e+06 particles, grid 130^3
built in 18.5 s

   step     t (fs)    x_ion (a0)   q(<8 a0)        out    ms/step
     50       0.60        -60.0      0.000          0      100.1
    100       1.21        -50.0      0.000          0       72.1
    ...
    350       4.23         -0.0      0.635      74061       94.1
    ...
    700       8.47         70.0      0.840     216829      141.5

700 steps in 69.5 s — 99.3 ms/step, plus 6.8 s reading 176 frames back
charge carried away by the ion: 0.840 electrons
wrote /Users/you/Vlasov.jl/xenon_data_cache.jls
wrote /Users/you/Vlasov.jl/xenon_snapshots.png
wrote /Users/you/Vlasov.jl/film_xenon.mp4 — 176 frames in 3.9 s (44.8 frames/s)
wrote /Users/you/Vlasov.jl/film_xenon.gif
```

Two minutes and twelve seconds from nothing, on a ten-core M1 Max — and the
last four lines say where the film is, which is the first thing one wants to
know.

**`q(<8 a0)` is the physics**: the charge sitting within 8 a₀ of the ion, in
electrons. It starts at zero, rises as the ion passes the cluster, peaks above
one, and settles at what the ion actually carries off. `out` counts the
particles that have left the fine grid — they are not lost, the coarse grid
carries them, and watching it climb is watching the ion strip the cluster.

The first line is worth reading. A missing package is otherwise silent, and the
same command then measures something else: it names the path taken **and** the
renderer chosen.

### Three options, and nothing else to learn

| | | default |
|---|---|---|
| `--particules=N` | pseudo-particles | 8×10⁶ with a GPU, 6×10⁵ without |
| `--nfine=n` | intervals of the fine grid — **even** | 64, i.e. a 130³ grid |
| `--cpu` / `--gpu` | force the path | take whatever is there |

The vendor is found on its own — Metal, CUDA, or the processor. The defaults are
chosen for the film: the same 130³ grid either way, so the picture is the same
and only its noise differs.

```sh
julia --project=viz -t auto scripts/xenon.jl --particules=100000 --nfine=28
```

is the quickest whole crossing — **a minute** on the ten cores above, film
included, and the capture still comes out at 1.0 electron. Coarser than the run
before it, and it tells the same story.

### What it writes

`film_xenon.mp4`, `film_xenon.gif` and `xenon_snapshots.png`, at the root of the
repository and all gitignored, plus `xenon_data_cache.jls` — the run itself. To
redraw it after changing a colour, without recomputing anything:

```sh
julia --project=viz -t auto scripts/render_xenon_glmakie.jl
```

Timed from nothing, no cache, cold start:

| | physics | film | whole command |
|---|---:|---:|---:|
| `--project=gpu`, M1 Max, 8×10⁶ | 69.5 s | **3.9 s** | **2 min 12** |
| `--project=cuda`, RTX 4070, 8×10⁶, headless → Cairo | 53.5 s | 18.8 s | 2 min 15 |
| `--project=viz`, 10 CPU cores, 6×10⁵ | 173.8 s | 3.9 s | 3 min 31 |
| `--project=viz`, 10 CPU cores, 10⁵ on 58³ | 24.1 s | 3.8 s | 1 min 02 |

The capture comes out at 0.840, 0.842 and 0.826 electron on the first three —
the last is a tenth of the particles, and its noise is the difference.

### Why the film is quick

Because of the **primitive**, not the backend — which is the opposite of what
one expects. Encoding the same 176 frames:

| | CairoMakie | GLMakie |
|---|---:|---:|
| `contourf`, 45 levels, on a 350×350 resample | 30.9 s | 31.3 s |
| `heatmap`, `interpolate = true`, raw 130³ slice | *refused* | **3.9 s** |

Swapping the backend under `contourf` buys nothing — the tessellation is CPU
work inside Makie either way. A `heatmap` is a texture, which is what a GPU can
actually draw, and its sampler does the bilinear smoothing that the 350×350
pre-pass was doing by hand. Cairo declines that combination outright: the
collocation points are not equally spaced, and it does not interpolate
non-regular grids.

So the script draws with **GLMakie and `heatmap`** wherever it can, and falls
back to Cairo where it cannot — `--rendu=cairo`, `--rendu=gl` and
`--rendu=aucun` force the matter.

> [!NOTE]
> **GLMakie needs a display** — it opens a window. On a headless box (a rented
> GPU, a cluster node over `ssh`) it would stop at `GLFW: X11: The DISPLAY
> environment variable is missing`, so the script looks for one first and takes
> Cairo when there is none. That is also why `cuda/` carries Cairo and not
> GLMakie: an `instantiate` printing GLFW errors looks like a broken setup when
> nothing is broken.

### Two vendors, the same physics

The same kernels on an M1 Max and on an RTX 4070, over the whole crossing at
8×10⁶ particles, both measured the same afternoon:

| | M1 Max, 32-core GPU | RTX 4070, 12 GB |
|---|---:|---:|
| charge carried away | 0.840 e | **0.842 e** |
| ms/step, averaged over the crossing | 99.3 | **76.4** |
| 700 steps | 69.5 s | **53.5 s** |
| reading 176 frames back for the film | 6.8 s | 15.2 s |
| the whole command | 2 min 12 | 2 min 15 |

Two independent vendors agreeing to the third digit is the strongest statement
this port can make about itself — and the third digit is where agreement stops
being meaningful anyway: two runs on the *same* card differ by as much, the
`Float32` atomics not committing in the same order twice, which moves the count
of particles outside the fine grid by 0.09 % from one run to the next. It was
first established on a rented RTX 5060 (Blackwell, 8 GB), which read 0.840 e.

The one line where the two machines genuinely differ is the **readback**: the
film needs 176 density slices and 176 looks at the cloud, which on unified
memory is not a copy at all and over PCIe is 15 seconds. The step itself, where
nothing crosses, is where the discrete card wins.

> [!WARNING]
> **ROCm and oneAPI have still never been run**, and CUDA only on two cards.
> Getting there took five bugs that Apple's unified memory had been hiding — a
> data file read from a path that only exists after the Fortran is built; a GPU
> probe that mistook *loading* Metal for *having* Metal; and three places where
> the host half of a buffer was read, or written, as if it were the device's.
> The worst of them silently froze the cloud: the capture curve read 0.000 for
> an entire crossing. Please report the next one.

### The other environments

| | carries | for |
|---|---|---|
| `--project=gpu` | Metal, AppleAccelerate, CairoMakie, GLMakie | Apple Silicon |
| `--project=cuda` | CUDA, CairoMakie | an NVIDIA card, headless included |
| `--project=viz` | GLMakie | no GPU, and redrawing |
| `--project=.` | the package alone | the test suite, and using it as a library |
| `--project=docs` | Documenter, CairoMakie | building the site |

`gpu/` is **the Apple one** — the name predates the port being portable.

---

## How long, and how big

Measured with the command above on one machine — **Apple M1 Max, 32-core GPU,
64 GB unified, 10 CPU threads** — so read the shape of it, not the absolute
numbers:

| path | particles | grid | ms/step | whole crossing |
|---|---:|---|---:|---:|
| CPU, 10 threads | 5×10⁵ | 90³ | 160 | **112 s** |
| GPU | 2×10⁶ | 90³ | 37 | ≈ 30 s |
| **GPU** | **8×10⁶** | **130³** | **99** | **70 s** |
| GPU | 3.2×10⁷ | 130³ | 240 | ≈ 3 min |
| GPU | 8×10⁷ | 222³ | 588 | ≈ 8 min |

The two bold rows are complete 700-step runs; the others are 200-step
measurements, and the crossing takes more than the multiplication suggests —
the step gets slower as the cloud spreads over more cells, 100 ms at the start
against 141 at the end of the bold row. None of these includes the film, which
is a fixed four seconds.

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

Checked on a real discrete card, which until then the table had only reasoned
about: 3×10⁷ particles on 130³ asks 3.78 GiB by the formula and **took 3.91** on
an RTX 5060 — 3.4 % over, the CUDA allocator rounding and its context — leaving
3.5 GiB of the 8 free, and a step ran.

A gigabyte is left to the driver and the runtime. **8 million is the safe first
number**: one gigabyte of device memory, a minute and a half of wall clock, and
it fits everywhere. The Apple row is bounded by the *construction* rather than
by the run, which peaks 48 bytes a particle higher — that, and the rest of the
accounting, is in [the device path](docs/src/device.md).

> [!NOTE]
> Times on cards other than these two are **not measured**. The step is close to
> linear in the particle count, and the kernels are bound by memory bandwidth
> and by atomics rather than by arithmetic, so a card's bandwidth is the first
> thing to scale by — and not its FLOP rating, which the RTX 4070 shows: it
> beats an M1 Max by ×4.9 on the Poisson solve and loses to it on the sort.

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
