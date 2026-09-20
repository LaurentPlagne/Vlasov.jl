```@meta
CurrentModule = Vlasov
```

# Vlasov.jl

Semi-classical dynamics of the valence electrons of a metal cluster, and the
energy an ion loses crossing one.

This package is an idiomatic Julia port of the Fortran 77 code written for
L. Plagne's PhD thesis, prepared at **CEA-Grenoble** under the supervision of
**C. Guet** (1996–1998). The original solved the Vlasov–Poisson
system for a sodium cluster Na\_N by a pseudo-particle method, on a cubic Hermite
spline collocation grid, with a tensor-product Poisson solver. The port
reproduces it — and, on one point, corrects it.

## What it computes

```@raw html
<div style="overflow-x:auto">
```

| | |
|---|---|
| **System** | Na\_N cluster: `N` valence electrons in a jellium ionic background |
| **Model** | Vlasov equation (semi-classical limit of time-dependent LDA), self-consistent mean field |
| **Method** | Pseudo-particles pushed by a field read off a cubic-spline grid |
| **Observable** | Energy conservation on an isolated cluster; `dE/dx` for an ion crossing it |

```@raw html
</div>
```

## Where it stands

* The **stopping-power curve of the thesis is reproduced** (Na₁₀₀₀, σ\_ion = 1) to
  within the dispersion of the published data itself, and so is the **wake**
  of figure 5.2 — see [Validation](@ref).
* The port is checked against a **reconstructed Fortran oracle** at `1e-13` on
  every stage, and bit-for-bit on the random generator.
* A time step at production scale went from 307.9 ms to **77.9 ms** — see
  [Performance](@ref).
* One discrepancy found in the original code changes a physical conclusion: the
  Fortran does not apply the force its own thesis states. See
  [The softening: ball or Gaussian](@ref).

## Quick start

The shortest path to a running collision needs no Julia at all — one command,
and it prints the charge a xenon ion carries off a sodium cluster as it passes:

```
julia --project=.   -t auto scripts/xenon.jl        # 5×10⁵ particles, ~2 min
julia --project=gpu -t auto scripts/xenon.jl        # 8×10⁶ on the GPU, ~1.5 min
```

The README walks through it from installing Julia. From inside the language:

```julia
using Vlasov

# Initial state: the archived equilibrium radial density of Na₁₀₀₀
grid, ρ = read_radial_density("ref/these/rhorad.Na1000.dat")

params = SimulationParameters(nfine = 44, ninner = 22, nouter = 22,
                              rcluster = 78.0, rbox = 235.0,
                              nions = 1000.0, nelectrons = 1000.0,
                              nparticles = 200_000, nsteps = 0, dt = 1.0)

# A 4 keV proton, Gaussian softening of width σ_ion = 1
proj = Projectile(mass = 1836.154, charge = 1.0, energy = 4000 / HARTREE_TO_EV,
                  impact = 0.0, x0 = -65.0, dt = 1.0,
                  softening = GaussianSoftening(1.0))

sim = Simulation(params, PotentialProfile(grid, ρ); projectile = proj)

# The energy budget is a diagnostic: computing it one step in ten is worth ×1.6
history = run!(sim; nsteps = 400, energy_every = 10)

energy_loss(sim.projectile) * HARTREE_TO_EV     # eV lost by the proton
```

Run it with threads — every particle loop is parallel:

```
julia --project=. -t auto scripts/figure53.jl
```

## Reading order

| Page | What it answers |
|---|---|
| [Principles](@ref) | What equations are being solved, and why this model |
| [Numerics](@ref) | How they are discretised: splines, collocation, Poisson, Verlet |
| [Architecture](@ref) | How the code is laid out, and what one time step does |
| [Validation](@ref) | How we know it is right, and where the original was not |
| [Performance](@ref) | Where the time goes, and what was done about it |
| [The original code](@ref) | The 43 Fortran versions, and which one is the target |
| [API reference](@ref) | Every exported name |

## Conventions

Everything is in **atomic units** (ħ = mₑ = e = 1): lengths in bohr `a₀`,
energies in hartree. [`HARTREE_TO_EV`](@ref) converts for display —
the thesis plots `dE/dx` in eV/a₀.

Throughout the source, the original Fortran name is recalled next to its Julia
counterpart (`makerho`, `pspech2`, `enertot2g`, …). That is deliberate: it is
what makes the port auditable against the thesis.
