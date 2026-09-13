```@meta
CurrentModule = Vlasov
```

# The original code

Forty-three Fortran versions survive in the thesis archive, across parallel
directories, with no version control and — in places — no reliable timestamps.
Knowing *which* of them the port targets is not archaeology for its own sake: the
versions differ on physics, and porting the wrong one would reproduce the wrong
results.

The full reconstruction is in
[`docs/chronologie-versions-fortran.md`](https://github.com/laurentplagne/Vlasov.jl/blob/master/docs/chronologie-versions-fortran.md).

## Timeline

```
1996-07 ─ 1996-10   setting up, single precision
1996-10             double precision
1996-12 ─ 1997-02   flux, multipoles
        1997-03     ├─ "3 grids" branch (lu/)                 ✗ abandoned
1997-05 ─ 1997-06   ★ the version ported to Julia  (1997-06-06)
        1997-06     ├─ T3E / PGI-HPF branch (solver only)     ✗ partial
        1997-08     ├─ sequential sort branch                 ✗ abandoned
1997-09 ─ 1997-11   self-energy, exchange-correlation, radial potential
1998-01-05          ★ the port's target — last version that still collides
1998-01-07          last complete sequential version (no projectile)
1998                ★★ vlas.hpf — HPF + MPI + two-level parallel sort, 16 PEs
1998-12-18          thesis defence
```

`.hpf` is High Performance Fortran. The parallel line is not a side experiment:
the thesis's Xe²⁵⁺ runs were produced by it, on 16 processing elements of a
Cray T3E.

## Which version is the target

The most recent by date is not the right one. `lucifer/initial/vlas.f`
(1998-01-07) **does not call `initpro`**: its main program has no projectile at
all. It is a relaxation experiment — hence the `initial/` directory, a
`qpold = 0.0` after initialisation, a friction term `kconv = -1e-5` added to the
forces, and a Thomas–Fermi kinetic term added to `pspech`.

**The target is `lucifer/vlas.f` (1998-01-05)**: the last version of the main
line that still performs a collision (`initpro`, `incproj`, `force2g`,
`enerele2g`).

## What separates the ported version from the target

61 routines identical, 15 modified, 16 added, **none removed**.

### Changes physics

| What | Effect |
|---|---|
| `makeinit` calls **`initialise4`** instead of `initialise` | sampling by **rejection in 6D phase space**: draw `r = rmax·x₁^⅓`, `p = pmax·x₄^⅓`, accept if `p²/2 + V(r) < E_F`. The exact Thomas–Fermi distribution, where the old one inverted a tabulated radial profile |
| `griech`: `nbprem` 10 → 2 | changes the sampling grid |
| `initialise`: `integer rmax` → `real*8 rmax` | fixes anomaly 4 |
| `pspech2`: `rr` moved out of the `if (rho > 1e-7)` | previously `rr` kept the previous point's value where the density was negligible — anomaly 8 |

A new input parameter **`rcmax`** appears at the end of `vlas.inp` and replaces
a hard-coded `100.d0`: the radius beyond which an electron counts as having left
the cluster. It is [`SimulationParameters`](@ref)'s `rcmax`, whose default
reproduces the earlier behaviour.

!!! note "One change that changes nothing"
    The 1998 version replaces `sqrt`/`cos`/`sin` with `dsqrt`/`dcos`/`dsin`.
    This was initially recorded here as a numerical fix. It is not: `SQRT` is
    generic in F77 and identical on `real*8` — verified with gfortran. Style
    only.

### Diagnostics only

`multrcmax`, `mkpotrad`, `mkrhoradx`, `mkdensene`, `distene`, `denseta`,
`sortietest`, and the `litpotexa` + `enertot2gix` pair that replays the energy
budget against an external radial potential. None of them is ported; none
changes a trajectory.

!!! warning "`multrcmax` is not what its name suggests"
    Despite the name it has nothing to do with the projectile's cutoff radius:
    it counts electrons in spheres from 50 to 100 a₀ and writes `rcm.dat` — an
    evaporation diagnostic.

### Dead code that matters

`pspech3` is called from nowhere. Yet it computes the exchange-correlation
**energy density** (Dirac with its ¾ factor, and the full Gunnarsson–Lundqvist
expression) rather than the *potential*, and corrects the Hartree double
counting by `csol ← ½csol + echsol`.

That is exactly the correction anomaly 8 calls for. The author had written it
and never wired it in. Porting it faithfully means leaving it dead.

## The `pot.dat` lock

`initialise4` reads a file `pot.dat` — header `nbgrid`, then `rmax pmax Ef`,
then `(r, ·, V(r))` — that **no version of `vlas.f` writes**. It is produced by
`mkpotradx` during a previous run: a self-consistent bootstrapping loop. **No
copy survived in the archive.**

`ref/fortran98/pot.dat` is therefore a **reconstruction**, not the original. It
is exact for what the file is used for: the rejection test `p²/2 + V(r) < E_F`
is equivalent to `p < p_F(r)`, so setting `V(r) = E_F − p_F(r)²/2` with
`p_F = (3π²ρ)^⅓` reproduces the intended Thomas–Fermi distribution, and `E_F`
cancels. The density comes from `rhoinit.dat`, the equilibrium profile the older
initialisation already used.

!!! warning "A standing caveat"
    Everything that goes through `initialise4` carries this assumption. It is
    recorded here so that it is never silently forgotten.

## Where the production runs are

`arkonnen/vlasov/vlas.inp`, sitting next to the ported source, is the **proton
production file**. It differs from the test input shipped in `ref/fortran/` by
two lines only: **800 000 pseudo-particles** instead of 20 000, and **4000
steps** instead of 2. Grid, domains, cutoff, energy and time step are identical.

That single fact settled a question the figures could not: the thesis's curves
were computed at 800 000 particles, and at 20 000 the local `dE/dx` estimator
does not converge. See [Validation](@ref).

`arkonnen/launch/` holds a twelve-entry sweep in the 71-line format of the
parallel version: projectile mass 236 864 a.u. (≈ Xe), charge 25, 1 310 720
particles, grid 32/63, domains 120/300, `cutoff = 5.0`, `rcmax = 45`, impact
parameters from 30 to 80 a₀ — the Xe²⁵⁺ runs, on the HPF code.

`temp/…/lucifer/Eloss/` keeps the **archived outputs**: `Em1q1e002i000.dat`
reads as mass 1, charge 1, 2 keV, impact 0 — exactly the name our 1998 oracle
produces. Those archived trajectories are a far safer reference than reading a
printed figure, and they are versioned under `ref/these/`.

## Reproducing the oracle

```
cd ref/fortran   && make oracle                              # 1997-06-06, instrumented
cd ref/fortran98 && make && python3 make_pot.py && ./vlas98  # 1998-01-05
```

`modernize.patch` is twelve lines — what it takes to make 1996 Fortran compile
on gfortran, including anomalies 4 and 5, without which it does not build at
all. `instrument.patch` adds the array dumps. Neither touches the numerics.
