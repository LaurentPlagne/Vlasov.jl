#!/usr/bin/env julia
"""
Produce the frames of a crossing movie: slices of the electron density.

    julia --project=gpu -t auto scripts/film_images.jl [--kev=4] [--images=400]
                                                       [--particules=800000] [--epaisseur=2]
                                                       [--finesse=3]

Run it under `--project=gpu` and it picks up **AppleAccelerate** (BLAS) and
**Metal** (the smoothed field and the forces) on its own; under `--project=.`
neither is present and the plain CPU path runs. The banner says which was used.

This is the view of the thesis's `snappro` figures — a cut of the density
through the plane the projectile travels in. What it shows is the deformation of
the cloud as the ion passes, and its **asymmetry**: the wake trails behind the
ion, the more so the faster it goes.

⚠️ `--epaisseur` is not cosmetic, and neither of its two obvious settings is
right: see [`cut_z0`](@ref), which carries the measured table. `--finesse`
decides how finely the density spline is *evaluated* for display, which is what
keeps the cluster's surface from coming out as a staircase of blocks.

Frames are not taken at every step — one step lasts 0.4 a.u. of time and a few
hundred are needed for the crossing, whereas a twenty-second movie calls for
five hundred. The sampling stride is computed to land right.

Two fields are kept per frame:

  * `ρ` — the density itself, as the thesis plots it;
  * `δρ = ρ − ρ₀` — the departure from the initial state, which **shows the
    deformation far better**: it amounts to a few per cent of a background a
    thousand times larger, and drowns on an absolute scale.

Output: `film.jls` (Serialization, stdlib — no dependency added), read by
[`film.jl`](film.jl).
"""

using Vlasov
using Printf
using Serialization
# ⚠️ Qualified below: `next!` is exported by BOTH Vlasov (the `Ran2`
# generator) and ProgressMeter, so the bare name resolves to neither. Same
# trap as `scatter!` in `film.jl`.
using ProgressMeter

# Optional accelerators. They live in the `gpu/` environment; under
# `--project=.` neither is present and the plain CPU path runs instead. Loading
# `AppleAccelerate` is enough — its `__init__` forwards BLAS and LAPACK
# correctly, which `BLAS.lbt_forward` on its own does not.
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end
# ⚠️ `functional()`, not merely `using`: `using Metal` SUCCEEDS on a machine
# with no Apple GPU — it only logs an error — and the script then took the
# Metal path on a Linux box with an NVIDIA card. Reported from one.
const METAL = try; @eval using Metal; @eval Metal.functional(); catch; false; end

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV

function parse_args(argv)
    # ⚠️ `epaisseur = 2` is the **measured** optimum, not a guess, and the
    # default matters: at 40 the slab spans the cluster's whole depth and the
    # picture becomes a column density — a smooth radial gradient with a ragged
    # masked edge, in which the wake is diluted fourteenfold. See `slab_z0`.
    o = Dict("kev" => "4", "images" => "400", "particules" => "800000",
             "epaisseur" => "2", "finesse" => "3")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("unrecognised argument: $a")
        o[m[1]] = m[2]
    end
    o
end

"""Density averaged over the planes with `|z| ≤ halfwidth`, in `Float32`.

⚠️ **A thicker slab is not a better one, and the reasoning that says otherwise
is wrong.** Averaging `n` planes divides the shot noise by `√n`, which is true
and tempting — but the wake fits inside a **single cell in `z`**, so widening
the slab dilutes the signal faster than it kills the noise. Measured, at 4 keV
with the ion at the centre:

| `\\|z\\| ≤` | planes | signal | noise | S/N |
|---|---|---|---|---|
| 0 (one plane) | 1 | 1.09e-3 | 3.60e-4 | 3.02 |
| **2** | **2** | 1.18e-3 | 3.18e-4 | **3.70** |
| 7 | 8 | 5.57e-4 | 1.79e-4 | 3.11 |
| 40 (full depth) | 46 | 7.89e-5 | 6.34e-5 | **1.24** |

So the optimum is `|z| ≤ 2`, worth 23 % over a single plane, and a full-depth
projection is **worse than one plane**. It also stops being a cut and becomes a
*column density*: a smooth radial gradient with a ragged masked edge, which is
what a projected sphere looks like.

`halfwidth = 0` samples the single plane `z = 0` exactly.

**The cut is evaluated from the spline, not read off the grid.** The density is
a C¹ cubic spline and the collocation values are point samples of it; displaying
those samples as hard cells throws the representation away and keeps only the
sample. It shows at the cluster's surface, which is one or two cells thick and
therefore comes out as a staircase of blue blocks. `finesse` is how many output
points per collocation interval — 3 is enough to stop the grid showing through.

⚠️ This costs nothing in accuracy and **nothing in noise**: it interpolates the
same noisy values. It removes the staircase, not the grain.
"""
function cut_z0(ρ, mesh, halfwidth, xs, ys)
    csol = spline_coefficients(ρ, mesh)
    ax = mesh.axes
    zs = halfwidth > 0 ? range(-halfwidth, halfwidth; length = 3) : range(0, 0; length = 1)
    out = zeros(Float32, length(xs), length(ys))
    Threads.@threads for j in eachindex(ys)
        @inbounds for i in eachindex(xs)
            s = 0.0
            for z in zs
                v = spline_potential(ax, csol, (xs[i], ys[j], z))
                v === nothing || (s += v)
            end
            out[i, j] = Float32(s / length(zs))
        end
    end
    out
end

function main(argv)
    o = parse_args(argv)
    keV = parse(Int, o["kev"])
    nimg = parse(Int, o["images"])
    npart = parse(Int, o["particules"])
    halfwidth = parse(Float64, o["epaisseur"])
    finesse = parse(Int, o["finesse"])

    grid, ρr = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
    p = SimulationParameters(nfine = 44, ninner = 22, nouter = 22, rcluster = 78.0,
                             rbox = 235.0, nions = 1000.0, nelectrons = 1000.0,
                             nparticles = npart, nsteps = 0, dt = 1.0)
    energy = keV * KEV
    proj = Projectile(mass = 1836.154, charge = 1.0, energy = energy, impact = 0.0,
                      x0 = -65.0, dt = 1.0, softening = GaussianSoftening(1.0))
    sim = Simulation(p, PotentialProfile(grid, ρr); projectile = proj)

    v = sqrt(2energy / 1836.154)
    nsteps = ceil(Int, 1.1 * (80 + 65) / v)
    stride = max(1, round(Int, nsteps / nimg))

    fine = sim.meshes[1]
    # Output grid: `finesse` points per collocation interval, spanning the same
    # domain. The spline is evaluated here, not sampled at the knots.
    # ⚠️ `outx`, not `xs`: `xs` is the projectile's trajectory further down, and
    # naming both the same silently fed the growing trajectory to the evaluator.
    cx, cy = fine.axes[1].colloc, fine.axes[2].colloc
    outx = collect(range(cx[1], cx[end]; length = finesse * length(cx)))
    outy = collect(range(cy[1], cy[end]; length = finesse * length(cy)))
    gx, gy = Float32.(outx), Float32.(outy)
    ρ0 = cut_z0(sim.ρ[1], fine, halfwidth, outx, outy)

    # ⚠️ The GPU path works in `Float32`: the trajectory is no longer the CPU
    # reference's, only the same to within `4e-5`. For a film that is far below
    # anything visible; for a published `dE/dx` it is not — use the CPU path.
    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing

    @printf("Na1000, %d keV (v = %.2f), %d particles, %d steps, one frame every %d\n",
            keV, v, npart, nsteps, stride)
    @printf("BLAS: %s    forces: %s    cut: |z| ≤ %g, %d×%d points (finesse %d)\n",
            ACCELERATE ? "Accelerate" : "OpenBLAS", METAL ? "GPU (Float32)" : "CPU",
            halfwidth, length(outx), length(outy), finesse)

    frames = Matrix{Float32}[]
    xs = Float64[]; eks = Float64[]
    prog = ProgressMeter.Progress(nsteps + 1; desc = "crossing ", showspeed = true)
    for i in 0:nsteps
        i > 0 && step!(sim; energy = false, accelerator = acc)
        if i % stride == 0
            push!(frames, cut_z0(sim.ρ[1], fine, halfwidth, outx, outy))
            push!(xs, sim.projectile.position[1])
            push!(eks, kinetic_energy(sim.projectile))
        end
        ProgressMeter.next!(prog; showvalues = [("x (a₀)", round(sim.projectile.position[1], digits = 1)),
                                  ("frames", length(frames))])
        # The projectile can leave before `nsteps` — `nsteps` carries a 10 %
        # margin because it slows down. Close the bar rather than leave it
        # hanging at 92 %.
        sim.projectile.position[1] > 80 && break
    end
    ProgressMeter.finish!(prog)
    @printf("  %d frames\n", length(frames))

    out = joinpath(ROOT, "film.jls")
    serialize(out, (; gx, gy, finesse, halfwidth, rho0 = ρ0, frames, xs, eks,
                    keV, v, npart, stride, dt = p.dt,
                    e0 = proj.initial_energy, hartree = HARTREE_TO_EV))
    @printf("→ %s (%.1f MB)\n", out, filesize(out) / 2^20)
end

main(ARGS)
