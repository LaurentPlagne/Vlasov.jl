#!/usr/bin/env julia
"""
Produce the frames of a crossing movie: slices of the electron density.

    julia --project=gpu -t auto scripts/film_images.jl [--kev=4] [--images=400]
                                                       [--particules=800000] [--epaisseur=40]

Run it under `--project=gpu` and it picks up **AppleAccelerate** (BLAS) and
**Metal** (the smoothed field and the forces) on its own; under `--project=.`
neither is present and the plain CPU path runs. The banner says which was used.

This is the view of the thesis's `snappro` figures — a cut of the density
through the plane the projectile travels in. What it shows is the deformation of
the cloud as the ion passes, and its **asymmetry**: the wake trails behind the
ion, the more so the faster it goes.

⚠️ `--epaisseur` is not cosmetic. It averages the planes with `|z| ≤ epaisseur`
instead of keeping the single plane `z = 0`, and that is what decides whether
anything is visible at all — see [`slab_z0`](@ref). `--epaisseur=0` restores the
single plane.

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

# Optional accelerators. They live in the `gpu/` environment; under
# `--project=.` neither is present and the plain CPU path runs instead. Loading
# `AppleAccelerate` is enough — its `__init__` forwards BLAS and LAPACK
# correctly, which `BLAS.lbt_forward` on its own does not.
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end
const METAL = try; @eval using Metal; true; catch; false; end

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV

function parse_args(argv)
    o = Dict("kev" => "4", "images" => "400", "particules" => "800000",
             "epaisseur" => "40")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("unrecognised argument: $a")
        o[m[1]] = m[2]
    end
    o
end

"""Density averaged over the planes with `|z| ≤ halfwidth`, in `Float32`.

⚠️ **This averaging is what makes the picture readable, and it is free.** With
800 000 pseudo-particles a cell of the fine grid holds about 134 of them, so the
shot noise is `1/√134 ≈ 8.7 %` — measured at 9.6 % on `δρ` away from the wake.
The deformation the projectile leaves behind is of the same order, so on a
**single** plane it is buried: signal-to-noise of 3 per cell, which reads as
salt and pepper.

Averaging the `n` planes of a slab divides that noise by `√n` without costing a
single extra time step. Over the cluster's full depth (22 planes at `h = 3.5`)
that is ×4.7, and the wake comes out.

`halfwidth = 0` keeps the single plane nearest `z = 0` — the grid does not
necessarily hold one exactly, the points being at the Gauss nodes.
"""
function slab_z0(ρ, mesh, halfwidth)
    gz = mesh.axes[3].colloc
    ks = halfwidth > 0 ? findall(z -> abs(z) <= halfwidth, gz) : [argmin(abs.(gz))]
    isempty(ks) && (ks = [argmin(abs.(gz))])
    out = zeros(Float32, size(ρ, 1), size(ρ, 2))
    @inbounds for k in ks, j in axes(ρ, 2), i in axes(ρ, 1)
        out[i, j] += Float32(ρ[i, j, k])
    end
    out ./= length(ks)
    (out, length(ks))
end

function main(argv)
    o = parse_args(argv)
    keV = parse(Int, o["kev"])
    nimg = parse(Int, o["images"])
    npart = parse(Int, o["particules"])
    halfwidth = parse(Float64, o["epaisseur"])

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
    gx = Float32.(fine.axes[1].colloc)
    gy = Float32.(fine.axes[2].colloc)
    ρ0, nplanes = slab_z0(sim.ρ[1], fine, halfwidth)

    # ⚠️ The GPU path works in `Float32`: the trajectory is no longer the CPU
    # reference's, only the same to within `4e-5`. For a film that is far below
    # anything visible; for a published `dE/dx` it is not — use the CPU path.
    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing

    @printf("Na1000, %d keV (v = %.2f), %d particles, %d steps, one frame every %d\n",
            keV, v, npart, nsteps, stride)
    @printf("BLAS: %s    forces: %s    slab: %d planes (|z| ≤ %g), noise / %.1f\n",
            ACCELERATE ? "Accelerate" : "OpenBLAS", METAL ? "GPU (Float32)" : "CPU",
            nplanes, halfwidth, sqrt(nplanes))

    frames = Matrix{Float32}[]
    xs = Float64[]; eks = Float64[]
    t0 = time()
    for i in 0:nsteps
        i > 0 && step!(sim; energy = false, accelerator = acc)
        if i % stride == 0
            push!(frames, first(slab_z0(sim.ρ[1], fine, halfwidth)))
            push!(xs, sim.projectile.position[1])
            push!(eks, kinetic_energy(sim.projectile))
        end
        sim.projectile.position[1] > 80 && break
    end
    @printf("  %d frames in %.0f s\n", length(frames), time() - t0)

    out = joinpath(ROOT, "film.jls")
    serialize(out, (; gx, gy, nplanes, halfwidth, rho0 = ρ0, frames, xs, eks,
                    keV, v, npart, stride, dt = p.dt,
                    e0 = proj.initial_energy, hartree = HARTREE_TO_EV))
    @printf("→ %s (%.1f MB)\n", out, filesize(out) / 2^20)
end

main(ARGS)
