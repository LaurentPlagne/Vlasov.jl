#!/usr/bin/env julia
"""
Produce the frames of a crossing movie: slices of the electron density.

    julia --project=. -t auto scripts/film_images.jl [--kev=4] [--images=400] [--particules=800000]

This is the view of the thesis's `snappro` figures — a **slice** of the density
in the plane `z = 0`, the one the projectile travels through. What it shows is
the deformation of the cloud as the ion passes, and its **asymmetry**: the wake
trails behind the ion, the more so the faster it goes.

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

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV

function parse_args(argv)
    o = Dict("kev" => "4", "images" => "400", "particules" => "800000")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("unrecognised argument: $a")
        o[m[1]] = m[2]
    end
    o
end

"""Slice of the density at `z = 0`, narrowed to `Float32`.

The plane is chosen at the collocation point nearest zero: the grid does not
necessarily hold one exactly, the points being at the Gauss nodes.
"""
function slice_z0(ρ, mesh)
    gz = mesh.axes[3].colloc
    k = argmin(abs.(gz))
    (Float32.(@view ρ[:, :, k]), k)
end

function main(argv)
    o = parse_args(argv)
    keV = parse(Int, o["kev"])
    nimg = parse(Int, o["images"])
    npart = parse(Int, o["particules"])

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
    ρ0, kplane = slice_z0(sim.ρ[1], fine)

    @printf("Na1000, %d keV (v = %.2f), %d particles, %d steps, one frame every %d\n",
            keV, v, npart, nsteps, stride)

    frames = Matrix{Float32}[]
    xs = Float64[]; eks = Float64[]
    t0 = time()
    for i in 0:nsteps
        i > 0 && step!(sim; energy = false)
        if i % stride == 0
            push!(frames, first(slice_z0(sim.ρ[1], fine)))
            push!(xs, sim.projectile.position[1])
            push!(eks, kinetic_energy(sim.projectile))
        end
        sim.projectile.position[1] > 80 && break
    end
    @printf("  %d frames in %.0f s\n", length(frames), time() - t0)

    out = joinpath(ROOT, "film.jls")
    serialize(out, (; gx, gy, kplane, rho0 = ρ0, frames, xs, eks,
                    keV, v, npart, stride, dt = p.dt,
                    e0 = proj.initial_energy, hartree = HARTREE_TO_EV))
    @printf("→ %s (%.1f MB)\n", out, filesize(out) / 2^20)
end

main(ARGS)
