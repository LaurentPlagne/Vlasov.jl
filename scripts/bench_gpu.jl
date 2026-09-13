#!/usr/bin/env julia
"""
Compare the smoothed-field evaluation on CPU (`Float64`) and on GPU (`Float32`).

    julia --project=gpu -t auto scripts/bench_gpu.jl

Two things at once, because they cannot be separated: **how much** faster the
GPU is, and **what it costs** in accuracy. A speed-up without the discrepancy
beside it would mean nothing — Apple GPUs have no double precision, so the
comparison is between two arithmetics.
"""

using Vlasov, Metal, Printf, LinearAlgebra

const ROOT = dirname(@__DIR__)

function setup(npart)
    grid, ρ = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
    p = SimulationParameters(nfine = 44, ninner = 22, nouter = 22, rcluster = 78.0,
                             rbox = 235.0, nions = 1000.0, nelectrons = 1000.0,
                             nparticles = npart, nsteps = 0, dt = 1.0)
    proj = Projectile(mass = 1836.154, charge = 1.0, energy = 147.0, x0 = -30.0,
                      dt = 1.0, softening = GaussianSoftening(1.0))
    sim = Simulation(p, PotentialProfile(grid, ρ); projectile = proj)
    step!(sim)
    sim
end

"""Timing: warm up first, then average."""
function timed(f, k = 5)
    f()
    t0 = time()
    for _ in 1:k; f(); end
    1000(time() - t0) / k
end

function main()
    @printf("%-10s %-10s %-10s %-8s %-12s %s\n",
            "particles", "CPU (ms)", "GPU (ms)", "speed-up", "norm error", "median error")
    for npart in (200_000, 400_000, 800_000)
        sim = setup(npart)
        fine, coarse = sim.meshes[1], sim.meshes[2]
        acc = ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                               size(sim.csol[1], 1))

        cpu() = forces!(sim.cloud, fine.axes, sim.csol[1], coarse.axes,
                        sim.csol[2], sim.smoothing)
        gpu() = forces!(sim.cloud, acc, fine.axes, sim.csol[1], coarse.axes,
                        sim.csol[2], sim.smoothing)

        cpu(); ref = copy(sim.cloud.forces)
        gpu(); got = copy(sim.cloud.forces)
        d = [norm(collect(got[i] .- ref[i])) for i in 1:npart]
        r = [norm(collect(ref[i])) for i in 1:npart]
        med = sort(d ./ max.(r, eps()))[npart ÷ 2]

        tc, tg = timed(cpu), timed(gpu)
        @printf("%-10d %-10.1f %-10.1f x%-7.2f %-12.2e %.2e\n",
                npart, tc, tg, tc / tg, norm(d) / norm(r), med)
    end
end

main()
