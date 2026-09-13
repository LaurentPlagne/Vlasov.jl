#!/usr/bin/env julia
"""
Profile of one time step, stage by stage, **in the real order**.

    julia --project=. -t auto scripts/profil_pas.jl

⚠️ Timing each stage in a loop of its own gives numbers that do not add up to
the total: the repetition leaves the data hot, whereas the real sequence
overflows the cache on every turn. An early version of this profile lost 42 %
of the step that way. Here each stage is measured in its place in the
sequence, and the sum closes at 99.8 %.
"""

using Vlasov, Printf
const ROOT = dirname(@__DIR__)
grid, ρr = read_radial_density(joinpath(ROOT, "ref/these/rhorad.Na1000.dat"))
npart = 800_000
p = SimulationParameters(nfine = 44, ninner = 22, nouter = 22, rcluster = 78.0,
                         rbox = 235.0, nions = 1000.0, nelectrons = 1000.0,
                         nparticles = npart, nsteps = 0, dt = 1.0)
proj = Projectile(mass = 1836.154, charge = 1.0, energy = 147.0, x0 = -30.0,
                  dt = 1.0, softening = GaussianSoftening(1.0))
sim = Simulation(p, PotentialProfile(grid, ρr); projectile = proj)
step!(sim)

"""Replay the body of `update_forces!` in order, timing each stage **in
sequence** — that is, with the caches the sequence leaves behind, not the ones
a loop over a single function keeps warm."""
function timed_step!(sim, acc)
    fine, coarse = sim.meshes[1], sim.meshes[2]
    ρf, ρc = sim.ρ; w = sim.cloud.weight
    m(k, f) = (t0 = time(); r = f(); acc[k] += 1000(time() - t0); r)

    m(1, () -> deposit_smoothed!(ρf, fine, sim.smoothing, sim.cloud.positions;
                                 charge = w, buffers = sim.scatter[1]))
    m(2, () -> deposit!(ρc, coarse, sim.cloud.positions; charge = w, buffers = sim.scatter[2]))
    m(3, () -> poisson!(sim.φ, sim.ρ, sim.meshes))
    csolf = m(4, () -> spline_coefficients!(sim.csol[1], sim.φ[1], fine))
    csolc = m(4, () -> spline_coefficients!(sim.csol[2], sim.φ[2], coarse))
    m(5, () -> interaction_energy(sim.cloud, fine.axes, csolf, coarse.axes, csolc, sim.smoothing))
    m(6, () -> (effective_potential!(csolf, ρf, fine, sim.jellium);
                effective_potential!(csolc, ρc, coarse, sim.jellium)))
    m(7, () -> forces!(sim.cloud, fine.axes, csolf, coarse.axes, csolc, sim.smoothing))
    m(8, () -> Vlasov.advance_projectile!(sim))
    m(9, () -> (sim.φ[1] .= csolf; sim.φ[2] .= csolc))
    m(10, () -> step!(sim.cloud, sim.params.dt; rcmax = sim.params.rcmax))
    m(11, () -> interaction_energy(sim.cloud, fine.axes, sim.φ[1], coarse.axes,
                                   sim.φ[2], sim.smoothing))
    nothing
end

acc = zeros(11); k = 6
timed_step!(sim, zeros(11))                       # warm-up
t0 = time(); for _ in 1:k; timed_step!(sim, acc); end
tot = 1000(time() - t0) / k
acc ./= k

stages = ("smoothed deposit (fine)", "deposit (coarse)", "poisson!", "spline coefficients",
          "energy (Hartree)", "mean field", "forces", "projectile",
          "copy φ ← csol", "Verlet", "energy (total)")
@printf("%-24s %8s %7s\n", "stage", "ms", "%")
for i in 1:11
    @printf("%-24s %8.1f %6.1f%%\n", stages[i], acc[i], 100acc[i]/tot)
end
@printf("%-24s %8.1f %6.1f%%\n", "--- sum", sum(acc), 100sum(acc)/tot)
@printf("%-24s %8.1f\n", "measured step", tot)
