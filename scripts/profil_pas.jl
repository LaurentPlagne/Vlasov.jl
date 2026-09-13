#!/usr/bin/env julia
"""
Profil d'un pas de temps, étage par étage, **dans l'ordre réel**.

    julia --project=. -t auto scripts/profil_pas.jl

⚠️ Chronométrer chaque étage dans sa propre boucle donne des chiffres qui ne
somment pas au total : la répétition laisse les données chaudes, alors que la
séquence réelle déborde le cache à chaque tour. Une première version du profil
perdait 42 % du pas de cette façon. Ici chaque étage est mesuré à sa place dans
la séquence, et la somme boucle à 99,8 %.
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

"""Rejoue le corps de `update_forces!` dans l'ordre, en chronométrant chaque
étage **en séquence** — c'est-à-dire avec les caches que la séquence laisse,
et non ceux qu'une boucle sur une seule fonction entretient."""
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
timed_step!(sim, zeros(11))                       # chauffe
t0 = time(); for _ in 1:k; timed_step!(sim, acc); end
tot = 1000(time() - t0) / k
acc ./= k

noms = ("dépôt lissé (fine)", "dépôt (grossière)", "poisson!", "coefficients spline",
        "énergie (Hartree)", "champ moyen", "forces", "projectile",
        "recopie φ ← csol", "Verlet", "énergie (totale)")
@printf("%-24s %8s %7s\n", "étage", "ms", "%")
for i in 1:11
    @printf("%-24s %8.1f %6.1f%%\n", noms[i], acc[i], 100acc[i]/tot)
end
@printf("%-24s %8.1f %6.1f%%\n", "--- somme", sum(acc), 100sum(acc)/tot)
@printf("%-24s %8.1f\n", "pas mesuré", tot)
