#!/usr/bin/env julia
using Printf
using AppleAccelerate
using Metal
using Vlasov

const ROOT = dirname(@__DIR__)
println("=== 80 MILLION PARTICLES CALIBRATION TEST ===")
flush(stdout)

const npart = 80_000_000
const nfine = 110 # mesh 222³ points, h ≈ 1.42 a0
const dt = 1.0

m = nfine ÷ 2 + 1
n1 = m ÷ 2
n2 = m - n1
ninner = 2 * n1
nouter = 2 * n2 - 2
rcluster = 78.0
rbox = 235.0

p = SimulationParameters(nfine = nfine, ninner = ninner, nouter = nouter,
                         rcluster = rcluster, rbox = rbox,
                         nions = 1000.0, nelectrons = 1000.0,
                         nparticles = npart, nsteps = 0, dt = dt)

grid, ρr = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
profile = PotentialProfile(grid, ρr)

mass_p = 1836.154
const KEV = 1000.0 / 27.211386245988
energy_au = 16.0 * KEV
x0 = -65.0
proj = Projectile(mass = mass_p, charge = 1.0, energy = energy_au,
                  impact = 0.0, x0 = x0, dt = dt,
                  softening = GaussianSoftening(1.0))

println("1. Initializing 80M particle simulation...")
flush(stdout)
t0 = time()
sim = Simulation(p, profile; projectile = proj)
t_init = time() - t0
@printf("   Simulation initialized in %.2f s (%.1f MB/s)\n", t_init, (npart * 72) / 1024^2 / t_init)
flush(stdout)

println("2. Initializing Metal GPU accelerator (10+ GB buffers)...")
flush(stdout)
fine = sim.meshes[1]
t0 = time()
acc = ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart, size(sim.csol[1], 1))
t_acc = time() - t0
@printf("   Metal accelerator initialized in %.2f s\n", t_acc)
flush(stdout)

println("3. Executing Step 1...")
flush(stdout)
t0 = time()
step!(sim; accelerator = acc)
t_step = time() - t0
@printf("   Step 1 completed in %.2f s!\n", t_step)
@printf("   Estimated 200-step simulation runtime: %.1f minutes\n", (t_step * 200) / 60)
flush(stdout)

println("=== 80M CALIBRATION SUCCESSFUL ===")
flush(stdout)
