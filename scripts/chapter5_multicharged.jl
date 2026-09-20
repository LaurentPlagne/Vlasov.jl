#!/usr/bin/env julia
"""
Chapter 5: Peripheral Collisions with Highly Charged Ions (Na40 + Ar8+)
Simulates peripheral collision at non-zero impact parameter b = 20 a₀,
tracking electron capture around the projectile ion, ionization, and dipole excitation,
and comparing against historical 1998 thesis data (Vlasov, OBM, DOBM).
"""

using Vlasov
using Printf
using Serialization

# ⚠️ `functional()`, not merely `using`: `using Metal` SUCCEEDS on a machine
# with no Apple GPU — it only logs an error — and the script then took the
# Metal path on a Linux box with an NVIDIA card. Reported from one.
const METAL = try; @eval using Metal; @eval Metal.functional(); catch; false; end
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end
const MAKIE = try; @eval using CairoMakie; CairoMakie.activate!(type = "png"); true; catch; false; end

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV
const FS_TO_AU = 41.34137

function run_peripheral_gpu(profile; npart = 400_000, impact = 20.0, energy_kev = 80.0)
    # Na40 cluster
    nions = 40.0
    nelectrons = 40.0
    rcluster = 25.0
    rbox = 70.0
    nfine = 44
    m = nfine ÷ 2 + 1
    n1 = m ÷ 2
    n2 = m - n1
    ninner = 2 * n1
    nouter = 2 * n2 - 2

    # Ar8+ projectile: mass ≈ 40 * 1836.154, charge = 8.0, energy = 80 keV
    mass_ar = 40 * 1836.154
    energy_au = energy_kev * KEV
    v = sqrt(2 * energy_au / mass_ar) # v ≈ 0.28 u.a.
    dt = 0.5

    p = SimulationParameters(nfine = nfine, ninner = ninner,
                             nouter = nouter, rcluster = rcluster,
                             rbox = rbox, nions = nions, nelectrons = nelectrons,
                             nparticles = npart, nsteps = 0, dt = dt)

    x0 = -60.0
    proj = Projectile(mass = mass_ar, charge = 8.0, energy = energy_au,
                      impact = impact, x0 = x0, dt = dt,
                      softening = BallSoftening(2.0))
    sim = Simulation(p, profile; projectile = proj)
    fine = sim.meshes[1]
    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing

    nsteps = ceil(Int, (abs(x0) + 60.0) / (v * dt))

    xs = Float64[]
    times_fs = Float64[]
    captured_q5 = Float64[]
    captured_q8 = Float64[]
    cluster_charges = Float64[]

    r_cluster_edge = 15.0 # Cluster jellium radius is ~13.7 a0

    @printf("Simulating Na40 + Ar8+ (80 keV, b = %.1f a₀): %d particles...\n", impact, npart)
    for step in 1:nsteps
        step!(sim; energy = false, accelerator = acc)
        x_proj = proj.position[1]
        t_fs = (step * dt) / FS_TO_AU

        # Capture diagnostic: charge within R = 5 a0 and R = 8 a0 around ion
        q_cap5 = enclosed_charge(sim.cloud, proj, 5.0)
        q_cap8 = enclosed_charge(sim.cloud, proj, 8.0)

        # Net cluster charge (electrons within R = 18 a0)
        n_cluster = count(pt -> sum(abs2, pt) < 18.0^2, sim.cloud.positions)
        q_net = nions - sim.cloud.weight * n_cluster

        push!(xs, x_proj)
        push!(times_fs, t_fs)
        push!(captured_q5, q_cap5)
        push!(captured_q8, q_cap8)
        push!(cluster_charges, q_net)

        if step % 20 == 0
            @printf("  x = %6.1f a₀ (t = %5.1f fs): Q_cap(R=5) = %5.2f e, Q_cluster = +%5.2f\n",
                    x_proj, t_fs, q_cap5, q_net)
            flush(stdout)
        end
        x_proj > 60.0 && break
    end

    (times_fs, xs, captured_q5, captured_q8, cluster_charges)
end

function plot_chapter5(sim_res, outfile)
    times_fs, xs, q5, q8, q_net = sim_res

    fig = CairoMakie.Figure(size = (900, 650), backgroundcolor = :white)

    # Subplot 1: Captured charge vs time
    ax1 = CairoMakie.Axis(fig[1, 1],
                          title = "Chapter 5: Electron Capture in Na₄₀ + Ar⁸⁺ (80 keV, b = 20 a₀)",
                          xlabel = "Projectile Position x (a₀)",
                          ylabel = "Captured Electronic Charge (e)",
                          xgridvisible = true, ygridvisible = true)

    CairoMakie.lines!(ax1, xs, q5; color = :crimson, linewidth = 2.5,
                      label = "Captured Charge Q_cap (R = 5 a₀)")
    CairoMakie.lines!(ax1, xs, q8; color = :dodgerblue, linewidth = 2.0, linestyle = :dash,
                      label = "Captured Charge Q_cap (R = 8 a₀)")
    CairoMakie.vlines!(ax1, [0.0]; color = (:gray, 0.5), linestyle = :dot, label = "Closest Approach (x = 0)")
    CairoMakie.axislegend(ax1, position = :lt)

    # Subplot 2: Cluster Ionization vs time
    ax2 = CairoMakie.Axis(fig[2, 1],
                          title = "Cluster Net Ionization vs Projectile Trajectory",
                          xlabel = "Projectile Position x (a₀)",
                          ylabel = "Cluster Net Charge Q_net (e)",
                          xgridvisible = true, ygridvisible = true)

    CairoMakie.lines!(ax2, xs, q_net; color = :darkorange, linewidth = 2.5,
                      label = "Net Cluster Charge Q_cluster")
    CairoMakie.vlines!(ax2, [0.0]; color = (:gray, 0.5), linestyle = :dot)
    CairoMakie.axislegend(ax2, position = :lt)

    CairoMakie.save(outfile, fig)
    @printf("→ Graphique Chapitre 5 : %s\n", outfile)
end

function main()
    profile = read_radial_profile(joinpath(ROOT, "ref", "these", "hm1.Na40.dat"),
                                  joinpath(ROOT, "ref", "these", "rhoinit.Na40.dat"))
    sim_res = run_peripheral_gpu(profile; npart = 400_000, impact = 20.0, energy_kev = 80.0)

    outfile = joinpath(ROOT, "chapter5_multicharged_peripheral.png")
    if MAKIE
        plot_chapter5(sim_res, outfile)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
