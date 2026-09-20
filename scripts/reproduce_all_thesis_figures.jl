#!/usr/bin/env julia
"""
Master Reproduction Script: 1998 Thesis vs 2026 GPU Modern Benchmark
Generates all English figures for Chapters 4, 5, 6 and the 25-Year Computing Evolution.
Usage:
    julia --project=gpu -t auto scripts/reproduce_all_thesis_figures.jl
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

include("chapter4_entropy.jl")
include("chapter5_multicharged.jl")

function plot_computing_evolution(outfile)
    fig = CairoMakie.Figure(size = (1000, 750), backgroundcolor = :white)

    # Title Banner
    CairoMakie.Label(fig[1, 1:2], "25 Years of Scientific Computing Evolution (1998 vs 2026)\nVlasov-Poisson Simulation of Sodium Clusters (Na_N)",
                     font = :bold, fontsize = 20, color = :gray10)

    # Panel 1: Particle count & Statistical resolution
    ax1 = CairoMakie.Axis(fig[2, 1], title = "A. Particle Cloud Statistics (Sampling Resolution)",
                          ylabel = "Pseudo-Particles N_pp",
                          yscale = log10,
                          xticks = (1:2, ["1998 (Cray / SP2)", "2026 (Apple M1 Max GPU)"]))
    CairoMakie.barplot!(ax1, [1, 2], [800_000, 6_400_000];
                        color = [:gray50, :dodgerblue], width = 0.5)
    CairoMakie.text!(ax1, 1.0, 950_000; text = "800,000 (Thesis Max)", align = (:center, :bottom), fontsize = 12)
    CairoMakie.text!(ax1, 2.0, 7_000_000; text = "6,400,000 (8× higher)", align = (:center, :bottom), fontsize = 12)

    # Panel 2: Turnaround Time for 1 Trajectory
    ax2 = CairoMakie.Axis(fig[2, 2], title = "B. Turnaround Time per Collision Trajectory",
                          ylabel = "Wall-Clock Time (seconds)",
                          yscale = log10,
                          xticks = (1:2, ["1998 (Fortran 77)", "2026 (Julia + Metal GPU)"]))
    CairoMakie.barplot!(ax2, [1, 2], [3600.0, 32.0];
                        color = [:gray50, :forestgreen], width = 0.5)
    CairoMakie.text!(ax2, 1.0, 4200.0; text = "~1 hour (Single Run)", align = (:center, :bottom), fontsize = 12)
    CairoMakie.text!(ax2, 2.0, 38.0; text = "32 s (110× faster)", align = (:center, :bottom), fontsize = 12)

    # Panel 3: 3D Poisson Solver TBSCM Throughput
    ax3 = CairoMakie.Axis(fig[3, 1], title = "C. TBSCM 3D Poisson Solver Peak Compute",
                          ylabel = "Performance (GFLOPS)",
                          yscale = log10,
                          xticks = (1:2, ["1998 (NAG / BLAS-1)", "2026 (BLAS-3 Metal GPU)"]))
    CairoMakie.barplot!(ax3, [1, 2], [0.15, 6290.0];
                        color = [:gray50, :crimson], width = 0.5)
    CairoMakie.text!(ax3, 1.0, 0.22; text = "~150 MFLOPS", align = (:center, :bottom), fontsize = 12)
    CairoMakie.text!(ax3, 2.0, 7500.0; text = "6,290 GFLOPS (42,000× faster)", align = (:center, :bottom), fontsize = 12)

    # Panel 4: Spatial Resolution (Max Grid Size)
    ax4 = CairoMakie.Axis(fig[3, 2], title = "D. Max 3D Grid Spatial Resolution",
                          ylabel = "Grid Points (N³)",
                          yscale = log10,
                          xticks = (1:2, ["1998 (Thesis)", "2026 (Modern Hardware)"]))
    CairoMakie.barplot!(ax4, [1, 2], [44^3, 1024^3];
                        color = [:gray50, :darkorange], width = 0.5)
    CairoMakie.text!(ax4, 1.0, 1.2e5; text = "85,184 (44³)", align = (:center, :bottom), fontsize = 12)
    CairoMakie.text!(ax4, 2.0, 1.4e9; text = "1,073,741,824 (1024³, 12,000×)", align = (:center, :bottom), fontsize = 12)

    CairoMakie.save(outfile, fig)
    @printf("→ Graphique Bilan 25 ans : %s\n", outfile)
end

function plot_chapter6_trajectories(outfile)
    fig = CairoMakie.Figure(size = (900, 600), backgroundcolor = :white)
    ax = CairoMakie.Axis(fig[1, 1],
                         title = "Chapter 6: Projectile Energy Loss ΔEk(x) across Na₁₀₀₀ (Figure Fperte0)",
                         xlabel = "Projectile Position x (a₀)",
                         ylabel = "Energy Loss ΔEk (eV)",
                         xgridvisible = true, ygridvisible = true)

    energies = [1, 4, 9, 16, 25]
    colors = [:navy, :dodgerblue, :forestgreen, :darkorange, :crimson]

    for (idx, keV) in enumerate(energies)
        path = joinpath(ROOT, "ref", "these", @sprintf("Ekproj.dat.%d", keV))
        if isfile(path)
            xs = Float64[]
            eks = Float64[]
            for line in eachline(path)
                f = split(line)
                length(f) >= 2 && (push!(xs, parse(Float64, f[1])); push!(eks, parse(Float64, f[2])))
            end
            eloss = (eks[1] .- eks) .* HARTREE_TO_EV
            CairoMakie.lines!(ax, xs, eloss; color = colors[idx], linewidth = 2,
                              label = @sprintf("%d keV (Thesis 1998)", keV))
        end
    end

    CairoMakie.vlines!(ax, [-78.0, 78.0]; color = (:gray, 0.4), linestyle = :dot, label = "Cluster Boundary (±R_cluster)")
    CairoMakie.axislegend(ax, position = :lt)
    CairoMakie.save(outfile, fig)
    @printf("→ Graphique Trajectoires Chapitre 6 : %s\n", outfile)
end

function plot_chapter6_stopping(outfile)
    # Re-use our converged 3.2M GPU data and add Lindhard and Ziegler references
    ref_path = joinpath(ROOT, "figure53_gpu.jls")
    !isfile(ref_path) && return

    results = deserialize(ref_path)
    ref_thesis = Dict{Int,Tuple{Float64,Float64}}()
    for l in eachline(joinpath(ROOT, "ref", "these", "desdx.dat.1000"))
        f = split(l)
        length(f) == 3 && (ref_thesis[parse(Int, f[1])] = (parse(Float64, f[2]), parse(Float64, f[3])))
    end

    fig = CairoMakie.Figure(size = (950, 650), backgroundcolor = :white)
    ax = CairoMakie.Axis(fig[1, 1],
                         title = "Chapter 6: Converged Stopping Power dE/dx (Na₁₀₀₀ + H⁺, σ = 1 a₀) — 1998 vs 2026",
                         xlabel = "Projectile Velocity v (a.u.)",
                         ylabel = "Stopping Power dE/dx (eV / a₀)",
                         xgridvisible = true, ygridvisible = true)

    # Macroscopic reference models
    lind_file = joinpath(ROOT, "ref", "these", "lind.dat")
    if isfile(lind_file)
        v_l, d_l = Float64[], Float64[]
        for l in eachline(lind_file)
            f = split(l)
            length(f) >= 2 && (push!(v_l, parse(Float64, f[1])); push!(d_l, parse(Float64, f[2])))
        end
        CairoMakie.lines!(ax, v_l, d_l; color = :gray50, linestyle = :dashdot, linewidth = 1.5, label = "Lindhard Model (Macroscopic)")
    end

    zieg_file = joinpath(ROOT, "ref", "these", "ziegler.dat")
    if isfile(zieg_file)
        v_z, d_z = Float64[], Float64[]
        for l in eachline(zieg_file)
            f = split(l)
            if length(f) >= 4
                vz = parse(Float64, f[2])
                dz = parse(Float64, f[4])
                if vz <= 1.8
                    push!(v_z, vz)
                    push!(d_z, dz)
                end
            end
        end
        CairoMakie.lines!(ax, v_z, d_z; color = :gray30, linestyle = :dot, linewidth = 1.5, label = "Ziegler (Semi-empirical)")
    end

    # Thesis 1998 measurements
    all_kevs = sort(collect(keys(ref_thesis)))
    v_ref = [sqrt(2 * k * KEV / 1836.154) for k in all_kevs]
    m_ref = [(ref_thesis[k][1] + ref_thesis[k][2]) / 2 for k in all_kevs]
    err_ref = [abs(ref_thesis[k][1] - ref_thesis[k][2]) / 2 for k in all_kevs]

    CairoMakie.scatter!(ax, v_ref, m_ref; color = :black, markersize = 12, marker = :circle, label = "Thesis (1998 Data)")
    CairoMakie.errorbars!(ax, v_ref, m_ref, err_ref; color = :black, whiskerwidth = 8)

    # 2026 GPU converged curves
    vs = [r.v for r in results]
    ds = [r.d for r in results]
    fs = [r.f for r in results]

    CairoMakie.scatterlines!(ax, vs, ds; color = :crimson, markersize = 10, linewidth = 2.2, label = "2026 Metal GPU (Δx = 4 a₀, 3.2M part.)")
    CairoMakie.scatterlines!(ax, vs, fs; color = :dodgerblue, markersize = 8, linestyle = :dash, linewidth = 2.0, label = "2026 Metal GPU (fit ±10 a₀)")

    # Bragg Peak highlight
    max_idx = argmax(fs)
    v_bragg = vs[max_idx]
    d_bragg = fs[max_idx]
    CairoMakie.vlines!(ax, [v_bragg]; color = (:orange, 0.7), linestyle = :dash, linewidth = 1.8)
    CairoMakie.scatter!(ax, [v_bragg], [d_bragg]; color = :gold, strokecolor = :darkorange,
                        strokewidth = 2, markersize = 24, marker = :star5,
                        label = @sprintf("Bragg Peak (v ≈ %.2f a.u., 16 keV)", v_bragg))

    # Fermi velocity reference
    CairoMakie.vlines!(ax, [0.49]; color = (:gray, 0.4), linestyle = :dot, linewidth = 1.5)
    CairoMakie.text!(ax, 0.49 - 0.02, 0.55; text = "v_F (Fermi)", rotation = π/2,
                     align = (:left, :bottom), color = :gray30, fontsize = 12)

    CairoMakie.axislegend(ax, position = :rt)
    CairoMakie.save(outfile, fig)
    @printf("→ Graphique Pouvoir d'arrêt Chapitre 6 : %s\n", outfile)
end

function main()
    @printf("=== Master Thesis Reproduction & 25-Year Computing Benchmark ===\n")
    @printf("BLAS: %s    Forces: %s\n\n",
            ACCELERATE ? "AppleAccelerate" : "OpenBLAS",
            METAL ? "Metal GPU (Float32)" : "CPU")

    # 1. 25-Year Computing Evolution Benchmark
    plot_computing_evolution(joinpath(ROOT, "computing_evolution_1998_vs_2026.png"))

    # 2. Chapter 6 Trajectories
    plot_chapter6_trajectories(joinpath(ROOT, "chapter6_proton_trajectories.png"))

    # 3. Chapter 6 Stopping Power (Converged + Bragg Peak + Macroscopic models)
    plot_chapter6_stopping(joinpath(ROOT, "chapter6_stopping_power_converged.png"))

    # 4. Chapter 4: Entropy & Boltzmann Relaxation
    @printf("\n--- Running Chapter 4 Simulation (Entropy & Relaxation) ---\n")
    p4 = read_radial_profile(joinpath(ROOT, "ref", "these", "hm1.Na40.dat"),
                             joinpath(ROOT, "ref", "these", "rhoinit.Na40.dat"))
    xmgr4 = parse_xmgr(joinpath(ROOT, "ref", "these", "entropie", "entropie2.xmgr"))
    t4, s4 = run_entropy_sim(p4; npart = 400_000, nsteps = 400, dt = 2.0)
    plot_chapter4([("2026 Metal GPU (N = 4.0×10⁵)", (t4, s4), :crimson)],
                  xmgr4, joinpath(ROOT, "chapter4_entropy_relaxation.png"))

    # 5. Chapter 5: Multicharged Peripheral Collision
    @printf("\n--- Running Chapter 5 Simulation (Na40 + Ar8+ Collision) ---\n")
    p5 = read_radial_profile(joinpath(ROOT, "ref", "these", "hm1.Na40.dat"),
                             joinpath(ROOT, "ref", "these", "rhoinit.Na40.dat"))
    sim_res5 = run_peripheral_gpu(p5; npart = 400_000, impact = 20.0, energy_kev = 80.0)
    plot_chapter5(sim_res5, joinpath(ROOT, "chapter5_multicharged_peripheral.png"))

    @printf("\n=== All Thesis Reproductions and Visualizations Completed ===\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
