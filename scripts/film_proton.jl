#!/usr/bin/env julia
"""
Film & Snapshots of Na₁₀₀₀ + H⁺ Central Crossing (16 keV, b = 0)
Reproducing and converging Chapter 6 (Figure 5.2 / snappro) of the 1998 PhD thesis.

High-resolution simulation on Metal GPU:
- Grid: nfine = 66, ninner = 34, nouter = 32 (mesh 134×134×134)
- Converged particle statistics: 1,600,000 pseudo-particles (4× to 80× thesis count)
- Continuous cubic Hermite spline evaluation (cut_z0 with finesse = 6)
- Projectile stopping power dE/dx and energy loss ΔE tracking
- Cluster excitation energy tracking
- Generates film_proton.mp4, film_proton.gif, and proton_snapshots.png
"""

using Vlasov
using Printf
using Serialization
using CairoMakie
CairoMakie.activate!(type = "png")

const METAL = try; @eval using Metal; true; catch; false; end
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV
const GREY = RGBf(0.80, 0.80, 0.80)

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

function run_proton_simulation(; npart = 1_600_000, nfine = 66, dt = 1.0, finesse = 6)
    cache_file = joinpath(ROOT, "proton_converged_data_cache.jls")
    if isfile(cache_file)
        println("Loading cached Proton simulation run from $cache_file...")
        return deserialize(cache_file)
    end

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

    # 16 keV proton (Bragg peak resonant velocity v ≈ 0.80 a.u.)
    mass_p = 1836.154
    energy_au = 16.0 * KEV
    v0 = sqrt(2 * energy_au / mass_p) # v ≈ 0.800 a.u.
    x0 = -65.0
    proj = Projectile(mass = mass_p, charge = 1.0, energy = energy_au,
                      impact = 0.0, x0 = x0, dt = dt,
                      softening = GaussianSoftening(1.0))

    sim = Simulation(p, profile; projectile = proj)
    fine = sim.meshes[1]
    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing

    cx, cy = fine.axes[1].colloc, fine.axes[2].colloc
    outx = collect(range(cx[1], cx[end]; length = finesse * length(cx)))
    outy = collect(range(cy[1], cy[end]; length = finesse * length(cy)))

    println("Simulating Na₁₀₀₀ + H⁺ (16 keV, b = 0): $npart particles on grid $nfine...")
    println("BLAS: $(ACCELERATE ? "AppleAccelerate" : "OpenBLAS") | GPU: $(METAL ? "Metal (MtlArray)" : "CPU")")

    nsteps = ceil(Int, 1.1 * (80.0 + abs(x0)) / v0) # ~200 steps
    stride = 2

    # Measure initial cluster total energy
    b_init = step!(sim; energy = true, accelerator = acc)
    e_cluster_0 = b_init.total

    frames = Matrix{Float32}[]
    x_projs = Float64[]
    e_losses_ev = Float64[]
    times_fs = Float64[]
    e_excs_ev = Float64[]
    cur_exc = 0.0

    t_start = time()
    for step in 1:nsteps
        # Compute energy budget every 20 steps to observe excitation
        do_energy = (step % 20 == 0)
        b = step!(sim; energy = do_energy, accelerator = acc)
        if b !== nothing
            cur_exc = (b.total - e_cluster_0) * HARTREE_TO_EV
        end

        xp = sim.projectile.position[1]
        loss = (energy_au - kinetic_energy(sim.projectile)) * HARTREE_TO_EV
        t_fs = (step * dt) / 41.34137

        if step % stride == 0
            push!(frames, cut_z0(sim.ρ[1], fine, 2.0, outx, outy))
            push!(x_projs, xp)
            push!(e_losses_ev, loss)
            push!(times_fs, t_fs)
            push!(e_excs_ev, cur_exc)

            if length(frames) % 15 == 0
                @printf("  frame %3d (step %3d/%3d, x = %+5.1f a₀): loss = %5.2f eV, E_exc = %5.2f eV\n",
                        length(frames), step, nsteps, xp, loss, cur_exc)
                flush(stdout)
            end
        end

        xp > 80.0 && break
    end

    elapsed = time() - t_start
    @printf("Simulation complete in %.2f s (%d frames captured)\n", elapsed, length(frames))

    # Calculate stopping power around center: ΔE / 4 a0 between x = -2 and x = +2
    idx_m2 = argmin(abs.(x_projs .- (-2.0)))
    idx_p2 = argmin(abs.(x_projs .- (+2.0)))
    dx_center = x_projs[idx_p2] - x_projs[idx_m2]
    dE_dx_center = (e_losses_ev[idx_p2] - e_losses_ev[idx_m2]) / dx_center
    total_loss = last(e_losses_ev)
    final_exc = last(e_excs_ev)

    @printf("=== CONVERGED PROTON OBSERVABLES ===\n")
    @printf("  Central Stopping Power dE/dx: %5.3f eV/a₀ (1998 Thesis: 1.587 eV/a₀)\n", dE_dx_center)
    @printf("  Total Projectile Energy Loss: %5.2f eV\n", total_loss)
    @printf("  Final Cluster Excitation:     %5.2f eV\n", final_exc)

    data = (frames = frames, x_projs = x_projs, e_losses_ev = e_losses_ev, times_fs = times_fs,
            e_excs_ev = e_excs_ev, outx = outx, outy = outy, r_cluster = 78.0,
            dE_dx_center = dE_dx_center, total_loss = total_loss, final_exc = final_exc,
            npart = npart, nfine = nfine)

    serialize(cache_file, data)
    println("Saved Proton simulation run to $cache_file")
    data
end

function render_proton_results(data, out_mp4, out_gif, out_snapshots)
    println("Rendering Proton results...")
    nframes = length(data.frames)
    xs = data.outx
    ys = data.outy

    n0 = 0.00373f0
    c_min = 0.02f0 * n0
    c_max = 1.45f0 * n0
    levels = range(c_min, c_max, length = 45)

    # 1. 4-Panel Snapshot Strip showing Wake Dynamics
    target_xs = [-40.0, -15.0, 0.0, 30.0]
    snap_indices = [argmin(abs.(data.x_projs .- tx)) for tx in target_xs]

    fig_snap = CairoMakie.Figure(size = (1100, 950), backgroundcolor = :white)
    CairoMakie.Label(fig_snap[0, 1:2],
        @sprintf("Na₁₀₀₀ + H⁺ (16 keV, b = 0) — High-Resolution Wake Structures\n(Converged GPU Simulation: N_pp = %d, Mesh %d, finesse = 6)",
                 data.npart, data.nfine),
        fontsize = 18, font = :bold)

    R_cluster = 40.0 # Wigner-Seitz cluster edge ~ 3.93 * 10 = 39.3 a0
    θ = range(0, 2π, length = 150)

    titles = [
        "1. Ingress: Proton Entering Cluster (x = -40 a₀)",
        "2. Wake Onset: Compression & Polarisation (x = -15 a₀)",
        "3. Core Passage: Dual Plasmon Wake Nodes (x = 0 a₀)",
        "4. Exit: Oscillatory Trailing Wake (x = +30 a₀)"
    ]

    for (k, idx) in enumerate(snap_indices)
        row = (k - 1) ÷ 2 + 1
        col = (k - 1) % 2 + 1
        xp = data.x_projs[idx]
        loss = data.e_losses_ev[idx]

        ax = CairoMakie.Axis(fig_snap[row, col],
            title = @sprintf("%s\nloss = %4.1f eV", titles[k], loss),
            xlabel = row == 2 ? "x (a₀)" : "",
            ylabel = col == 1 ? "y (a₀)" : "",
            aspect = CairoMakie.DataAspect(),
            backgroundcolor = GREY)
        CairoMakie.xlims!(ax, -60, 60)
        CairoMakie.ylims!(ax, -50, 50)

        CairoMakie.contourf!(ax, xs, ys, data.frames[idx],
                             levels = levels, colormap = :turbo, extendlow = GREY)

        CairoMakie.lines!(ax, R_cluster .* cos.(θ), R_cluster .* sin.(θ),
                          color = :white, linestyle = :dash, linewidth = 1.5)

        CairoMakie.scatter!(ax, [xp], [0.0], color = :white, strokecolor = :black,
                            strokewidth = 2.0, markersize = 12)
    end

    CairoMakie.save(out_snapshots, fig_snap, px_per_unit = 2)
    println("Saved Proton snapshot strip to $out_snapshots")

    # 2. Dynamic Film (MP4 + GIF)
    fig_anim = CairoMakie.Figure(size = (800, 950), backgroundcolor = :white)
    f_idx = CairoMakie.Observable(1)

    ax_cut = CairoMakie.Axis(fig_anim[1, 1],
        title = CairoMakie.@lift(@sprintf("Na₁₀₀₀ + H⁺ (16 keV, b = 0) — x_p = %+5.1f a₀, ΔE = %5.1f eV",
                                          data.x_projs[$f_idx], data.e_losses_ev[$f_idx])),
        xlabel = "x (a₀)", ylabel = "y (a₀)",
        aspect = CairoMakie.DataAspect(),
        backgroundcolor = GREY)
    CairoMakie.xlims!(ax_cut, -65, 65)
    CairoMakie.ylims!(ax_cut, -50, 50)

    cur_frame = CairoMakie.@lift data.frames[$f_idx]
    CairoMakie.contourf!(ax_cut, xs, ys, cur_frame,
                         levels = levels, colormap = :turbo, extendlow = GREY)

    CairoMakie.lines!(ax_cut, R_cluster .* cos.(θ), R_cluster .* sin.(θ),
                      color = :white, linestyle = :dash, linewidth = 2.0)

    p_pt = CairoMakie.@lift CairoMakie.Point2f(data.x_projs[$f_idx], 0.0)
    CairoMakie.scatter!(ax_cut, p_pt, color = :white, strokecolor = :black,
                        strokewidth = 2.5, markersize = 14)

    # Diagnostic plot: Energy Loss and Excitation
    ax_diag = CairoMakie.Axis(fig_anim[2, 1],
        title = "Energy Transfer: Projectile Stopping Loss vs Cluster Excitation",
        xlabel = "Projectile x (a₀)", ylabel = "Energy (eV)",
        xgridvisible = true, ygridvisible = true)
    CairoMakie.xlims!(ax_diag, data.x_projs[1], data.x_projs[end])
    CairoMakie.ylims!(ax_diag, -5, max(maximum(data.e_losses_ev), maximum(data.e_excs_ev)) * 1.15)

    CairoMakie.lines!(ax_diag, data.x_projs, data.e_losses_ev, color = (:crimson, 0.4), linewidth = 1.5)
    CairoMakie.lines!(ax_diag, data.x_projs, data.e_excs_ev, color = (:darkblue, 0.4), linewidth = 1.5)

    cur_xs = CairoMakie.@lift data.x_projs[1:$f_idx]
    cur_loss = CairoMakie.@lift data.e_losses_ev[1:$f_idx]
    cur_exc = CairoMakie.@lift data.e_excs_ev[1:$f_idx]

    CairoMakie.lines!(ax_diag, cur_xs, cur_loss, color = :crimson, linewidth = 2.5, label = "Projectile Energy Loss ΔE (eV)")
    CairoMakie.lines!(ax_diag, cur_xs, cur_exc, color = :darkblue, linewidth = 2.5, label = "Cluster Excitation E_exc (eV)")
    CairoMakie.vlines!(ax_diag, [-R_cluster, R_cluster], color = (:gray, 0.5), linestyle = :dash, label = "Cluster Jellium Edge")
    CairoMakie.axislegend(ax_diag, position = :lt)

    cur_pt_loss = CairoMakie.@lift [CairoMakie.Point2f(data.x_projs[$f_idx], data.e_losses_ev[$f_idx])]
    CairoMakie.scatter!(ax_diag, cur_pt_loss, color = :crimson, markersize = 10)

    println("Encoding Proton MP4 to $out_mp4...")
    CairoMakie.record(fig_anim, out_mp4, 1:nframes; framerate = 20) do i
        f_idx[] = i
    end
    println("Encoding Proton GIF to $out_gif...")
    CairoMakie.record(fig_anim, out_gif, 1:nframes; framerate = 15) do i
        f_idx[] = i
    end
    println("Proton rendering complete!")
end

function main()
    data = run_proton_simulation(npart = 1_600_000, nfine = 66, dt = 1.0, finesse = 6)
    out_mp4 = joinpath(ROOT, "film_proton_converged.mp4")
    out_gif = joinpath(ROOT, "film_proton_converged.gif")
    out_snapshots = joinpath(ROOT, "proton_snapshots_converged.png")
    render_proton_results(data, out_mp4, out_gif, out_snapshots)

    # Copy assets for documentation
    assets_dir = joinpath(ROOT, "docs", "src", "assets")
    cp(out_gif, joinpath(assets_dir, "film_proton_converged.gif"), force = true)
    cp(out_snapshots, joinpath(assets_dir, "proton_snapshots_converged.png"), force = true)
    println("Proton converged assets copied to $assets_dir")
end

main()
