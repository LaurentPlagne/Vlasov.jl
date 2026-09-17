#!/usr/bin/env julia
# 80 Million Particles Simulation: Na₄₀ + Ar⁸⁺ (80 keV, b = 20 a₀)
# Converged 2026 Simulation on GPU Metal + BLAS AppleAccelerate (80,000,000 particles)

using Vlasov
using Printf
using Serialization
using CairoMakie
CairoMakie.activate!(type = "png")

const METAL = try; @eval using Metal; true; catch; false; end
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end

const ROOT = dirname(@__DIR__)
const KEV = 1000.0 / HARTREE_TO_EV
const FS_TO_AU = 41.34137
const GREY = CairoMakie.RGBf(0.82, 0.82, 0.82)

function resample_2d(xs, ys, V, n_out_x = 350, n_out_y = 350)
    x_min, x_max = xs[1], xs[end]
    y_min, y_max = ys[1], ys[end]
    xs_fine = range(x_min, x_max, length = n_out_x)
    ys_fine = range(y_min, y_max, length = n_out_y)
    V_fine = Matrix{Float32}(undef, n_out_x, n_out_y)
    nx, ny = length(xs), length(ys)
    for (j, y) in enumerate(ys_fine)
        jy = clamp(searchsortedlast(ys, y), 1, ny - 1)
        uy = clamp(Float32((y - ys[jy]) / (ys[jy+1] - ys[jy])), 0.0f0, 1.0f0)
        for (i, x) in enumerate(xs_fine)
            ix = clamp(searchsortedlast(xs, x), 1, nx - 1)
            tx = clamp(Float32((x - xs[ix]) / (xs[ix+1] - xs[ix])), 0.0f0, 1.0f0)
            v00 = V[ix, jy]
            v10 = V[ix+1, jy]
            v01 = V[ix, jy+1]
            v11 = V[ix+1, jy+1]
            V_fine[i, j] = (1.0f0 - tx)*(1.0f0 - uy)*v00 + tx*(1.0f0 - uy)*v10 + (1.0f0 - tx)*uy*v01 + tx*uy*v11
        end
    end
    (xs_fine, ys_fine, V_fine)
end

function run_argon_80M_simulation(; npart = 80_000_000, dt = 0.5)
    cache_file = joinpath(ROOT, "argon_converged_80M_data_cache.jls")
    if isfile(cache_file)
        println("Loading cached 80M Argon simulation run from $cache_file...")
        return deserialize(cache_file)
    end

    nions = 40.0
    nelectrons = 40.0
    rcluster = 50.0
    rbox = 150.0
    nfine = 64
    m = nfine ÷ 2 + 1
    n1 = m ÷ 2
    n2 = m - n1
    ninner = 2 * n1
    nouter = 2 * n2 - 2

    mass_ar = 40 * 1836.154
    energy_au = 80.0 * KEV
    v = sqrt(2 * energy_au / mass_ar) # v ≈ 0.2838 u.a.
    impact = 20.0
    x0 = -76.0

    p = SimulationParameters(nfine = nfine, ninner = ninner,
                             nouter = nouter, rcluster = rcluster,
                             rbox = rbox, nions = nions, nelectrons = nelectrons,
                             nparticles = npart, nsteps = 0, dt = dt)

    profile = read_radial_profile(joinpath(ROOT, "ref", "these", "hm1.Na40.dat"),
                                  joinpath(ROOT, "ref", "these", "rhoinit.Na40.dat"))

    proj = Projectile(mass = mass_ar, charge = 8.0, energy = energy_au,
                      impact = impact, x0 = x0, dt = dt,
                      softening = BallSoftening(2.0))

    println("=== STARTING 80M ARGON SIMULATION ===")
    println("Particles: $npart | Fine Grid: $nfine (Mesh $(2 * nfine + 2)³)")
    println("BLAS: $(ACCELERATE ? "AppleAccelerate" : "OpenBLAS") | GPU: $(METAL ? "Metal (MtlArray)" : "CPU") | Threads: $(Threads.nthreads())")
    flush(stdout)

    t0 = time()
    sim = Simulation(p, profile; projectile = proj)
    fine = sim.meshes[1]
    println("Cluster initialized in $(round(time() - t0, digits=1)) s.")
    flush(stdout)

    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing
    println("Metal 10.4 GB buffers ready.")
    flush(stdout)

    nsteps = 960
    record_every = 4

    xs_grid = collocation_points(fine.axes[1].knots)
    ys_grid = collocation_points(fine.axes[2].knots)
    iz1 = nfine
    iz2 = nfine + 1

    times_fs = Float64[]
    x_projs = Float64[]
    y_projs = Float64[]
    q_caps = Float64[]
    q_nets = Float64[]
    e_losses_ev = Float64[]
    e_excs_ev = Float64[]
    density_frames = Matrix{Float32}[]

    b_init = step!(sim; energy = true, accelerator = acc)
    e_cluster_0 = b_init.total
    cur_exc = 0.0

    t_start = time()
    for step in 1:nsteps
        b = step!(sim; energy = false, accelerator = acc)

        if step % record_every == 0
            t_fs = (step * dt) / FS_TO_AU
            x_p = proj.position[1]
            y_p = proj.position[2]
            loss_ev = (energy_au - kinetic_energy(proj)) * HARTREE_TO_EV
            q_cap = enclosed_charge(sim.cloud, proj, 6.0)
            n_cluster = count(pt -> sum(abs2, pt) < 16.0^2, sim.cloud.positions)
            q_net = nions - sim.cloud.weight * n_cluster

            slice2d = Float32.((sim.ρ[1][:, :, iz1] .+ sim.ρ[1][:, :, iz2]) ./ 2)

            push!(times_fs, t_fs)
            push!(x_projs, x_p)
            push!(y_projs, y_p)
            push!(q_caps, q_cap)
            push!(q_nets, q_net)
            push!(e_losses_ev, loss_ev)
            push!(e_excs_ev, cur_exc)
            push!(density_frames, copy(slice2d))

            if length(density_frames) % 20 == 0 || length(density_frames) == 1
                t_elapsed = time() - t_start
                rate = length(density_frames) / t_elapsed
                eta_min = (nsteps ÷ record_every - length(density_frames)) / rate / 60
                @printf("  frame %3d/%3d (t = %5.2f fs, x = %+5.1f a₀): Q_cap = %4.2f e, Q_net = %4.2f e, loss = %5.1f eV (ETA: %.1f min)\n",
                        length(density_frames), nsteps ÷ record_every, t_fs, x_p, q_cap, q_net, loss_ev, eta_min)
                flush(stdout)
            end
        end
    end

    b_final = step!(sim; energy = true, accelerator = acc)
    final_exc = (b_final.total - e_cluster_0) * HARTREE_TO_EV

    elapsed = time() - t_start
    @printf("80M Argon complete in %.2f s (%.1f min) with %d frames\n", elapsed, elapsed / 60, length(density_frames))
    @printf("  Final Captured Charge: %4.2f e | Net Cluster Charge: +%4.2f e | Total Loss: %5.1f eV\n",
            last(q_caps), last(q_nets), last(e_losses_ev))
    flush(stdout)

    data = (frames = density_frames, times_fs = times_fs, x_projs = x_projs, y_projs = y_projs,
            q_caps = q_caps, q_nets = q_nets, e_losses_ev = e_losses_ev, e_excs_ev = e_excs_ev,
            final_exc = final_exc, xs = xs_grid, ys = ys_grid, r_jel = 13.68, impact = impact,
            npart = npart, nfine = nfine)

    serialize(cache_file, data)
    println("Saved 80M Argon simulation cache to $cache_file")
    flush(stdout)
    data
end

function render_argon_80M_results(data, out_mp4, out_gif, out_snapshots)
    println("Rendering 80M Argon results...")
    flush(stdout)
    nframes = length(data.frames)
    xs_raw = data.xs
    ys_raw = data.ys

    xs_f, ys_f, _ = resample_2d(xs_raw, ys_raw, data.frames[1], 350, 350)
    fine_frames = Matrix{Float32}[]
    for fr in data.frames
        _, _, fr_f = resample_2d(xs_raw, ys_raw, fr, 350, 350)
        push!(fine_frames, fr_f)
    end

    rho_bulk = 0.00373f0
    c_min = 0.02f0 * rho_bulk
    c_max = 1.45f0 * rho_bulk

    # 1. 12-panel snapshot strip matching historical thesis Figure Fsnap1
    target_times = range(3.6, 11.5, length = 12)
    chosen_indices = [argmin(abs.(data.times_fs .- tt)) for tt in target_times]

    fig_snap = CairoMakie.Figure(size = (1200, 950), backgroundcolor = :white)
    CairoMakie.Label(fig_snap[0, 1:4],
        @sprintf("Snapshots of Electron Density during Na₄₀ + Ar⁸⁺ Peripheral Collision (80 keV, b = 20 a₀)\n25 Years Later: %d GPU Macro-Particles (Mesh %d, rcluster = 50 a₀)",
                 data.npart, data.nfine),
        fontsize = 18, font = :bold)

    for (k, fi) in enumerate(chosen_indices)
        row = (k - 1) ÷ 4 + 1
        col = (k - 1) % 4 + 1
        ax = CairoMakie.Axis(fig_snap[row, col],
            title = @sprintf("T = %.1f fs (Q_cap = %.1fe)", data.times_fs[fi], data.q_caps[fi]),
            aspect = CairoMakie.DataAspect(), backgroundcolor = GREY,
            xlabel = row == 3 ? "x (a₀)" : "", ylabel = col == 1 ? "y (a₀)" : "")
        CairoMakie.xlims!(ax, -50, 50); CairoMakie.ylims!(ax, -30, 45)

        fm = copy(fine_frames[fi])
        fm[fm .< c_min] .= NaN32
        CairoMakie.heatmap!(ax, xs_f, ys_f, fm,
                            colormap = :turbo, colorrange = (c_min, c_max),
                            nan_color = GREY, interpolate = true)

        θ = range(0, 2π, length = 100)
        CairoMakie.lines!(ax, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ),
                          color = :white, linestyle = :dash, linewidth = 1.5)

        CairoMakie.scatter!(ax, [data.x_projs[fi]], [data.y_projs[fi]],
                            color = :white, strokecolor = :black, strokewidth = 2, markersize = 10)
    end

    CairoMakie.save(out_snapshots, fig_snap, px_per_unit = 2)
    println("Saved 80M Argon snapshot strip to $out_snapshots")
    flush(stdout)

    # 2. Dynamic Animation (MP4 & GIF)
    fig_anim = CairoMakie.Figure(size = (800, 950), backgroundcolor = :white)
    frame_idx = CairoMakie.Observable(1)

    ax_heat = CairoMakie.Axis(fig_anim[1, 1],
        title = CairoMakie.@lift(@sprintf("Na₄₀ + Ar⁸⁺ (80 keV, b = 20 a₀) — t = %5.2f fs, x_ion = %+5.1f a₀",
                                          data.times_fs[$frame_idx], data.x_projs[$frame_idx])),
        xlabel = "x (a₀)", ylabel = "y (a₀)",
        aspect = CairoMakie.DataAspect(),
        backgroundcolor = GREY)
    CairoMakie.xlims!(ax_heat, -50, 50)
    CairoMakie.ylims!(ax_heat, -30, 45)

    cur_frame_obs = CairoMakie.@lift begin
        fm = copy(fine_frames[$frame_idx])
        fm[fm .< c_min] .= NaN32
        fm
    end
    CairoMakie.heatmap!(ax_heat, xs_f, ys_f, cur_frame_obs,
                        colormap = :turbo, colorrange = (c_min, c_max),
                        nan_color = GREY, interpolate = true)

    θ = range(0, 2π, length = 100)
    CairoMakie.lines!(ax_heat, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ),
                      color = :white, linestyle = :dash, linewidth = 2.0)

    proj_pt = CairoMakie.@lift CairoMakie.Point2f(data.x_projs[$frame_idx], data.y_projs[$frame_idx])
    CairoMakie.scatter!(ax_heat, proj_pt, color = :white, strokecolor = :black,
                        strokewidth = 2.5, markersize = 14)

    # Diagnostic plot: Captured charge and ionization
    ax_diag = CairoMakie.Axis(fig_anim[2, 1],
        title = "Real-Time Observables: Captured Charge, Cluster Ionization & Energy Loss",
        xlabel = "Time (fs)", ylabel = "Charge (e) / Energy (10 eV)",
        xgridvisible = true, ygridvisible = true)
    CairoMakie.xlims!(ax_diag, 0, data.times_fs[end])
    CairoMakie.ylims!(ax_diag, -0.5, 9.0)

    scaled_loss = data.e_losses_ev ./ 10.0
    CairoMakie.lines!(ax_diag, data.times_fs, data.q_caps, color = (:crimson, 0.3), linewidth = 1.5)
    CairoMakie.lines!(ax_diag, data.times_fs, data.q_nets, color = (:navy, 0.3), linewidth = 1.5)
    CairoMakie.lines!(ax_diag, data.times_fs, scaled_loss, color = (:darkgreen, 0.3), linewidth = 1.5)

    cur_t = CairoMakie.@lift data.times_fs[1:$frame_idx]
    cur_qc = CairoMakie.@lift data.q_caps[1:$frame_idx]
    cur_qn = CairoMakie.@lift data.q_nets[1:$frame_idx]
    cur_loss = CairoMakie.@lift scaled_loss[1:$frame_idx]

    CairoMakie.lines!(ax_diag, cur_t, cur_qc, color = :crimson, linewidth = 2.2, label = "Captured Charge Q_cap (e)")
    CairoMakie.lines!(ax_diag, cur_t, cur_qn, color = :navy, linewidth = 2.2, label = "Net Cluster Ionization Q_net (e)")
    CairoMakie.lines!(ax_diag, cur_t, cur_loss, color = :darkgreen, linewidth = 2.2, label = "Energy Loss ΔE (10 eV)")

    CairoMakie.axislegend(ax_diag, position = :lt)

    pt_q = CairoMakie.@lift [CairoMakie.Point2f(data.times_fs[$frame_idx], data.q_caps[$frame_idx])]
    CairoMakie.scatter!(ax_diag, pt_q, color = :crimson, markersize = 10)

    println("Encoding 80M Argon MP4 to $out_mp4...")
    flush(stdout)
    t0 = time()
    CairoMakie.record(fig_anim, out_mp4, 1:nframes; framerate = 20) do i
        frame_idx[] = i
    end
    t_rec = time() - t0
    @printf("MP4 recorded in %.2f s (%.1f fps)\n", t_rec, nframes / t_rec)
    flush(stdout)

    println("Converting MP4 to GIF via ffmpeg...")
    flush(stdout)
    run(`ffmpeg -y -loglevel error -i $out_mp4 -vf "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $out_gif`)
    println("80M Argon rendering complete!")
    flush(stdout)
end

function main()
    data = run_argon_80M_simulation(npart = 80_000_000, dt = 0.5)
    out_mp4 = joinpath(ROOT, "film_argon_80M.mp4")
    out_gif = joinpath(ROOT, "film_argon_80M.gif")
    out_snapshots = joinpath(ROOT, "argon_snapshots_80M.png")
    render_argon_80M_results(data, out_mp4, out_gif, out_snapshots)

    assets_dir = joinpath(ROOT, "docs", "src", "assets")
    cp(out_gif, joinpath(assets_dir, "film_argon_80M.gif"), force = true)
    cp(out_snapshots, joinpath(assets_dir, "argon_snapshots_80M.png"), force = true)
    println("Argon 80M assets copied to $assets_dir")
    flush(stdout)
end

main()
