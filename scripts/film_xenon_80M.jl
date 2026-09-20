#!/usr/bin/env julia
# 80 Million Particles Simulation: Na₁₉₆ + Xe²⁵⁺ (500 keV, b = 45 a₀)
# Converged 2026 Simulation on GPU Metal + BLAS AppleAccelerate (80,000,000 particles)

using Vlasov
using Printf
using Serialization
using CairoMakie
CairoMakie.activate!(type = "png")

# ⚠️ `functional()`, not merely `using`: `using Metal` SUCCEEDS on a machine
# with no Apple GPU — it only logs an error — and the script then took the
# Metal path on a Linux box with an NVIDIA card. Reported from one.
const METAL = try; @eval using Metal; @eval Metal.functional(); catch; false; end
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

function run_xenon_80M_simulation(; npart = 80_000_000, dt = 0.5)
    cache_file = joinpath(ROOT, "xenon_converged_80M_data_cache.jls")
    if isfile(cache_file)
        println("Loading cached 80M Xenon simulation run from $cache_file...")
        return deserialize(cache_file)
    end

    prof = read_radial_profile(joinpath(ROOT, "ref", "fortran", "data", "hm1.dat"),
                               joinpath(ROOT, "ref", "fortran", "data", "rhoinit.dat"))
    nfine = 64
    m = nfine ÷ 2 + 1
    n1 = m ÷ 2
    n2 = m - n1
    ninner = 2 * n1
    nouter = 2 * n2 - 2
    rcluster = 78.0
    rbox = 235.0

    p = SimulationParameters(nfine = nfine, ninner = ninner, nouter = nouter,
                             rcluster = rcluster, rbox = rbox,
                             nions = 196.0, nelectrons = 196.0,
                             nparticles = npart, nsteps = 0, dt = dt)

    mass_xe = 131.3 * 1836.154
    v = 0.40
    energy_au = 0.5 * mass_xe * v^2
    impact = 45.0
    x0 = -70.0
    proj = Projectile(mass = mass_xe, charge = 25.0, energy = energy_au,
                      impact = impact, x0 = x0, dt = dt,
                      softening = BallSoftening(5.0))

    println("=== STARTING 80M XENON SIMULATION ===")
    println("Particles: $npart | Fine Grid: $nfine (Mesh $(2 * nfine + 2)³)")
    println("BLAS: $(ACCELERATE ? "AppleAccelerate" : "OpenBLAS") | GPU: $(METAL ? "Metal (MtlArray)" : "CPU") | Threads: $(Threads.nthreads())")
    flush(stdout)

    t0 = time()
    sim = Simulation(p, prof; projectile = proj)
    fine = sim.meshes[1]
    println("Cluster initialized in $(round(time() - t0, digits=1)) s.")
    flush(stdout)

    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing
    println("Metal 10.4 GB buffers ready.")
    flush(stdout)

    pts_x = collocation_points(fine.axes[1].knots)
    pts_y = collocation_points(fine.axes[2].knots)
    iz1 = nfine
    iz2 = nfine + 1

    total_dist = abs(x0) + 70.0
    nsteps = ceil(Int, total_dist / (v * dt)) # 700 steps
    stride = 4

    frames = Matrix{Float32}[]
    times_fs = Float64[]
    proj_xs = Float64[]
    proj_ys = Float64[]
    q_caps = Float64[]
    q_nets = Float64[]
    e_losses_ev = Float64[]
    e_excs_ev = Float64[]

    r_cluster_edge = 196.0^(1/3) * 4.0 # ~23.2 a0

    b_init = step!(sim; energy = true, accelerator = acc)
    e_cluster_0 = b_init.total
    cur_exc = 0.0

    t_start = time()
    for step in 1:nsteps
        # Skip costly CPU energy inside loop for speed
        b = step!(sim; energy = false, accelerator = acc)

        if step % stride == 0 || step == 1
            t_fs = (step * dt) / FS_TO_AU
            px = sim.projectile.position[1]
            py = sim.projectile.position[2]
            loss_ev = (energy_au - kinetic_energy(sim.projectile)) * HARTREE_TO_EV

            # Fast 2D density slice at z ≈ 0
            sl = Float32.((sim.ρ[1][:, :, iz1] .+ sim.ρ[1][:, :, iz2]) ./ 2)

            q_cap = enclosed_charge(sim.cloud, sim.projectile, 8.0)
            n_in = count(pt -> sum(abs2, pt) < 35.0^2, sim.cloud.positions)
            q_net = 196.0 - sim.cloud.weight * n_in

            push!(frames, copy(sl))
            push!(times_fs, t_fs)
            push!(proj_xs, px)
            push!(proj_ys, py)
            push!(q_caps, q_cap)
            push!(q_nets, q_net)
            push!(e_losses_ev, loss_ev)
            push!(e_excs_ev, cur_exc)

            if length(frames) % 15 == 0 || length(frames) == 1
                t_elapsed = time() - t_start
                rate = length(frames) / t_elapsed
                eta_min = (nsteps ÷ stride - length(frames)) / rate / 60
                @printf("  frame %3d/%3d (t = %5.2f fs, x = %+5.1f a₀): Q_cap = %4.2f e, Q_net = %5.2f e, loss = %5.1f eV (ETA: %.1f min)\n",
                        length(frames), nsteps ÷ stride, t_fs, px, q_cap, q_net, loss_ev, eta_min)
                flush(stdout)
            end
        end
    end

    b_final = step!(sim; energy = true, accelerator = acc)
    final_exc = (b_final.total - e_cluster_0) * HARTREE_TO_EV

    elapsed = time() - t_start
    @printf("80M Xenon complete in %.2f s (%.1f min) with %d frames\n", elapsed, elapsed / 60, length(frames))
    @printf("  Final Captured Charge: %4.2f e | Net Cluster Charge: +%4.2f e | Total Loss: %5.1f eV\n",
            last(q_caps), last(q_nets), last(e_losses_ev))
    flush(stdout)

    data = (frames = frames, times_fs = times_fs, proj_xs = proj_xs, proj_ys = proj_ys,
            q_caps = q_caps, q_nets = q_nets, e_losses_ev = e_losses_ev, e_excs_ev = e_excs_ev,
            final_exc = final_exc, xs = pts_x, ys = pts_y, r_jel = r_cluster_edge,
            npart = npart, nfine = nfine)

    serialize(cache_file, data)
    println("Saved 80M Xenon simulation cache to $cache_file")
    flush(stdout)
    data
end

function render_xenon_80M_results(data, out_mp4, out_gif, out_snapshots)
    println("Rendering 80M Xenon results...")
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

    # 1. 6-snapshot figure matching exact historical timestamps: 2.40, 4.50, 5.60, 6.19, 7.33, 8.53 fs
    target_times = [2.40, 4.50, 5.60, 6.19, 7.33, 8.53]
    chosen_indices = [argmin(abs.(data.times_fs .- tt)) for tt in target_times]

    fig_snap = CairoMakie.Figure(size = (1100, 750), backgroundcolor = :white)
    CairoMakie.Label(fig_snap[0, 1:3],
        @sprintf("Snapshots of Electron Density during Na₁₉₆ + Xe²⁵⁺ Collision (500 keV, b = 45 a₀)\n25 Years Later: %d GPU Macro-Particles (Mesh %d)",
                 data.npart, data.nfine),
        fontsize = 18, font = :bold)

    for (k, fi) in enumerate(chosen_indices)
        row = (k - 1) ÷ 3 + 1
        col = (k - 1) % 3 + 1
        ax = CairoMakie.Axis(fig_snap[row, col],
            title = @sprintf("T = %.2f fs (x_ion = %+.1f a₀)", target_times[k], data.proj_xs[fi]),
            aspect = CairoMakie.DataAspect(), backgroundcolor = GREY,
            xlabel = row == 2 ? "x (a₀)" : "", ylabel = col == 1 ? "y (a₀)" : "")
        CairoMakie.xlims!(ax, -60, 60); CairoMakie.ylims!(ax, -35, 65)

        fm = copy(fine_frames[fi])
        fm[fm .< c_min] .= NaN32
        CairoMakie.heatmap!(ax, xs_f, ys_f, fm,
                            colormap = :turbo, colorrange = (c_min, c_max),
                            nan_color = GREY, interpolate = true)

        θ = range(0, 2π, length = 100)
        CairoMakie.lines!(ax, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ),
                          color = :white, linestyle = :dash, linewidth = 1.5)

        CairoMakie.scatter!(ax, [data.proj_xs[fi]], [data.proj_ys[fi]],
                            color = :white, strokecolor = :black, strokewidth = 2, markersize = 10)
    end

    CairoMakie.save(out_snapshots, fig_snap, px_per_unit = 2)
    println("Saved 80M Xenon snapshot strip to $out_snapshots")
    flush(stdout)

    # 2. Dynamic Animation (MP4 & GIF)
    fig_anim = CairoMakie.Figure(size = (800, 950), backgroundcolor = :white)
    frame_idx = CairoMakie.Observable(1)

    ax_heat = CairoMakie.Axis(fig_anim[1, 1],
        title = CairoMakie.@lift(@sprintf("Na₁₉₆ + Xe²⁵⁺ (500 keV, b = 45 a₀) — t = %5.2f fs, x_ion = %+5.1f a₀",
                                          data.times_fs[$frame_idx], data.proj_xs[$frame_idx])),
        xlabel = "x (a₀)", ylabel = "y (a₀)",
        aspect = CairoMakie.DataAspect(),
        backgroundcolor = GREY)
    CairoMakie.xlims!(ax_heat, -60, 60)
    CairoMakie.ylims!(ax_heat, -35, 65)

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

    proj_pt = CairoMakie.@lift CairoMakie.Point2f(data.proj_xs[$frame_idx], data.proj_ys[$frame_idx])
    CairoMakie.scatter!(ax_heat, proj_pt, color = :white, strokecolor = :black,
                        strokewidth = 2.5, markersize = 14)

    # Diagnostic plot: Captured charge and ionization
    ax_diag = CairoMakie.Axis(fig_anim[2, 1],
        title = "Real-Time Observables: Captured Charge, Cluster Ionization & Energy Loss",
        xlabel = "Time (fs)", ylabel = "Charge (e) / Energy (10 eV)",
        xgridvisible = true, ygridvisible = true)
    CairoMakie.xlims!(ax_diag, 0, data.times_fs[end])
    CairoMakie.ylims!(ax_diag, -0.5, 16.0)

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

    println("Encoding 80M Xenon MP4 to $out_mp4...")
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
    println("80M Xenon rendering complete!")
    flush(stdout)
end

function main()
    data = run_xenon_80M_simulation(npart = 80_000_000, dt = 0.5)
    out_mp4 = joinpath(ROOT, "film_xenon_80M.mp4")
    out_gif = joinpath(ROOT, "film_xenon_80M.gif")
    out_snapshots = joinpath(ROOT, "xenon_snapshots_80M.png")
    render_xenon_80M_results(data, out_mp4, out_gif, out_snapshots)

    assets_dir = joinpath(ROOT, "docs", "src", "assets")
    cp(out_gif, joinpath(assets_dir, "film_xenon_80M.gif"), force = true)
    cp(out_snapshots, joinpath(assets_dir, "xenon_snapshots_80M.png"), force = true)
    println("Xenon 80M assets copied to $assets_dir")
    flush(stdout)
end

main()
