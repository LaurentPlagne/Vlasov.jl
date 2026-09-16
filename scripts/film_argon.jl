#!/usr/bin/env julia
"""
Film & Snapshots of Na₄₀ + Ar⁸⁺ Peripheral Collision (80 keV, b = 20 a₀)
Matching Chapter 4 (Figure Fsnap1) of the 1998 PhD thesis.

High-resolution simulation on Metal GPU with smooth contour rendering:
- 2D field bilinear upsampling (350×350)
- Iso-contour band rendering (contourf!) eliminating pixelation artifacts
- 12-panel snapshot strip at exact thesis timestamps (T = 3.6 to 11.5 fs)
- Dynamic real-time dual-panel video (MP4 + GIF)
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
const FS_TO_AU = 41.34137
const GREY = RGBf(0.80, 0.80, 0.80)

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

function run_argon_simulation(; npart = 600_000, dt = 0.5)
    cache_file = joinpath(ROOT, "argon_data_cache.jls")
    if isfile(cache_file)
        println("Loading cached Argon simulation run from $cache_file...")
        return deserialize(cache_file)
    end

    # Na40 cluster: expand fine grid to rcluster = 50.0 so ion (b = 20 a0) and hollow atom stay inside
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

    # Ar8+ projectile: mass ≈ 40 * 1836.154, charge = 8.0, energy = 80 keV
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

    sim = Simulation(p, profile; projectile = proj)
    fine = sim.meshes[1]
    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing

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
    density_frames = Matrix{Float32}[]

    println("Simulating Na₄₀ + Ar⁸⁺ (80 keV, b = 20 a₀): $nsteps steps on GPU Metal...")
    t_start = time()

    for step in 1:nsteps
        step!(sim; energy = false, accelerator = acc)
        t_fs = (step * dt) / FS_TO_AU

        if step % record_every == 0
            x_p = proj.position[1]
            y_p = proj.position[2]
            q_cap = enclosed_charge(sim.cloud, proj, 6.0)
            n_cluster = count(pt -> sum(abs2, pt) < 16.0^2, sim.cloud.positions)
            q_net = nions - sim.cloud.weight * n_cluster

            slice2d = Float32.((sim.ρ[1][:, :, iz1] .+ sim.ρ[1][:, :, iz2]) ./ 2)

            push!(times_fs, t_fs)
            push!(x_projs, x_p)
            push!(y_projs, y_p)
            push!(q_caps, q_cap)
            push!(q_nets, q_net)
            push!(density_frames, copy(slice2d))

            if length(density_frames) % 20 == 0
                @printf("  frame %3d/%3d (t = %5.2f fs, x = %+5.1f a₀): Q_cap = %4.2f e, Q_net = %4.2f e\n",
                        length(density_frames), nsteps ÷ record_every, t_fs, x_p, q_cap, q_net)
                flush(stdout)
            end
        end
    end

    elapsed = time() - t_start
    @printf("Simulation complete in %.2f s (%d frames captured)\n", elapsed, length(density_frames))

    data = (frames = density_frames, times_fs = times_fs, x_projs = x_projs, y_projs = y_projs,
            q_caps = q_caps, q_nets = q_nets, xs = xs_grid, ys = ys_grid, r_jel = 13.68, impact = impact)

    serialize(cache_file, data)
    println("Saved Argon simulation run to $cache_file")
    data
end

function render_smooth_argon(data, out_mp4, out_gif, out_snapshots)
    println("Pre-resampling Argon frames to 350×350 smooth grid...")
    nframes = length(data.frames)
    xs_raw = data.xs
    ys_raw = data.ys

    xs_f, ys_f, _ = resample_2d(xs_raw, ys_raw, data.frames[1], 350, 350)
    fine_frames = Matrix{Float32}[]
    for fr in data.frames
        _, _, fr_f = resample_2d(xs_raw, ys_raw, fr, 350, 350)
        push!(fine_frames, fr_f)
    end
    println("Argon resampling complete.")

    n0 = 0.00373f0
    c_min = 0.02f0 * n0
    c_max = 1.45f0 * n0
    levels = range(c_min, c_max, length = 45)

    # 1. 12-Panel Snapshots matching Fsnap1
    target_times_fs = [3.6, 4.3, 5.0, 5.8, 6.5, 7.2, 7.9, 8.6, 9.4, 10.0, 10.8, 11.5]

    fig_snap = CairoMakie.Figure(size = (1100, 1400), backgroundcolor = :white)
    CairoMakie.Label(fig_snap[0, 1:3],
          "Snapshots of Electron Density during Na₄₀ + Ar⁸⁺ Collision (80 keV, b = 20 a₀)\n(Smooth Iso-Contour Representation — Thesis Chapter 4, Fig. 4.1)",
          fontsize = 18, font = :bold)

    for (i, target) in enumerate(target_times_fs)
        row = (i - 1) ÷ 3 + 1
        col = (i - 1) % 3 + 1
        idx = argmin(abs.(data.times_fs .- target))
        x_p = data.x_projs[idx]

        ax = CairoMakie.Axis(fig_snap[row, col],
                  title = @sprintf("T = %.1f fs (x_ion = %+.1f a₀)", target, x_p),
                  xlabel = row == 4 ? "x (a₀)" : "",
                  ylabel = col == 1 ? "y (a₀)" : "",
                  aspect = CairoMakie.DataAspect(),
                  backgroundcolor = GREY)
        CairoMakie.xlims!(ax, -35, 45)
        CairoMakie.ylims!(ax, -25, 35)

        CairoMakie.contourf!(ax, xs_f, ys_f, fine_frames[idx],
                            levels = levels, colormap = :turbo, extendlow = GREY)

        θ = range(0, 2π, length = 100)
        CairoMakie.lines!(ax, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ),
                          color = :white, linestyle = :dash, linewidth = 1.5)

        CairoMakie.scatter!(ax, [x_p], [data.impact], color = :yellow, strokecolor = :black,
                            strokewidth = 2, markersize = 10)
    end

    CairoMakie.save(out_snapshots, fig_snap, px_per_unit = 2)
    println("Saved smooth Argon snapshot strip to $out_snapshots")

    # 2. Render Animation (MP4 & GIF)
    fig_anim = CairoMakie.Figure(size = (800, 950), backgroundcolor = :white)
    frame_idx = CairoMakie.Observable(1)

    ax_heat = CairoMakie.Axis(fig_anim[1, 1],
                   title = CairoMakie.@lift(@sprintf("Na₄₀ + Ar⁸⁺ (80 keV, b = 20 a₀) — t = %5.2f fs, x_ion = %+5.1f a₀",
                                                     data.times_fs[$frame_idx], data.x_projs[$frame_idx])),
                   xlabel = "x (a₀)", ylabel = "y (a₀)",
                   aspect = CairoMakie.DataAspect(),
                   backgroundcolor = GREY)
    CairoMakie.xlims!(ax_heat, -40, 50)
    CairoMakie.ylims!(ax_heat, -30, 40)

    cur_frame_obs = CairoMakie.@lift fine_frames[$frame_idx]

    CairoMakie.contourf!(ax_heat, xs_f, ys_f, cur_frame_obs,
                         levels = levels, colormap = :turbo, extendlow = GREY)

    θ = range(0, 2π, length = 100)
    CairoMakie.lines!(ax_heat, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ),
                      color = :white, linestyle = :dash, linewidth = 2.0)

    proj_pt = CairoMakie.@lift CairoMakie.Point2f(data.x_projs[$frame_idx], data.y_projs[$frame_idx])
    CairoMakie.scatter!(ax_heat, proj_pt, color = :yellow, strokecolor = :black,
                        strokewidth = 2.5, markersize = 14)

    # Diagnostic plot
    ax_diag = CairoMakie.Axis(fig_anim[2, 1],
                   title = "Real-Time Charge Dynamics",
                   xlabel = "Time (fs)", ylabel = "Charge (e)",
                   xgridvisible = true, ygridvisible = true)
    CairoMakie.xlims!(ax_diag, 0, data.times_fs[end])
    CairoMakie.ylims!(ax_diag, -0.5, 7.5)

    CairoMakie.lines!(ax_diag, data.times_fs, data.q_nets, color = (:darkorange, 0.4), linewidth = 1.5)
    CairoMakie.lines!(ax_diag, data.times_fs, data.q_caps, color = (:crimson, 0.4), linewidth = 1.5)

    cur_t = CairoMakie.@lift data.times_fs[1:$frame_idx]
    cur_qnet = CairoMakie.@lift data.q_nets[1:$frame_idx]
    cur_qcap = CairoMakie.@lift data.q_caps[1:$frame_idx]

    CairoMakie.lines!(ax_diag, cur_t, cur_qnet, color = :darkorange, linewidth = 2.5, label = "Cluster Net Charge Q_net")
    CairoMakie.lines!(ax_diag, cur_t, cur_qcap, color = :crimson, linewidth = 2.5, label = "Captured Charge Q_cap (R = 6 a₀)")
    CairoMakie.vlines!(ax_diag, [data.times_fs[argmin(abs.(data.x_projs))]], color = (:gray, 0.5), linestyle = :dash, label = "Closest Approach (x = 0)")
    CairoMakie.axislegend(ax_diag, position = :lt)

    cur_pt_cap = CairoMakie.@lift [CairoMakie.Point2f(data.times_fs[$frame_idx], data.q_caps[$frame_idx])]
    cur_pt_net = CairoMakie.@lift [CairoMakie.Point2f(data.times_fs[$frame_idx], data.q_nets[$frame_idx])]
    CairoMakie.scatter!(ax_diag, cur_pt_cap, color = :crimson, markersize = 10)
    CairoMakie.scatter!(ax_diag, cur_pt_net, color = :darkorange, markersize = 10)

    println("Encoding MP4 to $out_mp4...")
    CairoMakie.record(fig_anim, out_mp4, 1:nframes; framerate = 25) do i
        frame_idx[] = i
    end

    println("Encoding GIF to $out_gif...")
    gif_indices = 1:2:nframes
    CairoMakie.record(fig_anim, out_gif, gif_indices; framerate = 15) do i
        frame_idx[] = i
    end

    println("Smooth Argon animations rendered successfully!")
end

function main()
    out_mp4 = joinpath(ROOT, "film_argon.mp4")
    out_gif = joinpath(ROOT, "film_argon.gif")
    out_snapshots = joinpath(ROOT, "argon_snapshots.png")

    data = run_argon_simulation(npart = 600_000)
    render_smooth_argon(data, out_mp4, out_gif, out_snapshots)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
