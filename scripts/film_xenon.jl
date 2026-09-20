#!/usr/bin/env julia
"""
Film & Snapshots of Na₁₉₆ + Xe²⁵⁺ Peripheral Collision (500 keV, b = 45 a₀)
Matching the 1997 Springer study ("Dynamics of clusters in collision with multicharged ions").

High-resolution simulation on Metal GPU with smooth contour rendering:
- Fine mesh nfine = 64 for high spatial fidelity
- 2D field bilinear upsampling (350×350)
- Iso-contour band rendering (contourf!) eliminating pixelation artifacts
- Dynamic real-time dual-panel video (MP4 + GIF) and 6-panel snapshot figure
"""

using Vlasov
using Printf
using Serialization
using CairoMakie
CairoMakie.activate!(type = "png")

# ⚠️ `functional()`, not merely `using`: `using Metal` SUCCEEDS on a machine
# with no Apple GPU — it only logs an error — and the script then took the
# Metal path on a Linux box with an NVIDIA card. Reported from one.
const METAL = try; @eval using Metal; @eval Metal.functional(); catch; false; end
const CUDA_OK = METAL ? false :
                try; @eval using CUDA; @eval CUDA.functional(); catch; false; end
const GPU = METAL || CUDA_OK

"""The device this machine offers, or `nothing`.

⚠️ **The resident path, not `ForceAccelerator(MtlArray, …)`.** That constructor
lives in the Metal extension and exists for no other vendor, so the film used to
be Apple-only by construction. `Simulation(…; backend)` takes any
`KernelAbstractions` backend and keeps the cloud on the device — which is both
portable and the faster of the two routes."""
device_backend() = METAL ? @eval(MetalBackend()) :
                   CUDA_OK ? @eval(CUDABackend()) : nothing
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end

const ROOT = dirname(@__DIR__)
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

function run_xenon_simulation(; npart = 600_000, dt = 0.5)
    cache_file = joinpath(ROOT, "xenon_data_cache.jls")
    if isfile(cache_file)
        println("Loading cached simulation run from $cache_file...")
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
    energy = 0.5 * mass_xe * v^2
    impact = 45.0
    x0 = -70.0
    proj = Projectile(mass = mass_xe, charge = 25.0, energy = energy,
                      impact = impact, x0 = x0, dt = dt,
                      softening = BallSoftening(5.0))

    bk = device_backend()
    sim = bk === nothing ?
          Simulation(p, prof; projectile = proj) :
          Simulation(p, prof; projectile = proj, backend = bk,
                     precision = Float32, packed = true)
    fine = sim.meshes[1]

    pts_x = collocation_points(fine.axes[1].knots)
    pts_y = collocation_points(fine.axes[2].knots)

    # Reaction plane indices (z ≈ 0)
    iz1 = nfine
    iz2 = nfine + 1

    total_dist = abs(x0) + 70.0
    nsteps = ceil(Int, total_dist / (v * dt))
    stride = 4

    frames = Matrix{Float32}[]
    times_fs = Float64[]
    proj_xs = Float64[]
    proj_ys = Float64[]
    q_caps = Float64[]
    q_nets = Float64[]

    r_cluster_edge = 196.0^(1/3) * 4.0

    @printf("Simulating Na₁₉₆ + Xe²⁵⁺ (500 keV, b = 45 a₀, nfine = %d) on %s: %d steps...\n",
            nfine, METAL ? "GPU (Metal)" : CUDA_OK ? "GPU (CUDA)" : "CPU", nsteps)
    t_start = time()

    for step in 1:nsteps
        step!(sim; energy = false)

        if step % stride == 0 || step == 1
            # ⚠️ Everything below reads the simulation **on the host**: the
            # density grid, which a resident run keeps on the device, and the
            # cloud, whose host half is a different array on a discrete GPU.
            # Both are no-ops on unified memory, and neither is in the hot loop.
            if bk !== nothing
                sync_host!(sim)
                Vlasov.download!(sim.device.accelerator.particles, sim.device.backend)
            end
            t_fs = (step * dt) / FS_TO_AU
            px = sim.projectile.position[1]
            py = sim.projectile.position[2]

            # 2D density slice at z ≈ 0
            sl = Float32.((sim.ρ[1][:, :, iz1] .+ sim.ρ[1][:, :, iz2]) ./ 2)

            # Capture diagnostic: charge within R = 8 a0 of Xe ion
            q_cap = enclosed_charge(sim.cloud, sim.projectile, 8.0)

            # Cluster net ionization (loss of electrons from R < 35 a0)
            n_in = count(pt -> sum(abs2, pt) < 35.0^2, sim.cloud.positions)
            q_net = 196.0 - sim.cloud.weight * n_in

            push!(frames, copy(sl))
            push!(times_fs, t_fs)
            push!(proj_xs, px)
            push!(proj_ys, py)
            push!(q_caps, q_cap)
            push!(q_nets, q_net)

            if length(frames) % 20 == 0
                @printf("  frame %3d/%3d (t = %5.2f fs, x = %+5.1f a₀): Q_cap = %4.2f e, Q_net = %5.2f e\n",
                        length(frames), nsteps ÷ stride, t_fs, px, q_cap, q_net)
                flush(stdout)
            end
        end
    end

    elapsed = time() - t_start
    @printf("Simulation complete in %.2f s (%d frames captured)\n", elapsed, length(frames))

    data = (frames = frames, times_fs = times_fs, proj_xs = proj_xs, proj_ys = proj_ys,
            q_caps = q_caps, q_nets = q_nets, xs = pts_x, ys = pts_y, r_jel = r_cluster_edge)

    serialize(cache_file, data)
    println("Saved simulation run to $cache_file")
    data
end

function render_smooth_movie_and_snapshots(data, out_mp4, out_gif, out_snapshots)
    println("Pre-resampling frames to 350×350 smooth grid...")
    nframes = length(data.frames)
    xs_raw = data.xs
    ys_raw = data.ys

    # Upsample the first frame to determine fine grid
    xs_f, ys_f, _ = resample_2d(xs_raw, ys_raw, data.frames[1], 350, 350)
    fine_frames = Matrix{Float32}[]
    for (fi, fr) in enumerate(data.frames)
        _, _, fr_f = resample_2d(xs_raw, ys_raw, fr, 350, 350)
        push!(fine_frames, fr_f)
    end
    println("Resampling complete ($nframes frames).")

    rho_bulk = 0.00373f0
    c_min = 0.02f0 * rho_bulk
    c_max = 1.45f0 * rho_bulk
    n_levels = 45
    levels = range(c_min, c_max, length = n_levels)

    # -------------------------------------------------------------
    # 1. Generate the 6-snapshot figure (matching Springer 1997)
    # -------------------------------------------------------------
    target_times = [1.2, 2.5, 3.8, 4.8, 6.0, 7.5]
    chosen_indices = Int[]
    for tt in target_times
        idx = argmin(abs.(data.times_fs .- tt))
        push!(chosen_indices, idx)
    end

    fig_snap = CairoMakie.Figure(size = (1100, 750), backgroundcolor = :white)
    CairoMakie.Label(fig_snap[0, 1:3],
        "Snapshots of Electron Density during Na₁₉₆ + Xe²⁵⁺ Collision (500 keV, b = 45 a₀)\n(Smooth Iso-Contour Representation)",
        fontsize = 18, font = :bold)

    for (k, fi) in enumerate(chosen_indices)
        row = (k - 1) ÷ 3 + 1
        col = (k - 1) % 3 + 1
        ax = CairoMakie.Axis(fig_snap[row, col],
            title = @sprintf("t = %.2f fs (x_ion = %+.1f a₀)", data.times_fs[fi], data.proj_xs[fi]),
            aspect = CairoMakie.DataAspect(), backgroundcolor = GREY,
            xlabel = row == 2 ? "x (a₀)" : "", ylabel = col == 1 ? "y (a₀)" : "")

        CairoMakie.contourf!(ax, xs_f, ys_f, fine_frames[fi],
                            levels = levels, colormap = :turbo, extendlow = GREY)

        # Draw cluster jellium edge
        θ = range(0, 2π, length = 100)
        CairoMakie.lines!(ax, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ),
            color = :white, linewidth = 1.5, linestyle = :dash)

        # Draw Xe25+ projectile
        CairoMakie.scatter!(ax, [data.proj_xs[fi]], [data.proj_ys[fi]],
            color = :yellow, strokecolor = :black, strokewidth = 2, markersize = 12)

        CairoMakie.xlims!(ax, -65, 65)
        CairoMakie.ylims!(ax, -35, 65)
    end

    CairoMakie.save(out_snapshots, fig_snap, px_per_unit = 2)
    println("Saved smooth snapshot strip to $out_snapshots")

    # -------------------------------------------------------------
    # 2. Render Animated Movie (MP4 + GIF)
    # -------------------------------------------------------------
    fig_anim = CairoMakie.Figure(size = (900, 850), backgroundcolor = :white)
    idx = CairoMakie.Observable(1)

    # Upper panel: 2D Electron density slice
    ax1 = CairoMakie.Axis(fig_anim[1, 1], aspect = CairoMakie.DataAspect(), backgroundcolor = GREY,
        title = CairoMakie.@lift(@sprintf("Na₁₉₆ + Xe²⁵⁺ (500 keV, b = 45 a₀) — t = %.2f fs, x_ion = %+.1f a₀",
                                          data.times_fs[$idx], data.proj_xs[$idx])),
        xlabel = "x (a₀)", ylabel = "y (a₀)")

    cur_slice_obs = CairoMakie.@lift fine_frames[$idx]

    CairoMakie.contourf!(ax1, xs_f, ys_f, cur_slice_obs,
                         levels = levels, colormap = :turbo, extendlow = GREY)

    # Cluster jellium boundary
    θ = range(0, 2π, length = 100)
    CairoMakie.lines!(ax1, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ),
        color = :white, linewidth = 2.0, linestyle = :dash)

    # Moving Xe25+ Projectile marker
    proj_pt = CairoMakie.@lift CairoMakie.Point2f(data.proj_xs[$idx], data.proj_ys[$idx])
    CairoMakie.scatter!(ax1, proj_pt, color = :yellow, strokecolor = :black,
                        strokewidth = 2, markersize = 14)
    CairoMakie.xlims!(ax1, -65, 65)
    CairoMakie.ylims!(ax1, -35, 65)

    # Lower panel: Real-time charge transfer curves
    ax2 = CairoMakie.Axis(fig_anim[2, 1],
        title = "Dynamic Charge Transfer: Hollow Atom Capture & Cluster Ionization",
        xlabel = "Time t (fs)", ylabel = "Charge (e)",
        xgridvisible = true, ygridvisible = true)

    CairoMakie.lines!(ax2, data.times_fs, data.q_caps, color = (:dodgerblue, 0.4), linewidth = 1.5)
    CairoMakie.lines!(ax2, data.times_fs, data.q_nets, color = (:darkorange, 0.4), linewidth = 1.5)

    cur_t = CairoMakie.@lift data.times_fs[1:$idx]
    cur_qcap = CairoMakie.@lift data.q_caps[1:$idx]
    cur_qnet = CairoMakie.@lift data.q_nets[1:$idx]

    CairoMakie.lines!(ax2, cur_t, cur_qcap, color = :dodgerblue, linewidth = 2.5,
                      label = "Captured Charge Q_cap (Hollow Atom, R = 8 a₀)")
    CairoMakie.lines!(ax2, cur_t, cur_qnet, color = :darkorange, linewidth = 2.5,
                      label = "Net Cluster Ionization Q_cluster")

    cur_pt_cap = CairoMakie.@lift [CairoMakie.Point2f(data.times_fs[$idx], data.q_caps[$idx])]
    cur_pt_net = CairoMakie.@lift [CairoMakie.Point2f(data.times_fs[$idx], data.q_nets[$idx])]
    CairoMakie.scatter!(ax2, cur_pt_cap, color = :dodgerblue, markersize = 10)
    CairoMakie.scatter!(ax2, cur_pt_net, color = :darkorange, markersize = 10)

    CairoMakie.xlims!(ax2, 0, data.times_fs[end])
    CairoMakie.ylims!(ax2, -0.5, 14.5)
    CairoMakie.axislegend(ax2, position = :lt)

    println("Encoding MP4 to $out_mp4...")
    t_enc = time()
    CairoMakie.record(fig_anim, out_mp4, 1:nframes; framerate = 25) do i
        idx[] = i
    end
    @printf("Encoded %d frames in %.1f s (%.1f frames/s)\n",
            nframes, time() - t_enc, nframes / (time() - t_enc))

    println("Encoding GIF to $out_gif...")
    gif_indices = 1:2:nframes
    CairoMakie.record(fig_anim, out_gif, gif_indices; framerate = 15) do i
        idx[] = i
    end

    println("Smooth animations and snapshots generated successfully!")
end

function main()
    out_mp4 = joinpath(ROOT, "film_xenon.mp4")
    out_gif = joinpath(ROOT, "film_xenon.gif")
    out_snapshots = joinpath(ROOT, "xenon_snapshots.png")

    data = run_xenon_simulation(npart = 600_000)
    render_smooth_movie_and_snapshots(data, out_mp4, out_gif, out_snapshots)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
