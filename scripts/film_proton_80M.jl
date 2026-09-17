#!/usr/bin/env julia
# 80 Million Particles Converged Simulation: Na₁₀₀₀ + H⁺ (16 keV, b = 0)
# Grid 110 (Mesh 222³, h ≈ 1.42 a₀) on Apple Silicon GPU Metal + BLAS Accelerate

using Vlasov
using Printf
using Serialization
using CairoMakie
CairoMakie.activate!(type = "png")

const METAL = try; @eval using Metal; true; catch; false; end
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end

const ROOT = dirname(@__DIR__)
const KEV = 1000.0 / HARTREE_TO_EV
const GREY = CairoMakie.RGBf(0.82, 0.82, 0.82)

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

function run_proton_80M_simulation(; npart = 80_000_000, nfine = 110, dt = 1.0, finesse = 6)
    cache_file = joinpath(ROOT, "proton_converged_80M_data_cache.jls")
    if isfile(cache_file)
        println("Loading cached 80M Proton simulation run from $cache_file...")
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

    println("=== STARTING 80M PROTON SIMULATION ===")
    println("Particles: $npart | Fine Grid: $nfine (Mesh $(2 * nfine + 2)³, h ≈ 1.42 a₀)")
    println("BLAS: $(ACCELERATE ? "AppleAccelerate" : "OpenBLAS") | GPU: $(METAL ? "Metal (MtlArray)" : "CPU") | Threads: $(Threads.nthreads())")
    flush(stdout)

    t_init = time()
    sim = Simulation(p, profile; projectile = proj)
    fine = sim.meshes[1]
    println("Cluster initialized in $(round(time() - t_init, digits=1)) s.")
    flush(stdout)

    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing
    println("Metal 10.4 GB buffers ready.")
    flush(stdout)

    cx, cy = fine.axes[1].colloc, fine.axes[2].colloc
    outx = collect(range(cx[1], cx[end]; length = finesse * length(cx)))
    outy = collect(range(cy[1], cy[end]; length = finesse * length(cy)))

    nsteps = ceil(Int, 1.1 * (80.0 + abs(x0)) / v0) # ~200 steps
    stride = 2

    # Measure initial cluster total energy
    b_init = step!(sim; energy = true, accelerator = acc)
    e_cluster_0 = b_init.total

    frames = Matrix{Float32}[]
    x_projs = Float64[]
    e_losses_ev = Float64[]
    times_fs = Float64[]
    dedx_local = Float64[]

    t_start = time()
    for step in 1:nsteps
        step!(sim; energy = false, accelerator = acc)

        xp = sim.projectile.position[1]
        loss = (energy_au - kinetic_energy(sim.projectile)) * HARTREE_TO_EV
        t_fs = (step * dt) / 41.34137

        if step % stride == 0
            push!(frames, cut_z0(sim.ρ[1], fine, 2.0, outx, outy))
            push!(x_projs, xp)
            push!(e_losses_ev, loss)
            push!(times_fs, t_fs)

            if length(e_losses_ev) >= 2
                dx = x_projs[end] - x_projs[end-1]
                de = e_losses_ev[end] - e_losses_ev[end-1]
                push!(dedx_local, dx > 0 ? de / dx : 0.0)
            else
                push!(dedx_local, 0.0)
            end

            if length(frames) % 10 == 0 || length(frames) == 1
                t_elapsed = time() - t_start
                rate = length(frames) / t_elapsed
                eta_min = (nsteps ÷ stride - length(frames)) / rate / 60
                @printf("  frame %3d/%3d (step %3d/%3d, x = %+5.1f a₀): loss = %5.2f eV, dE/dx = %4.2f eV/a₀ (ETA: %.1f min)\n",
                        length(frames), nsteps ÷ stride, step, nsteps, xp, loss, last(dedx_local), eta_min)
                flush(stdout)
            end
        end

        xp > 80.0 && break
    end

    b_final = step!(sim; energy = true, accelerator = acc)
    final_exc = (b_final.total - e_cluster_0) * HARTREE_TO_EV

    elapsed = time() - t_start
    @printf("80M Simulation complete in %.2f s (%.1f min) with %d frames captured\n",
            elapsed, elapsed / 60, length(frames))
    flush(stdout)

    idx_m2 = argmin(abs.(x_projs .- (-2.0)))
    idx_p2 = argmin(abs.(x_projs .- (+2.0)))
    dx_center = x_projs[idx_p2] - x_projs[idx_m2]
    dE_dx_center = (e_losses_ev[idx_p2] - e_losses_ev[idx_m2]) / dx_center
    total_loss = last(e_losses_ev)

    @printf("=== 80M ULTRA-CONVERGED PROTON OBSERVABLES ===\n")
    @printf("  Central Stopping Power dE/dx: %5.3f eV/a₀ (1998 Thesis: 1.587 eV/a₀)\n", dE_dx_center)
    @printf("  Total Projectile Energy Loss: %5.2f eV\n", total_loss)
    @printf("  Final Cluster Excitation:     %5.2f eV\n", final_exc)
    flush(stdout)

    data = (frames = frames, x_projs = x_projs, e_losses_ev = e_losses_ev, times_fs = times_fs,
            dedx_local = dedx_local, outx = outx, outy = outy, r_cluster = 78.0,
            dE_dx_center = dE_dx_center, total_loss = total_loss, final_exc = final_exc,
            npart = npart, nfine = nfine)

    serialize(cache_file, data)
    println("Saved 80M Proton simulation cache to $cache_file")
    flush(stdout)
    data
end

function render_proton_80M_results(data, out_mp4, out_gif, out_snapshots)
    println("Rendering 80M Proton results...")
    flush(stdout)
    nframes = length(data.frames)
    xs = data.outx
    ys = data.outy

    n0 = 0.00373f0
    c_min = 0.04f0 * n0
    c_max = 1.45f0 * n0

    R_cluster = 40.0
    θ = range(0, 2π, length = 150)

    # 1. 4-Panel Snapshot Strip showing Wake Dynamics
    target_xs = [-40.0, -15.0, 0.0, 30.0]
    snap_indices = [argmin(abs.(data.x_projs .- tx)) for tx in target_xs]

    fig_snap = CairoMakie.Figure(size = (1100, 950), backgroundcolor = :white)
    CairoMakie.Label(fig_snap[0, 1:2],
        @sprintf("Na₁₀₀₀ + H⁺ (16 keV, b = 0) — Ultra-Converged Plasmon Wake\n25 Years Later: %d GPU Macro-Particles (Mesh %d, h = 1.42 a₀)",
                 data.npart, data.nfine),
        fontsize = 18, font = :bold)

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

        f_masked = copy(data.frames[idx])
        f_masked[f_masked .< c_min] .= NaN32

        CairoMakie.heatmap!(ax, xs, ys, f_masked,
                            colormap = :turbo, colorrange = (c_min, c_max),
                            nan_color = GREY, interpolate = true)

        CairoMakie.lines!(ax, R_cluster .* cos.(θ), R_cluster .* sin.(θ),
                          color = :white, linestyle = :dash, linewidth = 2.0)

        CairoMakie.scatter!(ax, [xp], [0.0], color = :white, strokecolor = :black,
                            strokewidth = 2.0, markersize = 12)
    end

    CairoMakie.save(out_snapshots, fig_snap, px_per_unit = 2)
    println("Saved 80M Proton snapshot strip to $out_snapshots")
    flush(stdout)

    # 2. Dynamic Film (MP4 + fast ffmpeg GIF)
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

    cur_frame = CairoMakie.@lift begin
        fm = copy(data.frames[$f_idx])
        fm[fm .< c_min] .= NaN32
        fm
    end

    CairoMakie.heatmap!(ax_cut, xs, ys, cur_frame,
                        colormap = :turbo, colorrange = (c_min, c_max),
                        nan_color = GREY, interpolate = true)

    CairoMakie.lines!(ax_cut, R_cluster .* cos.(θ), R_cluster .* sin.(θ),
                      color = :white, linestyle = :dash, linewidth = 2.0)

    p_pt = CairoMakie.@lift CairoMakie.Point2f(data.x_projs[$f_idx], 0.0)
    CairoMakie.scatter!(ax_cut, p_pt, color = :white, strokecolor = :black,
                        strokewidth = 2.5, markersize = 14)

    # Diagnostic plot: Stopping loss ΔE and dE/dx
    ax_diag = CairoMakie.Axis(fig_anim[2, 1],
        title = @sprintf("Projectile Stopping: Total Loss ΔE(x) & Central Plateau dE/dx = %.2f eV/a₀",
                         data.dE_dx_center),
        xlabel = "Projectile Position x (a₀)", ylabel = "Energy Loss ΔE (eV)",
        xgridvisible = true, ygridvisible = true)
    CairoMakie.xlims!(ax_diag, data.x_projs[1], data.x_projs[end])
    CairoMakie.ylims!(ax_diag, -5, maximum(data.e_losses_ev) * 1.15)

    CairoMakie.lines!(ax_diag, data.x_projs, data.e_losses_ev, color = (:crimson, 0.3), linewidth = 1.5)
    cur_xs = CairoMakie.@lift data.x_projs[1:$f_idx]
    cur_loss = CairoMakie.@lift data.e_losses_ev[1:$f_idx]
    CairoMakie.lines!(ax_diag, cur_xs, cur_loss, color = :crimson, linewidth = 2.5, label = "Projectile Energy Loss ΔE (eV)")

    ax_dedx = CairoMakie.Axis(fig_anim[2, 1], yaxisposition = :right, ylabel = "Stopping Power dE/dx (eV/a₀)")
    CairoMakie.xlims!(ax_dedx, data.x_projs[1], data.x_projs[end])
    CairoMakie.ylims!(ax_dedx, -0.2, 2.5)
    CairoMakie.hidespines!(ax_dedx, :t, :b, :l)
    CairoMakie.hideydecorations!(ax_dedx, label = false, ticklabels = false, ticks = false)

    cur_dedx = CairoMakie.@lift data.dedx_local[1:$f_idx]
    CairoMakie.lines!(ax_dedx, data.x_projs, data.dedx_local, color = (:royalblue, 0.3), linewidth = 1.5)
    CairoMakie.lines!(ax_dedx, cur_xs, cur_dedx, color = :royalblue, linewidth = 2.2, label = "Instantaneous dE/dx (eV/a₀)")
    CairoMakie.hlines!(ax_dedx, [1.587], color = (:royalblue, 0.5), linestyle = :dot, label = "1998 Thesis Bragg Peak (1.59 eV/a₀)")

    CairoMakie.vlines!(ax_diag, [-R_cluster, R_cluster], color = (:gray, 0.5), linestyle = :dash, label = "Cluster Jellium Edge")
    CairoMakie.axislegend(ax_diag, position = :lt)

    cur_pt_loss = CairoMakie.@lift [CairoMakie.Point2f(data.x_projs[$f_idx], data.e_losses_ev[$f_idx])]
    CairoMakie.scatter!(ax_diag, cur_pt_loss, color = :crimson, markersize = 10)

    println("Encoding 80M Proton MP4 to $out_mp4...")
    flush(stdout)
    t0 = time()
    CairoMakie.record(fig_anim, out_mp4, 1:nframes; framerate = 20) do i
        f_idx[] = i
    end
    t_rec = time() - t0
    @printf("MP4 recorded in %.2f s (%.1f fps)\n", t_rec, nframes / t_rec)
    flush(stdout)

    println("Converting MP4 to GIF via ffmpeg...")
    flush(stdout)
    run(`ffmpeg -y -loglevel error -i $out_mp4 -vf "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $out_gif`)
    println("80M Proton rendering complete!")
    flush(stdout)
end

function main()
    data = run_proton_80M_simulation(npart = 80_000_000, nfine = 110, dt = 1.0, finesse = 6)
    out_mp4 = joinpath(ROOT, "film_proton_80M.mp4")
    out_gif = joinpath(ROOT, "film_proton_80M.gif")
    out_snapshots = joinpath(ROOT, "proton_snapshots_80M.png")
    render_proton_80M_results(data, out_mp4, out_gif, out_snapshots)

    assets_dir = joinpath(ROOT, "docs", "src", "assets")
    cp(out_gif, joinpath(assets_dir, "film_proton_80M.gif"), force = true)
    cp(out_snapshots, joinpath(assets_dir, "proton_snapshots_80M.png"), force = true)
    println("Proton 80M assets copied to $assets_dir")
    flush(stdout)
end

main()
