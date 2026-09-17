#!/usr/bin/env julia
# Fast GLMakie video rendering from cached Proton 8M simulation data

using Printf
using Serialization
using GLMakie

const ROOT = dirname(@__DIR__)

function render_proton_movie_glmakie()
    cache_file = joinpath(ROOT, "proton_converged_8M_data_cache.jls")
    isfile(cache_file) || error("Cache file $cache_file not found!")

    println("Loading cached 8M Proton simulation data...")
    data = deserialize(cache_file)
    println("Loaded $(length(data.frames)) frames (npart = $(data.npart), mesh = $(data.nfine))")

    out_mp4 = joinpath(ROOT, "film_proton_converged.mp4")
    out_gif = joinpath(ROOT, "film_proton_converged.gif")
    out_snapshots = joinpath(ROOT, "proton_snapshots_converged.png")

    nframes = length(data.frames)
    xs = data.outx
    ys = data.outy

    n0 = 0.00373f0
    c_min = 0.02f0 * n0
    c_max = 1.45f0 * n0
    levels = range(c_min, c_max, length = 50)
    GREY = RGBf(0.82, 0.82, 0.82)

    R_cluster = 40.0
    θ = range(0, 2π, length = 150)

    # GLMakie Offscreen Figure
    GLMakie.activate!()
    fig_anim = GLMakie.Figure(size = (800, 950), backgroundcolor = :white)
    f_idx = GLMakie.Observable(1)

    ax_cut = GLMakie.Axis(fig_anim[1, 1],
        title = GLMakie.@lift(@sprintf("Na₁₀₀₀ + H⁺ (16 keV, b = 0) — x_p = %+5.1f a₀, ΔE = %5.1f eV",
                                       data.x_projs[$f_idx], data.e_losses_ev[$f_idx])),
        xlabel = "x (a₀)", ylabel = "y (a₀)",
        aspect = GLMakie.DataAspect(),
        backgroundcolor = GREY)
    GLMakie.xlims!(ax_cut, -65, 65)
    GLMakie.ylims!(ax_cut, -50, 50)

    cur_frame = GLMakie.@lift data.frames[$f_idx]
    # GLMakie contourf is hardware GPU accelerated!
    GLMakie.contourf!(ax_cut, xs, ys, cur_frame,
                      levels = levels, colormap = :turbo, extendlow = GREY, extendhigh = :firebrick)

    GLMakie.lines!(ax_cut, R_cluster .* cos.(θ), R_cluster .* sin.(θ),
                   color = :white, linestyle = :dash, linewidth = 2.0)

    p_pt = GLMakie.@lift GLMakie.Point2f(data.x_projs[$f_idx], 0.0)
    GLMakie.scatter!(ax_cut, p_pt, color = :white, strokecolor = :black,
                     strokewidth = 2.5, markersize = 14)

    # Diagnostic plot: Projectile Stopping Loss ΔE and Stopping Power dE/dx
    ax_diag = GLMakie.Axis(fig_anim[2, 1],
        title = @sprintf("Projectile Stopping: Total Loss ΔE(x) & Central Plateau dE/dx = %.2f eV/a₀",
                         data.dE_dx_center),
        xlabel = "Projectile Position x (a₀)", ylabel = "Energy Loss ΔE (eV)",
        xgridvisible = true, ygridvisible = true)
    GLMakie.xlims!(ax_diag, data.x_projs[1], data.x_projs[end])
    GLMakie.ylims!(ax_diag, -5, maximum(data.e_losses_ev) * 1.15)

    # Static guide lines
    GLMakie.lines!(ax_diag, data.x_projs, data.e_losses_ev, color = (:crimson, 0.3), linewidth = 1.5)
    cur_xs = GLMakie.@lift data.x_projs[1:$f_idx]
    cur_loss = GLMakie.@lift data.e_losses_ev[1:$f_idx]
    GLMakie.lines!(ax_diag, cur_xs, cur_loss, color = :crimson, linewidth = 2.5, label = "Projectile Energy Loss ΔE (eV)")

    # Right axis for stopping power dE/dx
    ax_dedx = GLMakie.Axis(fig_anim[2, 1], yaxisposition = :right, ylabel = "Stopping Power dE/dx (eV/a₀)")
    GLMakie.xlims!(ax_dedx, data.x_projs[1], data.x_projs[end])
    GLMakie.ylims!(ax_dedx, -0.2, 2.5)
    GLMakie.hidespines!(ax_dedx, :t, :b, :l)
    GLMakie.hideydecorations!(ax_dedx, label = false, ticklabels = false, ticks = false)

    cur_dedx = GLMakie.@lift data.dedx_local[1:$f_idx]
    GLMakie.lines!(ax_dedx, data.x_projs, data.dedx_local, color = (:royalblue, 0.3), linewidth = 1.5)
    GLMakie.lines!(ax_dedx, cur_xs, cur_dedx, color = :royalblue, linewidth = 2.2, label = "Instantaneous dE/dx (eV/a₀)")
    GLMakie.hlines!(ax_dedx, [1.587], color = (:royalblue, 0.5), linestyle = :dot, label = "1998 Thesis Bragg Peak (1.59 eV/a₀)")

    GLMakie.vlines!(ax_diag, [-R_cluster, R_cluster], color = (:gray, 0.5), linestyle = :dash, label = "Cluster Jellium Edge")
    GLMakie.axislegend(ax_diag, position = :lt)

    cur_pt_loss = GLMakie.@lift [GLMakie.Point2f(data.x_projs[$f_idx], data.e_losses_ev[$f_idx])]
    GLMakie.scatter!(ax_diag, cur_pt_loss, color = :crimson, markersize = 10)

    println("Hardware-accelerated GLMakie encoding to $out_mp4...")
    t0 = time()
    GLMakie.record(fig_anim, out_mp4, 1:nframes; framerate = 20) do i
        f_idx[] = i
    end
    t_enc = time() - t0
    @printf("Encoded %d frames in %.2f s (%.1f fps)!\n", nframes, t_enc, nframes / t_enc)

    println("Converting MP4 to GIF via ffmpeg...")
    run(`ffmpeg -y -loglevel error -i $out_mp4 -vf "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $out_gif`)
    println("GLMakie rendering complete!")

    # Copy assets to docs/src/assets
    assets_dir = joinpath(ROOT, "docs", "src", "assets")
    cp(out_gif, joinpath(assets_dir, "film_proton_converged.gif"), force = true)
    cp(out_snapshots, joinpath(assets_dir, "proton_snapshots_converged.png"), force = true)
    println("Copied converged 8M proton assets to $assets_dir")
end

render_proton_movie_glmakie()
