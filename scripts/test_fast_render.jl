#!/usr/bin/env julia
using Printf
using Serialization
using CairoMakie

const ROOT = dirname(@__DIR__)
cache_file = joinpath(ROOT, "proton_converged_8M_data_cache.jls")
data = deserialize(cache_file)
println("Loaded $(length(data.frames)) frames.")

out_mp4 = joinpath(ROOT, "film_proton_converged.mp4")
out_gif = joinpath(ROOT, "film_proton_converged.gif")

nframes = length(data.frames)
xs = data.outx
ys = data.outy

n0 = 0.00373f0
c_min = 0.02f0 * n0
c_max = 1.45f0 * n0
GREY = RGBf(0.82, 0.82, 0.82)
R_cluster = 40.0
θ = range(0, 2π, length = 150)

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
# Fast interpolated heatmap instead of expensive vector contourf
CairoMakie.heatmap!(ax_cut, xs, ys, cur_frame,
                    colormap = :turbo, colorrange = (c_min, c_max), interpolate = true)

CairoMakie.lines!(ax_cut, R_cluster .* cos.(θ), R_cluster .* sin.(θ),
                  color = :white, linestyle = :dash, linewidth = 2.0)

p_pt = CairoMakie.@lift CairoMakie.Point2f(data.x_projs[$f_idx], 0.0)
CairoMakie.scatter!(ax_cut, p_pt, color = :white, strokecolor = :black,
                    strokewidth = 2.5, markersize = 14)

# Diagnostic plot
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

println("Encoding MP4 to $out_mp4...")
t0 = time()
CairoMakie.record(fig_anim, out_mp4, 1:nframes; framerate = 20) do i
    f_idx[] = i
end
t_rec = time() - t0
@printf("MP4 recorded in %.2f s (%.1f fps)\n", t_rec, nframes / t_rec)

println("Converting to GIF via ffmpeg...")
run(`ffmpeg -y -loglevel error -i $out_mp4 -vf "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $out_gif`)
println("Done!")

# Copy to assets
assets_dir = joinpath(ROOT, "docs", "src", "assets")
cp(out_gif, joinpath(assets_dir, "film_proton_converged.gif"), force = true)
println("Copied to $assets_dir/film_proton_converged.gif")
