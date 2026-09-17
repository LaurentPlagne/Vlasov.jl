#!/usr/bin/env julia
using Printf
using Serialization
using CairoMakie

const ROOT = dirname(@__DIR__)
cache_file = joinpath(ROOT, "proton_converged_8M_data_cache.jls")
data = deserialize(cache_file)

xs = data.outx
ys = data.outy
n0 = 0.00373f0
c_min = 0.05f0 * n0
c_max = 1.45f0 * n0
GREY = RGBf(0.82, 0.82, 0.82)
R_cluster = 40.0
θ = range(0, 2π, length = 150)

# Test on Frame 1 (t=0, the one Laurent showed) and Frame 45 (wake at core)
f1 = copy(data.frames[1])
f45 = copy(data.frames[45])

fig = Figure(size = (1100, 950), backgroundcolor = :white)

# 1. Option A: heatmap with nan_color = GREY (values < c_min masked to NaN)
ax1 = Axis(fig[1, 1], title = "Option A: Heatmap masked (nan_color = GREY) — Frame 1 (t=0)",
           aspect = DataAspect(), backgroundcolor = GREY)
f1_masked = copy(f1)
f1_masked[f1_masked .< c_min] .= NaN32
heatmap!(ax1, xs, ys, f1_masked, colormap = :turbo, colorrange = (c_min, c_max),
         nan_color = GREY, interpolate = true)
lines!(ax1, R_cluster .* cos.(θ), R_cluster .* sin.(θ), color = :white, linestyle = :dash, linewidth = 2.0)
scatter!(ax1, [data.x_projs[1]], [0.0], color = :white, strokecolor = :black, strokewidth = 2.0, markersize = 12)

# 2. Option B: contourf with 20 clean levels and extendlow = GREY
ax2 = Axis(fig[1, 2], title = "Option B: Contourf 20 levels (extendlow = GREY) — Frame 1 (t=0)",
           aspect = DataAspect(), backgroundcolor = GREY)
contourf!(ax2, xs, ys, f1, levels = range(c_min, c_max, length = 20),
          colormap = :turbo, extendlow = GREY, extendhigh = :firebrick)
lines!(ax2, R_cluster .* cos.(θ), R_cluster .* sin.(θ), color = :white, linestyle = :dash, linewidth = 2.0)
scatter!(ax2, [data.x_projs[1]], [0.0], color = :white, strokecolor = :black, strokewidth = 2.0, markersize = 12)

# 3. Option A on Wake Frame 45
ax3 = Axis(fig[2, 1], title = "Option A: Heatmap masked — Frame 45 (Wake)",
           aspect = DataAspect(), backgroundcolor = GREY)
f45_masked = copy(f45)
f45_masked[f45_masked .< c_min] .= NaN32
heatmap!(ax3, xs, ys, f45_masked, colormap = :turbo, colorrange = (c_min, c_max),
         nan_color = GREY, interpolate = true)
lines!(ax3, R_cluster .* cos.(θ), R_cluster .* sin.(θ), color = :white, linestyle = :dash, linewidth = 2.0)
scatter!(ax3, [data.x_projs[45]], [0.0], color = :white, strokecolor = :black, strokewidth = 2.0, markersize = 12)

# 4. Option B on Wake Frame 45
ax4 = Axis(fig[2, 2], title = "Option B: Contourf 20 levels — Frame 45 (Wake)",
           aspect = DataAspect(), backgroundcolor = GREY)
contourf!(ax4, xs, ys, f45, levels = range(c_min, c_max, length = 20),
          colormap = :turbo, extendlow = GREY, extendhigh = :firebrick)
lines!(ax4, R_cluster .* cos.(θ), R_cluster .* sin.(θ), color = :white, linestyle = :dash, linewidth = 2.0)
scatter!(ax4, [data.x_projs[45]], [0.0], color = :white, strokecolor = :black, strokewidth = 2.0, markersize = 12)

for ax in [ax1, ax2, ax3, ax4]
    xlims!(ax, -65, 65)
    ylims!(ax, -50, 50)
end

save(joinpath(ROOT, "test_visu_comparison.png"), fig, px_per_unit = 2)
println("Saved test_visu_comparison.png")
