#!/usr/bin/env julia
"""
The xenon film, drawn by the GPU instead of by Cairo.

    julia --project=gpu scripts/film_xenon.jl            # simulate, then draw with Cairo
    julia --project=viz scripts/render_xenon_glmakie.jl  # redraw the same run with GLMakie

It reads `xenon_data_cache.jls`, which `scripts/film_xenon.jl` leaves behind, so
the physics is not recomputed: the two renderers draw **the same 176 frames**
and can be compared.

⚠️ **It is not faster, and that is the measurement worth keeping.** Encoding
the same 176 frames:

| | 176 frames | rate |
|---|---:|---:|
| CairoMakie | 30.9 s | 5.7 frames/s |
| GLMakie | 31.3 s | 5.6 frames/s |

The cost is not rasterisation: it is the tessellation of a 45-level `contourf`
over 350×350 — done on the CPU, in Makie, whichever backend draws the result —
plus the video encoder, which is the same `ffmpeg` on both sides. A GPU backend
has nothing to speed up here.

What *did* make the film cheaper was the physics: moving `film_xenon.jl` to the
resident device path took the simulation from 73.8 s to **31.3 s**. The drawing
was never where the time went — and the whole pass differs (82.6 s against 47.0)
only because the Cairo script also draws the six-panel snapshot strip.

⚠️ `scripts/render_proton_glmakie.jl` calls itself "fast GLMakie rendering" on
the same assumption. That claim has **not** been re-measured.

The script stays because it works and because a machine without a usable Cairo
stack can still produce the film with it.

⚠️ **GLMakie needs a display.** On a headless Linux box it wants an EGL-capable
setup or a virtual framebuffer; the Cairo path in `film_xenon.jl` has no such
requirement and stays the fallback.
"""

using Printf
using Serialization
using GLMakie

const ROOT = dirname(@__DIR__)
const GREY = RGBf(0.80, 0.80, 0.80)

"""Bilinear upsampling of one slice — the same 350×350 the Cairo path uses, so
that the comparison is of renderers and not of resolutions."""
function resample_2d(xs, ys, V, n_out_x = 350, n_out_y = 350)
    xs_fine = range(xs[1], xs[end], length = n_out_x)
    ys_fine = range(ys[1], ys[end], length = n_out_y)
    V_fine = Matrix{Float32}(undef, n_out_x, n_out_y)
    nx, ny = length(xs), length(ys)
    for (j, y) in enumerate(ys_fine)
        jy = clamp(searchsortedlast(ys, y), 1, ny - 1)
        uy = clamp(Float32((y - ys[jy]) / (ys[jy+1] - ys[jy])), 0.0f0, 1.0f0)
        for (i, x) in enumerate(xs_fine)
            ix = clamp(searchsortedlast(xs, x), 1, nx - 1)
            tx = clamp(Float32((x - xs[ix]) / (xs[ix+1] - xs[ix])), 0.0f0, 1.0f0)
            V_fine[i, j] = (1 - tx) * (1 - uy) * V[ix, jy] + tx * (1 - uy) * V[ix+1, jy] +
                           (1 - tx) * uy * V[ix, jy+1] + tx * uy * V[ix+1, jy+1]
        end
    end
    xs_fine, ys_fine, V_fine
end

function main()
    cache = joinpath(ROOT, "xenon_data_cache.jls")
    isfile(cache) ||
        error("$cache not found — run `julia --project=gpu scripts/film_xenon.jl` first: " *
              "it computes the run and leaves the cache behind.")
    println("Loading the cached run from $cache...")
    data = deserialize(cache)
    nframes = length(data.frames)
    @printf("%d frames, grid %s\n", nframes, string(size(data.frames[1])))

    t0 = time()
    xs_f, ys_f, _ = resample_2d(data.xs, data.ys, data.frames[1])
    fine = [resample_2d(data.xs, data.ys, fr)[3] for fr in data.frames]
    @printf("resampled to 350x350 in %.1f s\n", time() - t0)

    rho_bulk = 0.00373f0
    levels = range(0.02f0 * rho_bulk, 1.45f0 * rho_bulk, length = 45)

    fig = Figure(size = (900, 850), backgroundcolor = :white)
    idx = Observable(1)

    ax1 = Axis(fig[1, 1], aspect = DataAspect(), backgroundcolor = GREY,
               title = @lift(@sprintf("Na₁₉₆ + Xe²⁵⁺ (500 keV, b = 45 a₀) — t = %.2f fs, x_ion = %+.1f a₀",
                                      data.times_fs[$idx], data.proj_xs[$idx])),
               xlabel = "x (a₀)", ylabel = "y (a₀)")
    contourf!(ax1, xs_f, ys_f, @lift(fine[$idx]);
              levels, colormap = :turbo, extendlow = GREY)
    θ = range(0, 2π, length = 100)
    lines!(ax1, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ),
           color = :white, linewidth = 2, linestyle = :dash)
    scatter!(ax1, @lift(Point2f(data.proj_xs[$idx], data.proj_ys[$idx]));
             color = :yellow, strokecolor = :black, strokewidth = 2, markersize = 14)
    xlims!(ax1, -65, 65); ylims!(ax1, -35, 65)

    ax2 = Axis(fig[2, 1], title = "Dynamic charge transfer: capture and cluster ionization",
               xlabel = "t (fs)", ylabel = "charge (e)")
    lines!(ax2, data.times_fs, data.q_caps, color = (:dodgerblue, 0.4), linewidth = 1.5)
    lines!(ax2, data.times_fs, data.q_nets, color = (:darkorange, 0.4), linewidth = 1.5)
    lines!(ax2, @lift(data.times_fs[1:$idx]), @lift(data.q_caps[1:$idx]);
           color = :dodgerblue, linewidth = 2.5, label = "Q_cap (R = 8 a₀)")
    lines!(ax2, @lift(data.times_fs[1:$idx]), @lift(data.q_nets[1:$idx]);
           color = :darkorange, linewidth = 2.5, label = "Q_cluster")
    scatter!(ax2, @lift([Point2f(data.times_fs[$idx], data.q_caps[$idx])]);
             color = :dodgerblue, markersize = 10)
    scatter!(ax2, @lift([Point2f(data.times_fs[$idx], data.q_nets[$idx])]);
             color = :darkorange, markersize = 10)
    xlims!(ax2, 0, data.times_fs[end]); ylims!(ax2, -0.5, 14.5)
    axislegend(ax2, position = :lt)

    out_mp4 = joinpath(ROOT, "film_xenon.mp4")
    out_gif = joinpath(ROOT, "film_xenon.gif")
    @printf("encoding %d frames to %s...\n", nframes, out_mp4)
    flush(stdout)
    t0 = time()
    record(fig, out_mp4, 1:nframes; framerate = 25) do i
        idx[] = i
    end
    t = time() - t0
    @printf("encoded in %.1f s — %.1f frames/s\n", t, nframes / t)

    # ffmpeg for the gif: Makie writes video, and the palette pass is what keeps
    # a turbo colormap from banding.
    println("converting to $out_gif...")
    run(`ffmpeg -y -loglevel error -i $out_mp4 -vf "fps=15,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" $out_gif`)
    println("done.")
end

main()
