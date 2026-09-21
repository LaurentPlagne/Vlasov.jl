#!/usr/bin/env julia
"""
The xenon film, drawn by the GPU instead of by Cairo.

    julia --project=gpu scripts/xenon.jl            # simulate, then draw
    julia --project=viz scripts/render_xenon_glmakie.jl  # redraw, without recomputing

It reads `xenon_data_cache.jls`, which `scripts/xenon.jl` leaves behind, so
the physics is not recomputed — which is what this script is *for* now: changing
a colour or a limit without paying for the run again.

⚠️ **It is no longer the fast path; `xenon.jl` is.** This script existed
because that one drew with Cairo and `contourf`. It now chooses GLMakie and
`heatmap` wherever a display allows, so the ×6.8 below is had without the
detour — and the two still draw the same 176 frames, which is what made the
comparison possible.

⚠️ **The primitive matters more than the backend.** Swapping CairoMakie for
GLMakie while keeping `contourf` changes nothing — the contour tessellation is
CPU work inside Makie either way. What a GPU backend can accelerate is a
**texture**, and that is what `heatmap` is. Encoding the same 176 frames:

| | CairoMakie | GLMakie |
|---|---:|---:|
| `contourf`, 45 levels, on a 350×350 resample | 30.9 s (5.7 fps) | 31.3 s (5.6 fps) |
| `heatmap`, `interpolate = true`, raw 130×130 | **refused** | **4.6 s (37.9 fps)** |
| `heatmap`, `interpolate = false` | 7.7 s (22.8 fps) | — |

Three things follow. The primitive is worth ×6.8 and the backend on its own is
worth nothing. Cairo **cannot** interpolate here at all — *"Vector{Float32} with
interpolate = true with a non-regular grid is not supported right now"*, the
collocation points not being equally spaced — which is exactly why the Cairo
script resamples to 350×350 first and draws bands. And the GPU sampler doing
that bilinear pass removes the resampling code along with the time it took.

What also made the film cheaper, on the other side of the ledger: moving
`xenon.jl` to the resident device path took its simulation from 73.8 s to
31.3 s.

⚠️ `scripts/render_proton_glmakie.jl` calls itself "fast GLMakie rendering"
while drawing `contourf`. By the table above that name is unearned; it has not
been re-measured.

⚠️ **GLMakie needs a display.** On a headless Linux box it wants an EGL-capable
setup or a virtual framebuffer; the Cairo path in `xenon.jl` has no such
requirement and stays the fallback.
"""

using Printf
using Serialization
using GLMakie

const ROOT = dirname(@__DIR__)
const GREY = RGBf(0.80, 0.80, 0.80)

function main()
    cache = joinpath(ROOT, "xenon_data_cache.jls")
    isfile(cache) ||
        error("$cache not found — run `julia --project=gpu scripts/xenon.jl` first: " *
              "it computes the run and leaves the cache behind.")
    println("Loading the cached run from $cache...")
    data = deserialize(cache)
    nframes = length(data.frames)
    @printf("%d frames, grid %s\n", nframes, string(size(data.frames[1])))

    # ⚠️ **No resampling, and no `contourf`.** The whole point of a GPU backend
    # is the one primitive it can actually accelerate: a `heatmap` is a texture
    # upload, and `interpolate = true` has the sampler do the bilinear
    # smoothing that the 350×350 pre-pass was doing on the CPU. The raw 130×130
    # slice goes straight to the card.
    rho_bulk = 0.00373f0
    c_min, c_max = 0.02f0 * rho_bulk, 1.45f0 * rho_bulk

    fig = Figure(size = (900, 850), backgroundcolor = :white)
    idx = Observable(1)

    ax1 = Axis(fig[1, 1], aspect = DataAspect(), backgroundcolor = GREY,
               title = @lift(@sprintf("Na₁₉₆ + Xe²⁵⁺ (500 keV, b = 45 a₀) — t = %.2f fs, x_ion = %+.1f a₀",
                                      data.times_fs[$idx], data.proj_xs[$idx])),
               xlabel = "x (a₀)", ylabel = "y (a₀)")
    heatmap!(ax1, data.xs, data.ys, @lift(data.frames[$idx]);
             colormap = :turbo, colorrange = (c_min, c_max),
             lowclip = GREY, interpolate = true)
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
