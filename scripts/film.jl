#!/usr/bin/env julia
"""
Movie of the crossing: a slice of the electron density, interactive and on video.

    julia --project=viz -t auto scripts/film.jl            # interactive window
    julia --project=viz -t auto scripts/film.jl --video    # writes film.mp4
    julia --project=viz -t auto scripts/film.jl --video --champ=rho

Reads `film.jls`, produced by [`film_images.jl`](film_images.jl).

(The command-line flags keep their French names — `champ`, `sortie` — since
they are the script's interface, not prose.)

The window carries a slider, a play button, and the choice of field:

  * **`ρ`** (default) — the density itself, as the thesis's `snappro` figures
    plot it. On an absolute scale the sampling grain is barely visible and the
    clump the ion drags along stands out on its own.
  * **`δρ`** — the departure from the initial state. In principle it isolates
    the deformation; in practice, at 800 000 pseudo-particles, it is
    **noise-limited**. A fine-grid cell holds about 134 of them, so the shot
    noise on `δρ` is 9.6 % of `ρ₀` — the same order as the deformation itself,
    giving a signal-to-noise of 3 per cell. The picture then reads as salt and
    pepper. Useful for reading off the sign of a local feature, not for a film.

!!! warning "Do not smooth `δρ` to rescue it"
    A box blur over one cell divides the noise by only 1.4, and turns the grain
    into large coherent blobs that **look like** structure. It makes the picture
    prettier and the reading wrong. Averaging over `z` fails for the same reason
    in reverse: the wake fits inside a single cell in `z`, so a thicker slab
    dilutes the signal faster than it kills the noise — measured, the optimum is
    `|z| ≤ 2 a₀`, worth 23 %, and a full-depth projection is **worse** than a
    single plane.

The lower panel tracks the projectile's energy: its slope is the stopping
power, and the vertical line marks the instant on display.
"""

using Vlasov
using GLMakie
using Printf
using Serialization

const ROOT = dirname(@__DIR__)

function parse_args(argv)
    o = Dict("champ" => "rho", "video" => "", "fps" => "24", "sortie" => "film.mp4")
    for a in argv
        if a == "--video"
            o["video"] = "oui"
        else
            m = match(r"^--([a-z]+)=(.+)$", a)
            (m === nothing || !haskey(o, m[1])) && error("unrecognised argument: $a")
            o[m[1]] = m[2]
        end
    end
    o
end

"""Robust colour bound: a high quantile rather than the maximum.

The maximum of a density field is reached on a handful of points and would
flatten everything else; the 0.995 quantile gives a scale on which the
deformation is visible."""
function color_limit(frames, ρ0, field)
    vals = Float32[]
    for i in round.(Int, range(1, length(frames); length = min(24, length(frames))))
        f = field == "rho" ? frames[i] : frames[i] .- ρ0
        append!(vals, abs.(vec(f)))
    end
    sort!(vals)
    vals[max(1, round(Int, 0.995 * length(vals)))]
end

function main(argv)
    o = parse_args(argv)
    path = joinpath(ROOT, "film.jls")
    isfile(path) || error("$path missing — run scripts/film_images.jl first")
    d = deserialize(path)
    n = length(d.frames)

    @printf("%d frames, Na1000 at %d keV (v = %.2f), %d particles\n",
            n, d.keV, d.v, d.npart)

    # ⚠️ Two independent bounds, one per field. Deriving the diverging bound
    # from whichever field happens to be the default washed out `δρ` as soon as
    # the menu switched to it.
    lim = color_limit(d.frames, d.rho0, "delta")

    # `ρ` is displayed the way figure 5.2 of the thesis does it: rainbow palette,
    # top of the scale at 1.45·ρ_bulk so the cluster body lands in the orange and
    # the wake's oscillation spans green→red, and **vacuum masked** rather than
    # coloured. From 0 → max the bulk sits at 89 % of full scale, the whole
    # cluster is one flat colour, and the range is spent on empty space.
    bulk = let c = sort(filter(>(0), vec(d.rho0)))
        c[round(Int, 0.75 * length(c))]
    end
    rhi = Float32(1.45bulk)
    void = Float32(0.04bulk)
    GREY = RGBf(0.78, 0.78, 0.78)

    fig = Figure(size = (1000, 780))
    idx = Observable(1)
    champ = Observable(o["champ"])

    # --- density map --------------------------------------------------------
    ax = Axis(fig[1, 1]; xlabel = "x (a₀)", ylabel = "y (a₀)", aspect = DataAspect(),
              title = @lift(@sprintf("Na₁₀₀₀ + H⁺ %d keV — x = %.1f a₀, loss = %.2f eV",
                                     d.keV, d.xs[$idx],
                                     (d.e0 - d.eks[$idx]) * d.hartree)))

    field = @lift($champ == "rho" ?
                  [a < void ? NaN32 : a for a in d.frames[$idx]] :
                  d.frames[$idx] .- d.rho0)
    limits = @lift($champ == "rho" ? (0.0f0, rhi) : (-lim, lim))
    cmap = @lift($champ == "rho" ? :jet : :balance)
    # `image!` and not `heatmap!`, for `interpolate`: the density is a C¹ spline
    # and the collocation values are point samples of it, so bilinear display is
    # closer to the field than hard cells. ⚠️ It does not reduce the noise — it
    # only stops the grid from showing through.
    hm = image!(ax, extrema(d.gx), extrema(d.gy), field;
                colorrange = limits, colormap = cmap,
                nan_color = GREY, interpolate = true)
    Colorbar(fig[1, 2], hm,
             label = @lift($champ == "rho" ? "ρ (a.u.)" : "ρ − ρ₀ (a.u.)"))

    # The jellium circle: R = r_s·N^⅓, to locate the cluster.
    R = WIGNER_SEITZ_NA * cbrt(1000.0)
    θ = range(0, 2π; length = 200)
    lines!(ax, R .* cos.(θ), R .* sin.(θ); color = (:black, 0.18), linewidth = 1)

    # The projectile. ⚠️ Qualified: `scatter!` is exported by BOTH Vlasov (the
    # deposit engine) and Makie, so the bare name resolves to neither.
    GLMakie.scatter!(ax, @lift([Point2f(d.xs[$idx], 0)]); color = :white,
                     markersize = 7, strokecolor = :black, strokewidth = 1)

    # --- projectile energy ---------------------------------------------------
    ax2 = Axis(fig[2, 1]; xlabel = "projectile x (a₀)", ylabel = "loss (eV)",
               height = 150)
    loss = (d.e0 .- d.eks) .* d.hartree
    lines!(ax2, d.xs, loss; color = :black)
    vlines!(ax2, @lift(d.xs[$idx]); color = :red)
    vlines!(ax2, [-R, R]; color = (:gray, 0.6), linestyle = :dash)
    xlims!(ax2, first(d.xs), last(d.xs))

    # --- controls -----------------------------------------------------------
    if isempty(o["video"])
        grid = fig[3, 1] = GridLayout()
        sl = Slider(grid[1, 2], range = 1:n, startvalue = 1)
        connect!(idx, sl.value)
        play = Button(grid[1, 1], label = "▶", width = 45)
        menu = Menu(grid[1, 3], options = ["δρ" => "delta", "ρ" => "rho"],
                    default = o["champ"] == "rho" ? "ρ" : "δρ", width = 90)
        on(menu.selection) do s
            s === nothing || (champ[] = s)
        end

        running = Ref(false)
        on(play.clicks) do _
            running[] = !running[]
            play.label = running[] ? "⏸" : "▶"
            running[] || return
            @async while running[]
                set_close_to!(sl, mod1(sl.value[] + 1, n))
                sleep(1 / parse(Int, o["fps"]))
            end
        end

        display(fig)
        @info "Window open — slider, ▶ to play, menu to switch field."
        wait(GLMakie.Screen(fig.scene))
    else
        out = joinpath(ROOT, o["sortie"])
        fps = parse(Int, o["fps"])
        @printf("recording %d frames at %d fps → %.1f s\n", n, fps, n / fps)
        record(fig, out, 1:n; framerate = fps) do i
            idx[] = i
        end
        @printf("→ %s (%.1f MB)\n", out, filesize(out) / 2^20)
    end
end

main(ARGS)
