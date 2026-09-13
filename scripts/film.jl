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

  * **`δρ`** (default) — the departure from the initial state. This is the one
    to watch: the deformation amounts to a few per cent of a background a
    thousand times larger, and an absolute scale drowns it. The colour scale is
    **diverging and centred on zero**, so the wake (a deficit) and the pile-up
    are told apart.
  * **`ρ`** — the density itself, as the thesis's `snappro` figures plot it.

The lower panel tracks the projectile's energy: its slope is the stopping
power, and the vertical line marks the instant on display.
"""

using Vlasov
using GLMakie
using Printf
using Serialization

const ROOT = dirname(@__DIR__)

function parse_args(argv)
    o = Dict("champ" => "delta", "video" => "", "fps" => "24", "sortie" => "film.mp4")
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

    lim = color_limit(d.frames, d.rho0, o["champ"])
    rlim = color_limit(d.frames, d.rho0, "rho")

    fig = Figure(size = (1000, 780))
    idx = Observable(1)
    champ = Observable(o["champ"])

    # --- density map --------------------------------------------------------
    ax = Axis(fig[1, 1]; xlabel = "x (a₀)", ylabel = "y (a₀)", aspect = DataAspect(),
              title = @lift(@sprintf("Na₁₀₀₀ + H⁺ %d keV — x = %.1f a₀, loss = %.2f eV",
                                     d.keV, d.xs[$idx],
                                     (d.e0 - d.eks[$idx]) * d.hartree)))

    image = @lift($champ == "rho" ? d.frames[$idx] : d.frames[$idx] .- d.rho0)
    limits = @lift($champ == "rho" ? (0.0f0, rlim) : (-lim, lim))
    cmap = @lift($champ == "rho" ? :viridis : :balance)
    hm = heatmap!(ax, d.gx, d.gy, image; colorrange = limits, colormap = cmap)
    Colorbar(fig[1, 2], hm,
             label = @lift($champ == "rho" ? "ρ (a.u.)" : "ρ − ρ₀ (a.u.)"))

    # The jellium circle: R = r_s·N^⅓, to locate the cluster.
    R = WIGNER_SEITZ_NA * cbrt(1000.0)
    θ = range(0, 2π; length = 200)
    lines!(ax, R .* cos.(θ), R .* sin.(θ); color = (:white, 0.35), linewidth = 1)

    # The projectile. ⚠️ Qualified: `scatter!` is exported by BOTH Vlasov (the
    # deposit engine) and Makie, so the bare name resolves to neither.
    GLMakie.scatter!(ax, @lift([Point2f(d.xs[$idx], 0)]); color = :red,
                     markersize = 13, strokecolor = :white, strokewidth = 1.5)

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
