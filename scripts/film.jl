#!/usr/bin/env julia
"""
Film de la traversée : coupe de la densité électronique, interactive et en vidéo.

    julia --project=viz -t auto scripts/film.jl            # fenêtre interactive
    julia --project=viz -t auto scripts/film.jl --video    # écrit film.mp4
    julia --project=viz -t auto scripts/film.jl --video --champ=rho

Lit `film.jls`, produit par [`film_images.jl`](film_images.jl).

La fenêtre porte un curseur, un bouton de lecture, et le choix du champ :

  * **`δρ`** (défaut) — l'écart à l'état initial. C'est lui qu'il faut regarder :
    la déformation vaut quelques pour cent d'un fond mille fois plus grand, et
    une échelle absolue la noie. L'échelle est **divergente et centrée sur
    zéro**, donc le sillage (déficit) et l'accumulation se distinguent.
  * **`ρ`** — la densité elle-même, comme la tracent les figures `snappro` de
    la thèse.

Le panneau du bas suit l'énergie du projectile : la pente y est le pouvoir
d'arrêt, et le trait vertical marque l'instant affiché.
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
            (m === nothing || !haskey(o, m[1])) && error("argument non reconnu : $a")
            o[m[1]] = m[2]
        end
    end
    o
end

"""Borne de couleur robuste : un quantile élevé plutôt que le maximum.

Le maximum d'un champ de densité est atteint sur une poignée de points et
écraserait tout le reste ; le quantile 0,995 donne une échelle où la
déformation se voit."""
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
    isfile(path) || error("$path absent — lancer d'abord scripts/film_images.jl")
    d = deserialize(path)
    n = length(d.frames)

    @printf("%d images, Na1000 à %d keV (v = %.2f), %d particules\n",
            n, d.keV, d.v, d.npart)

    lim = color_limit(d.frames, d.rho0, o["champ"])
    rlim = color_limit(d.frames, d.rho0, "rho")

    fig = Figure(size = (1000, 780))
    idx = Observable(1)
    champ = Observable(o["champ"])

    # --- carte de densité ---------------------------------------------------
    ax = Axis(fig[1, 1]; xlabel = "x (a₀)", ylabel = "y (a₀)", aspect = DataAspect(),
              title = @lift(@sprintf("Na₁₀₀₀ + H⁺ %d keV — x = %.1f a₀, perte = %.2f eV",
                                     d.keV, d.xs[$idx],
                                     (d.e0 - d.eks[$idx]) * d.hartree)))

    image = @lift($champ == "rho" ? d.frames[$idx] : d.frames[$idx] .- d.rho0)
    limits = @lift($champ == "rho" ? (0.0f0, rlim) : (-lim, lim))
    cmap = @lift($champ == "rho" ? :viridis : :balance)
    hm = heatmap!(ax, d.gx, d.gy, image; colorrange = limits, colormap = cmap)
    Colorbar(fig[1, 2], hm,
             label = @lift($champ == "rho" ? "ρ (u.a.)" : "ρ − ρ₀ (u.a.)"))

    # Le cercle du jellium : R = r_s·N^⅓, pour situer l'agrégat.
    R = WIGNER_SEITZ_NA * cbrt(1000.0)
    θ = range(0, 2π; length = 200)
    lines!(ax, R .* cos.(θ), R .* sin.(θ); color = (:white, 0.35), linewidth = 1)

    # The projectile. ⚠️ Qualified: `scatter!` is exported by BOTH Vlasov (the
    # deposit engine) and Makie, so the bare name resolves to neither.
    GLMakie.scatter!(ax, @lift([Point2f(d.xs[$idx], 0)]); color = :red,
                     markersize = 13, strokecolor = :white, strokewidth = 1.5)

    # --- énergie du projectile ----------------------------------------------
    ax2 = Axis(fig[2, 1]; xlabel = "x du projectile (a₀)", ylabel = "perte (eV)",
               height = 150)
    perte = (d.e0 .- d.eks) .* d.hartree
    lines!(ax2, d.xs, perte; color = :black)
    vlines!(ax2, @lift(d.xs[$idx]); color = :red)
    vlines!(ax2, [-R, R]; color = (:gray, 0.6), linestyle = :dash)
    xlims!(ax2, first(d.xs), last(d.xs))

    # --- commandes ----------------------------------------------------------
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
        @info "Fenêtre ouverte — curseur, ▶ pour lire, menu pour changer de champ."
        wait(GLMakie.Screen(fig.scene))
    else
        out = joinpath(ROOT, o["sortie"])
        fps = parse(Int, o["fps"])
        @printf("enregistrement de %d images à %d im/s → %.1f s\n", n, fps, n / fps)
        record(fig, out, 1:n; framerate = fps) do i
            idx[] = i
        end
        @printf("→ %s (%.1f Mo)\n", out, filesize(out) / 2^20)
    end
end

main(ARGS)
