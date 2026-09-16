#!/usr/bin/env julia
"""
Renders film.jls (produced by film_images.jl) into film.mp4 and film.gif using CairoMakie.

    julia --project=gpu scripts/make_film.jl [--champ=rho] [--fps=25] [--sortie=film.mp4]
"""

using CairoMakie
using Serialization
using Printf
using Vlasov

const ROOT = dirname(@__DIR__)
const GREY = RGBf(0.78, 0.78, 0.78)

function parse_args(argv)
    o = Dict("champ" => "rho", "fps" => "25", "entree" => "film.jls",
             "sortie" => "film.mp4", "gif" => "film.gif")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("unrecognised argument: $a")
        o[m[1]] = m[2]
    end
    o
end

function main(argv)
    o = parse_args(argv)
    infile = joinpath(ROOT, o["entree"])
    isfile(infile) || error("File not found: $infile. Run scripts/film_images.jl first.")

    println("Loading $infile...")
    d = deserialize(infile)

    fps = parse(Int, o["fps"])
    nframes = length(d.frames)
    champ_type = o["champ"]

    println("Configuring figure for $nframes frames ($champ_type mode)...")
    fig = Figure(size = (960, 800), backgroundcolor = :white)

    # Palette and limits
    c_bulk = sort(filter(>(0), vec(d.rho0)))
    bulk = c_bulk[round(Int, 0.75 * length(c_bulk))]
    hi = Float32(1.45 * bulk)
    void = Float32(0.04 * bulk)

    # Observable frame index
    idx = Observable(1)

    # Upper panel: Density cross section
    ax1 = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = GREY,
               title = @lift(@sprintf("Na₁₀₀₀ + H⁺ (%d keV, v = %.2f a.u.) — x_ion = %.1f a₀",
                                      d.keV, d.v, d.xs[$idx])),
               xlabel = "x (a₀)", ylabel = "y (a₀)")

    # Data slice observable
    field_slice = lift(idx) do i
        raw = d.frames[i]
        if champ_type == "rho"
            [a < void ? NaN32 : a for a in raw]
        else
            diff = raw .- d.rho0
            [abs(a) < 0.005f0 * bulk ? NaN32 : a for a in diff]
        end
    end

    limits = champ_type == "rho" ? (0.0f0, hi) : (-0.2f0 * bulk, 0.2f0 * bulk)
    cmap = champ_type == "rho" ? :jet : :balance

    hm = image!(ax1, extrema(d.gx), extrema(d.gy), field_slice;
                colorrange = limits, colormap = cmap, nan_color = GREY, interpolate = true)
    Colorbar(fig[1, 2], hm, label = champ_type == "rho" ? "Densité ρ (u.a.)" : "Déformation ρ − ρ₀ (u.a.)")

    # Jellium boundary circle: R = r_s * N^(1/3)
    R = Float32(WIGNER_SEITZ_NA * cbrt(1000.0))
    θ = range(0, 2π; length = 200)
    lines!(ax1, R .* cos.(θ), R .* sin.(θ); color = (:black, 0.3), linewidth = 1.5, linestyle = :dash)

    # Projectile position
    proj_pt = lift(idx) do i
        [Point2f(d.xs[i], 0.0)]
    end
    CairoMakie.scatter!(ax1, proj_pt; color = :white, markersize = 12,
                        strokecolor = :black, strokewidth = 1.5)
    xlims!(ax1, -60, 60); ylims!(ax1, -60, 60)

    # Lower panel: Energy loss
    ax2 = Axis(fig[2, 1]; xlabel = "Position du projectile x (a₀)",
               ylabel = "Perte d'énergie (eV)", height = 180)
    loss = (d.e0 .- d.eks) .* d.hartree
    lines!(ax2, d.xs, loss; color = :darkblue, linewidth = 2)
    vlines!(ax2, lift(i -> d.xs[i], idx); color = :red, linewidth = 2)
    vlines!(ax2, [-R, R]; color = (:gray, 0.6), linestyle = :dash)
    xlims!(ax2, first(d.xs), last(d.xs))

    # Output files
    mp4_out = joinpath(ROOT, o["sortie"])
    gif_out = joinpath(ROOT, o["gif"])

    println("Rendering video $mp4_out ($nframes frames at $fps fps)...")
    record(fig, mp4_out, 1:nframes; framerate = fps) do i
        idx[] = i
    end
    @printf("→ %s (%.1f Mo)\n", mp4_out, filesize(mp4_out) / 2^20)

    println("Rendering animated GIF $gif_out...")
    # For GIF, take every 2nd or 3rd frame to keep file size reasonable
    gif_step = max(1, nframes ÷ 120)
    record(fig, gif_out, 1:gif_step:nframes; framerate = fps ÷ gif_step) do i
        idx[] = i
    end
    @printf("→ %s (%.1f Mo)\n", gif_out, filesize(gif_out) / 2^20)
end

main(ARGS)
