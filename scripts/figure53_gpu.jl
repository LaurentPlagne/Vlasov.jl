#!/usr/bin/env julia
"""
Calcul ultra-rapide du pouvoir d'arrêt sur GPU Metal (Float32) avec mise en valeur du Pic de Bragg :
    julia --project=gpu -t auto scripts/figure53_gpu.jl [--particules=3200000] [--nfine=66]
"""

using Vlasov
using Printf
using Serialization

const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end
# ⚠️ `functional()`, not merely `using`: `using Metal` SUCCEEDS on a machine
# with no Apple GPU — it only logs an error — and the script then took the
# Metal path on a Linux box with an NVIDIA card. Reported from one.
const METAL = try; @eval using Metal; @eval Metal.functional(); catch; false; end
const MAKIE = try; @eval using CairoMakie; CairoMakie.activate!(type = "png"); true; catch; false; end

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV
const X0 = -65.0

function parse_args(argv)
    o = Dict("particules" => "3200000",
             "kev" => "1,4,9,12,14,16,18,20,25,36,50,64",
             "nfine" => "66",
             "sortie" => "figure53_gpu.png")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("unrecognised argument: $a")
        o[m[1]] = m[2]
    end
    o
end

function traverse_gpu(profile, npart, keV; nfine = 66)
    m = nfine ÷ 2 + 1
    n1 = m ÷ 2
    n2 = m - n1
    ninner = 2 * n1
    nouter = 2 * n2 - 2
    rcluster = 78.0
    rbox = 235.0

    energy = keV * KEV
    v = sqrt(2energy / 1836.154)
    dt = min(1.0, 0.5 / v)

    p = SimulationParameters(nfine = nfine, ninner = ninner,
                             nouter = nouter, rcluster = rcluster,
                             rbox = rbox, nions = 1000.0, nelectrons = 1000.0,
                             nparticles = npart, nsteps = 0, dt = dt)
    proj = Projectile(mass = 1836.154, charge = 1.0, energy = energy,
                      impact = 0.0, x0 = X0, dt = dt,
                      softening = GaussianSoftening(1.0))
    sim = Simulation(p, profile; projectile = proj)
    fine = sim.meshes[1]
    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing

    nsteps = ceil(Int, 1.1 * (80 - X0) / (v * dt))

    xs = Float64[proj.position[1]]
    eks = Float64[kinetic_energy(proj)]
    for _ in 1:nsteps
        step!(sim; energy = false, accelerator = acc)
        push!(xs, proj.position[1])
        push!(eks, kinetic_energy(proj))
        proj.position[1] > 80 && break
    end
    (xs, eks, v, dt)
end

function stopping_power(xs, eks; Δx = 4.0)
    nearest(t) = argmin(abs.(xs .- t))
    i, j = nearest(-Δx / 2), nearest(Δx / 2)
    (eks[i] - eks[j]) * HARTREE_TO_EV / (xs[j] - xs[i])
end

function fitted_power(xs, eks; half = 10.0)
    k = findall(x -> -half <= x <= half, xs)
    length(k) < 3 && return NaN
    x = view(xs, k); e = view(eks, k) .* HARTREE_TO_EV
    x̄ = sum(x) / length(x); ē = sum(e) / length(e)
    -sum((x .- x̄) .* (e .- ē)) / sum(abs2, x .- x̄)
end

function published()
    d = Dict{Int,Tuple{Float64,Float64}}()
    for l in eachline(joinpath(ROOT, "ref", "these", "desdx.dat.1000"))
        f = split(l)
        length(f) == 3 && (d[parse(Int, f[1])] = (parse(Float64, f[2]), parse(Float64, f[3])))
    end
    d
end

function render_figure53(results, ref, outfile; nfine = 66, npart = 0)
    fig = CairoMakie.Figure(size = (900, 650), backgroundcolor = :white)
    ax = CairoMakie.Axis(fig[1, 1],
                         title = @sprintf("Pouvoir d'arrêt dE/dx (Na₁₀₀₀ + H⁺, σ = 1 a₀) — GPU Metal, Grille %d, %d particules", nfine, npart),
                         xlabel = "Vitesse du projectile v (u.a.)",
                         ylabel = "dE/dx (eV / a₀)",
                         xgridvisible = true, ygridvisible = true)

    # Reference thesis points from desdx.dat.1000
    all_ref_kevs = sort(collect(keys(ref)))
    v_ref = [sqrt(2 * k * KEV / 1836.154) for k in all_ref_kevs]
    m_ref = [(ref[k][1] + ref[k][2]) / 2 for k in all_ref_kevs]
    err_ref = [abs(ref[k][1] - ref[k][2]) / 2 for k in all_ref_kevs]

    # Fermi velocity reference (v_F = 0.49 a.u. for Na)
    v_F = 0.49
    CairoMakie.vlines!(ax, [v_F]; color = (:gray, 0.4), linestyle = :dot, linewidth = 1.5)
    CairoMakie.text!(ax, v_F - 0.02, 0.55; text = "v_F (Fermi)", rotation = π/2,
                     align = (:left, :bottom), color = :gray30, fontsize = 12)

    # Plot thesis data
    CairoMakie.scatter!(ax, v_ref, m_ref; color = :black, markersize = 12, marker = :circle, label = "Thèse (1998)")
    CairoMakie.errorbars!(ax, v_ref, m_ref, err_ref; color = :black, whiskerwidth = 8)
    CairoMakie.lines!(ax, v_ref, m_ref; color = (:black, 0.35), linestyle = :dash)

    # Simulation results
    vs = [r.v for r in results]
    ds = [r.d for r in results]
    fs = [r.f for r in results]

    CairoMakie.scatterlines!(ax, vs, ds; color = :crimson, markersize = 10, linewidth = 2, label = "GPU (Δx = 4 a₀)")
    CairoMakie.scatterlines!(ax, vs, fs; color = :dodgerblue, markersize = 8, linestyle = :dot, linewidth = 2, label = "GPU (fit ±10 a₀)")

    # Bragg peak identification and highlight
    max_idx = argmax(fs)
    v_bragg = vs[max_idx]
    d_bragg = fs[max_idx]
    keV_bragg = results[max_idx].keV

    # Vertical line and star marker for Bragg peak
    CairoMakie.vlines!(ax, [v_bragg]; color = (:orange, 0.7), linestyle = :dash, linewidth = 1.8)
    CairoMakie.scatter!(ax, [v_bragg], [d_bragg]; color = :gold, strokecolor = :darkorange,
                        strokewidth = 2, markersize = 22, marker = :star5,
                        label = @sprintf("Pic de Bragg (%d keV, v = %.2f)", keV_bragg, v_bragg))
    CairoMakie.text!(ax, v_bragg + 0.03, d_bragg + 0.04;
                     text = @sprintf("Pic de Bragg\n(v ≈ %.2f u.a., %d keV)", v_bragg, keV_bragg),
                     color = :darkorange, font = :bold, fontsize = 13)

    CairoMakie.axislegend(ax, position = :rt)
    CairoMakie.save(outfile, fig)
    @printf("→ Graphique : %s\n", outfile)
end

function main(argv)
    o = parse_args(argv)
    npart = parse(Int, o["particules"])
    energies = parse.(Int, split(o["kev"], ","))
    nfine = parse(Int, o["nfine"])

    grid, ρ = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
    profile = PotentialProfile(grid, ρ)
    ref = published()

    @printf("Na1000, σ_ion = 1, %d pseudo-particles, grid %d (h = %.2f a₀)\n",
            npart, nfine, 2 * 78.0 / nfine)
    @printf("BLAS: %s    forces: %s\n\n",
            ACCELERATE ? "Accelerate" : "OpenBLAS", METAL ? "GPU (Float32)" : "CPU")
    @printf("%-5s %-7s %-6s %-9s %-9s %-18s %-8s %s\n",
            "keV", "v", "dt", "Δx=4", "fit ±10", "thesis", "error", "time")
    results = []
    for keV in energies
        t0 = time()
        xs, eks, v, dt = traverse_gpu(profile, npart, keV; nfine = nfine)
        d = stopping_power(xs, eks)
        f = fitted_power(xs, eks)
        r = get(ref, keV, nothing)
        push!(results, (; keV, v, dt, d, f, xs, eks, r))
        if r === nothing
            @printf("%-5d %-7.3f %-6.2f %-9.3f %-9.3f %-18s %-8s %.0f s\n",
                    keV, v, dt, d, f, "—", "—", time() - t0)
        else
            m = (r[1] + r[2]) / 2
            @printf("%-5d %-7.3f %-6.2f %-9.3f %-9.3f %-18s %+7.1f%% %.0f s\n",
                    keV, v, dt, d, f, @sprintf("%.3f / %.3f", r[1], r[2]),
                    100(d - m) / m, time() - t0)
        end
        flush(stdout)
    end

    outfile = joinpath(ROOT, o["sortie"])
    if MAKIE
        render_figure53(results, ref, outfile; nfine = nfine, npart = npart)
    else
        @printf("(CairoMakie absent, graphique non tracé)\n")
    end
    jls_file = replace(outfile, ".png" => ".jls")
    serialize(jls_file, results)
    @printf("→ Données sérialisées : %s\n", jls_file)
end

main(ARGS)
