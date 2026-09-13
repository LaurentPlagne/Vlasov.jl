#!/usr/bin/env julia
"""
Reproduit la courbe de freinage de la thèse : Na₁₀₀₀, σ_ion = 1 u.a.

    julia --project=. -t auto scripts/figure53.jl [--particules=200000] [--kev=1,4,9,16,25]

La grandeur tracée est celle que définit la thèse — une **pente locale au
centre**, pas la perte totale :

    dE/dx ≃ [E_k(+Δx/2) − E_k(−Δx/2)] / Δx,   Δx = 4 u.a.

Les valeurs publiées sont dans [`ref/these/`](../ref/these/) : `desdx.dat.1000`
donne le résultat, `Ekproj.dat.N` les trajectoires dont il est tiré.

L'état initial vient de `rhorad.Na1000.dat`, la densité radiale d'équilibre
archivée (octobre 1998, 998,7 électrons intégrés) : le `pot.dat` qu'attendait
`initialise4` n'a pas survécu, mais la densité suffit — voir
[`PotentialProfile`](@ref).

⚠️ La force est celle de la **thèse** (gaussienne), pas celle du Fortran
(boule). C'est tout l'objet de l'anomalie 10.
"""

using Vlasov
using Printf

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV        # 1 keV en hartree

# Grille : h = 2·rcluster/nfine ≈ 3,55, la résolution validée sur Na₁₉₆, mais
# étendue pour contenir Na₁₀₀₀ (R = 40 a₀) et le projectile dès son entrée.
const GRID = (nfine = 44, ncoarse = 22, rcluster = 78.0, rbox = 235.0)
const X0 = -65.0                        # départ, comme les trajectoires archivées

function parse_args(argv)
    o = Dict("particules" => "200000", "kev" => "1,4,9,16,25")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("argument non reconnu : $a")
        o[m[1]] = m[2]
    end
    o
end

"""Parcourt une traversée et rend `(xs, eks)` — position et énergie cinétique."""
function traverse(profile, npart, keV)
    p = SimulationParameters(nfine = GRID.nfine, ninner = GRID.ncoarse,
                             nouter = GRID.ncoarse, rcluster = GRID.rcluster,
                             rbox = GRID.rbox, nions = 1000.0, nelectrons = 1000.0,
                             nparticles = npart, nsteps = 0, dt = 1.0)
    energy = keV * KEV
    proj = Projectile(mass = 1836.154, charge = 1.0, energy = energy,
                      impact = 0.0, x0 = X0, dt = 1.0,
                      softening = GaussianSoftening(1.0))
    sim = Simulation(p, profile; projectile = proj)

    # Assez de pas pour ressortir : la distance à parcourir divisée par la
    # vitesse, plus une marge — le projectile ralentit.
    v = sqrt(2energy / 1836.154)
    nsteps = ceil(Int, 1.1 * (80 - X0) / v)

    xs = Float64[proj.position[1]]
    eks = Float64[kinetic_energy(proj)]
    for _ in 1:nsteps
        step!(sim)
        push!(xs, proj.position[1])
        push!(eks, kinetic_energy(proj))
        proj.position[1] > 80 && break
    end
    (xs, eks)
end

"""Pente locale au centre sur `Δx`, en eV/a₀ — la définition de la thèse.

⚠️ Sur quatre bohrs seulement, cet estimateur est **très sensible au bruit de
tirage** : à 20 000 pseudo-particules il rend des valeurs négatives alors que
la trajectoire complète est correcte à 3 %. C'est pourquoi la production en
employait 800 000. [`fitted_power`](@ref) sert de garde-fou.
"""
function stopping_power(xs, eks; Δx = 4.0)
    nearest(t) = argmin(abs.(xs .- t))
    i, j = nearest(-Δx / 2), nearest(Δx / 2)
    (eks[i] - eks[j]) * HARTREE_TO_EV / (xs[j] - xs[i])
end

"""Même pente, par moindres carrés sur une fenêtre plus large — moins fidèle à
la recette de la thèse, mais moins bruitée. Si les deux divergent, c'est que la
statistique ne suffit pas."""
function fitted_power(xs, eks; half = 10.0)
    k = findall(x -> -half <= x <= half, xs)
    length(k) < 3 && return NaN
    x = view(xs, k); e = view(eks, k) .* HARTREE_TO_EV
    x̄ = sum(x) / length(x); ē = sum(e) / length(e)
    -sum((x .- x̄) .* (e .- ē)) / sum(abs2, x .- x̄)
end

"""Valeurs publiées : `desdx.dat.1000`, colonnes keV puis deux mesures."""
function published()
    d = Dict{Int,Tuple{Float64,Float64}}()
    for l in eachline(joinpath(ROOT, "ref", "these", "desdx.dat.1000"))
        f = split(l)
        length(f) == 3 && (d[parse(Int, f[1])] = (parse(Float64, f[2]), parse(Float64, f[3])))
    end
    d
end

function main(argv)
    o = parse_args(argv)
    npart = parse(Int, o["particules"])
    energies = parse.(Int, split(o["kev"], ","))

    grid, ρ = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
    profile = PotentialProfile(grid, ρ)
    ref = published()

    @printf("Na1000, σ_ion = 1, %d pseudo-particules, grille %d (h = %.2f a₀)\n\n",
            npart, GRID.nfine, 2GRID.rcluster / GRID.nfine)
    @printf("%-5s %-7s %-9s %-9s %-18s %-8s %s\n",
            "keV", "v", "Δx=4", "fit ±10", "thèse", "écart", "temps")
    for keV in energies
        t0 = time()
        xs, eks = traverse(profile, npart, keV)
        d = stopping_power(xs, eks)
        f = fitted_power(xs, eks)
        v = sqrt(2 * keV * KEV / 1836.154)
        r = get(ref, keV, nothing)
        if r === nothing
            @printf("%-5d %-7.3f %-9.3f %-9.3f %-18s %-8s %.0f s\n",
                    keV, v, d, f, "—", "—", time() - t0)
        else
            m = (r[1] + r[2]) / 2
            @printf("%-5d %-7.3f %-9.3f %-9.3f %-18s %+7.1f%% %.0f s\n",
                    keV, v, d, f, @sprintf("%.3f / %.3f", r[1], r[2]),
                    100(d - m) / m, time() - t0)
        end
        flush(stdout)
    end
end

main(ARGS)
