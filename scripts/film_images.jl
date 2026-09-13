#!/usr/bin/env julia
"""
Produit les images d'un film de traversée : coupes de la densité électronique.

    julia --project=. -t auto scripts/film_images.jl [--kev=4] [--images=400] [--particules=800000]

C'est la vue des figures `snappro` de la thèse — une **coupe** de la densité
dans le plan `z = 0`, celui que parcourt le projectile. Ce qu'on y voit est la
déformation du nuage à son passage, et son **asymétrie** : le sillage traîne
derrière l'ion, et d'autant plus que sa vitesse est grande.

Les images ne sont pas prises à chaque pas — un pas dure 0,4 u.a. de temps et il
en faut quelques centaines pour la traversée, alors qu'un film de vingt secondes
en demande cinq cents. Le pas d'échantillonnage est calculé pour tomber juste.

Deux champs sont conservés par image :

  * `ρ` — la densité elle-même, comme la thèse la trace ;
  * `δρ = ρ − ρ₀` — l'écart à l'état initial, qui **montre bien mieux** la
    déformation : celle-ci vaut quelques pour cent d'un fond mille fois plus
    grand, et se noie dans une échelle absolue.

Sortie : `film.jls` (Serialization, stdlib — pas de dépendance ajoutée), lu par
[`film.jl`](film.jl).
"""

using Vlasov
using Printf
using Serialization

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV

function parse_args(argv)
    o = Dict("kev" => "4", "images" => "400", "particules" => "800000")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("argument non reconnu : $a")
        o[m[1]] = m[2]
    end
    o
end

"""Coupe `z = 0` de la densité, ramenée en `Float32`.

Le plan est choisi au point de collocation le plus proche de zéro : la grille
n'en contient pas forcément un exactement, les points étant aux nœuds de Gauss.
"""
function slice_z0(ρ, mesh)
    gz = mesh.axes[3].colloc
    k = argmin(abs.(gz))
    (Float32.(@view ρ[:, :, k]), k)
end

function main(argv)
    o = parse_args(argv)
    keV = parse(Int, o["kev"])
    nimg = parse(Int, o["images"])
    npart = parse(Int, o["particules"])

    grid, ρr = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
    p = SimulationParameters(nfine = 44, ninner = 22, nouter = 22, rcluster = 78.0,
                             rbox = 235.0, nions = 1000.0, nelectrons = 1000.0,
                             nparticles = npart, nsteps = 0, dt = 1.0)
    energy = keV * KEV
    proj = Projectile(mass = 1836.154, charge = 1.0, energy = energy, impact = 0.0,
                      x0 = -65.0, dt = 1.0, softening = GaussianSoftening(1.0))
    sim = Simulation(p, PotentialProfile(grid, ρr); projectile = proj)

    v = sqrt(2energy / 1836.154)
    nsteps = ceil(Int, 1.1 * (80 + 65) / v)
    stride = max(1, round(Int, nsteps / nimg))

    fine = sim.meshes[1]
    gx = Float32.(fine.axes[1].colloc)
    gy = Float32.(fine.axes[2].colloc)
    ρ0, kplane = slice_z0(sim.ρ[1], fine)

    @printf("Na1000, %d keV (v = %.2f), %d particules, %d pas, une image tous les %d\n",
            keV, v, npart, nsteps, stride)

    frames = Matrix{Float32}[]
    xs = Float64[]; eks = Float64[]
    t0 = time()
    for i in 0:nsteps
        i > 0 && step!(sim; energy = false)
        if i % stride == 0
            push!(frames, first(slice_z0(sim.ρ[1], fine)))
            push!(xs, sim.projectile.position[1])
            push!(eks, kinetic_energy(sim.projectile))
        end
        sim.projectile.position[1] > 80 && break
    end
    @printf("  %d images en %.0f s\n", length(frames), time() - t0)

    out = joinpath(ROOT, "film.jls")
    serialize(out, (; gx, gy, kplane, rho0 = ρ0, frames, xs, eks,
                    keV, v, npart, stride, dt = p.dt,
                    e0 = proj.initial_energy, hartree = HARTREE_TO_EV))
    @printf("→ %s (%.1f Mo)\n", out, filesize(out) / 2^20)
end

main(ARGS)
