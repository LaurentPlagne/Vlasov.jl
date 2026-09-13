#!/usr/bin/env julia
"""
Traversée d'un agrégat par un projectile, et pouvoir d'arrêt `dE/dx`.

    julia --project=. scripts/traversee.jl [--profil=radial|rejet] [--pas=600]
                                          [--graine=-1] [--sigma=0] [--n=196]

`--sigma` bascule l'adoucissement projectile ↔ pseudo-particule : `0` garde la
boule uniforme du Fortran (rayon `cutoff = 1`), une valeur positive prend la
gaussienne de la thèse, de largeur `σ_ion`. Les deux ne donnent pas le même
pouvoir d'arrêt, et c'est le sujet.

`--profil` choisit l'échantillonnage initial, donc la version du code
reproduite : `radial` est le `initialise` de 1997 (inversion d'une densité
tabulée, `ref/fortran/`), `rejet` le `initialise4` de 1998 (rejet dans
l'espace des phases, `ref/fortran98/pot.dat`). C'est le seul changement de
physique entre la version portée et la cible — ce script est là pour mesurer
ce qu'il déplace.

`dE/dx` est rapporté de deux façons, parce que la thèse ne dit pas laquelle
elle trace : perte totale sur la traversée `[−R, +R]`, et plateau local au
cœur. Voir `docs/validation-chapitre6.md`.
"""

using Vlasov
using Printf

const ROOT = dirname(@__DIR__)

function parse_args(argv)
    opts = Dict("profil" => "radial", "pas" => "600", "graine" => "-1",
                "sigma" => "0", "n" => "0")
    for a in argv
        m = match(r"^--([a-z]+)=(-?[a-z0-9.]+)$", a)
        m === nothing && error("argument non reconnu : $a")
        haskey(opts, m[1]) || error("option inconnue : --$(m[1])")
        opts[m[1]] = m[2]
    end
    opts
end

function build(profil, params)
    profil == "radial" && return read_radial_profile(joinpath(ROOT, "ref", "fortran"))
    profil == "rejet" && return read_potential_profile(
        joinpath(ROOT, "ref", "fortran98", "pot.dat"))
    error("profil inconnu : $profil (radial ou rejet)")
end

function main(argv)
    opts = parse_args(argv)
    nsteps = parse(Int, opts["pas"])
    seed = parse(Int, opts["graine"])
    σion = parse(Float64, opts["sigma"])

    params = read_parameters(joinpath(ROOT, "ref", "fortran", "vlas.inp"))
    profile = build(opts["profil"], params)

    # Paramètres du projectile : ceux de `vlas.inp`, proton 2 keV, impact nul.
    soft = σion > 0 ? GaussianSoftening(σion) : BallSoftening(1.0)
    proj = Projectile(mass = 1836.154, charge = 1.0, energy = 73.498,
                      impact = 0.0, x0 = -70.0, dt = params.dt, softening = soft)

    @printf("profil = %s, graine = %d, %d particules, %d pas, dt = %g\n",
            opts["profil"], seed, params.nparticles, nsteps, params.dt)
    @printf("adoucissement : %s\n", soft)
    sim = Simulation(params, profile; projectile = proj, rng = Ran2(seed))

    # Trajectoire : x et énergie cinétique à chaque pas.
    xs = Float64[proj.position[1]]
    es = Float64[kinetic_energy(proj)]
    t0 = time()
    for _ in 1:nsteps
        step!(sim)
        push!(xs, proj.position[1])
        push!(es, kinetic_energy(proj))
    end
    @printf("  %d pas en %.1f s\n", nsteps, time() - t0)

    R = WIGNER_SEITZ_NA * cbrt(params.nions)
    H = HARTREE_TO_EV

    """Perte entre les deux premiers instants où `x` franchit `a` puis `b`."""
    function loss(a, b)
        i = findfirst(>=(a), xs); j = findfirst(>=(b), xs)
        (i === nothing || j === nothing || j <= i) && return (NaN, NaN)
        ((es[i] - es[j]) * H, xs[j] - xs[i])
    end

    @printf("\nrayon de l'agrégat (r_s·N^⅓) : %.2f a₀\n", R)
    @printf("perte totale : %.2f eV\n", (es[1] - es[end]) * H)
    for (a, b, nom) in ((-2.0, 2.0, "centre Δx=4 (thèse)"), (-R, R, "traversée ±R"),
                        (-10.0, 10.0, "cœur ±10"), (xs[1], xs[end], "trajectoire entière"))
        ΔE, Δx = loss(a, b)
        isnan(ΔE) || @printf("  %-20s ΔE = %7.2f eV sur %6.1f a₀  →  dE/dx = %.4f eV/a₀\n",
                             nom, ΔE, Δx, ΔE / Δx)
    end
end

main(ARGS)
