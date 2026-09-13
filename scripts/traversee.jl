#!/usr/bin/env julia
"""
A projectile crossing a cluster, and the stopping power `dE/dx`.

    julia --project=. scripts/traversee.jl [--profil=radial|rejet] [--pas=600]
                                          [--graine=-1] [--sigma=0] [--n=196]

(The command-line flags keep their French names — `profil`, `pas`, `graine`
— since they are the script's interface, not prose.)

`--sigma` switches the projectile ↔ pseudo-particle softening: `0` keeps the
Fortran's uniform ball (radius `cutoff = 1`), a positive value takes the
thesis's Gaussian, of width `σ_ion`. The two do not give the same stopping
power, and that is the point.

`--profil` selects the initial sampling, hence which version of the code is
reproduced: `radial` is the 1997 `initialise` (inversion of a tabulated
density, `ref/fortran/`), `rejet` the 1998 `initialise4` (rejection sampling
in phase space, `ref/fortran98/pot.dat`). This is the only change of physics
between the ported version and the target — this script is here to measure
what it shifts.

`dE/dx` is reported two ways, because the thesis does not say which one it
plots: total loss over the crossing `[−R, +R]`, and the local plateau at the
core. See `docs/validation-chapitre6.md`.
"""

using Vlasov
using Printf

const ROOT = dirname(@__DIR__)

function parse_args(argv)
    opts = Dict("profil" => "radial", "pas" => "600", "graine" => "-1",
                "sigma" => "0", "n" => "0")
    for a in argv
        m = match(r"^--([a-z]+)=(-?[a-z0-9.]+)$", a)
        m === nothing && error("unrecognised argument: $a")
        haskey(opts, m[1]) || error("unknown option: --$(m[1])")
        opts[m[1]] = m[2]
    end
    opts
end

function build(profil, params)
    profil == "radial" && return read_radial_profile(joinpath(ROOT, "ref", "fortran"))
    profil == "rejet" && return read_potential_profile(
        joinpath(ROOT, "ref", "fortran98", "pot.dat"))
    error("unknown profile: $profil (radial or rejet)")
end

function main(argv)
    opts = parse_args(argv)
    nsteps = parse(Int, opts["pas"])
    seed = parse(Int, opts["graine"])
    σion = parse(Float64, opts["sigma"])

    params = read_parameters(joinpath(ROOT, "ref", "fortran", "vlas.inp"))
    profile = build(opts["profil"], params)

    # Projectile parameters: those of `vlas.inp`, 2 keV proton, zero impact.
    soft = σion > 0 ? GaussianSoftening(σion) : BallSoftening(1.0)
    proj = Projectile(mass = 1836.154, charge = 1.0, energy = 73.498,
                      impact = 0.0, x0 = -70.0, dt = params.dt, softening = soft)

    @printf("profile = %s, seed = %d, %d particles, %d steps, dt = %g\n",
            opts["profil"], seed, params.nparticles, nsteps, params.dt)
    @printf("softening: %s\n", soft)
    sim = Simulation(params, profile; projectile = proj, rng = Ran2(seed))

    # Trajectory: x and kinetic energy at every step.
    xs = Float64[proj.position[1]]
    es = Float64[kinetic_energy(proj)]
    t0 = time()
    for _ in 1:nsteps
        step!(sim)
        push!(xs, proj.position[1])
        push!(es, kinetic_energy(proj))
    end
    @printf("  %d steps in %.1f s\n", nsteps, time() - t0)

    R = WIGNER_SEITZ_NA * cbrt(params.nions)
    H = HARTREE_TO_EV

    """Loss between the first two instants where `x` crosses `a` then `b`."""
    function loss(a, b)
        i = findfirst(>=(a), xs); j = findfirst(>=(b), xs)
        (i === nothing || j === nothing || j <= i) && return (NaN, NaN)
        ((es[i] - es[j]) * H, xs[j] - xs[i])
    end

    @printf("\ncluster radius (r_s·N^⅓): %.2f a₀\n", R)
    @printf("total loss: %.2f eV\n", (es[1] - es[end]) * H)
    for (a, b, label) in ((-2.0, 2.0, "centre Δx=4 (thesis)"), (-R, R, "crossing ±R"),
                          (-10.0, 10.0, "core ±10"), (xs[1], xs[end], "whole trajectory"))
        ΔE, Δx = loss(a, b)
        isnan(ΔE) || @printf("  %-20s ΔE = %7.2f eV over %6.1f a₀  →  dE/dx = %.4f eV/a₀\n",
                             label, ΔE, Δx, ΔE / Δx)
    end
end

main(ARGS)
