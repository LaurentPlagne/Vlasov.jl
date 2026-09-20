#!/usr/bin/env julia
"""
Na₁₉₆ + Xe²⁵⁺, 500 keV, b = 45 a₀ — the 1997 Springer collision, as a first run.

    julia --project=.   scripts/xenon.jl                        # CPU
    julia --project=gpu scripts/xenon.jl --particules=8000000    # GPU, if Metal is there

A multicharged xenon ion grazes a sodium cluster. Its field tears an electron
bridge out of the cloud, part of which it carries away: the run prints, every
few steps, the charge sitting within 8 a₀ of the ion. That number rising from
zero and settling is the capture.

Options (their names are the script's interface, hence French, like the other
scripts here):

  * `--particules=N`  pseudo-particles (default 5×10⁵ on CPU, 8×10⁶ on GPU —
    each is about two minutes of wall clock for the whole crossing);
  * `--nfine=n`       intervals of the fine grid — **even**, see below
    (default 44, i.e. 90³, on CPU; 64, i.e. 130³, on GPU);
  * `--pas=N`         time steps (default 700, the whole crossing);
  * `--tous=N`        print one line every N steps.

⚠️ **The grid comes in pairs.** The coarse mesh must carry exactly as many
basis functions as the fine one, so `ninner` and `nouter` are derived from
`nfine` here rather than given. `nfine` must be **even**: an odd one leaves the
two meshes one function apart and the run stops on
`DimensionMismatch: ρ must cover the whole collocation grid`.

⚠️ **This script announces which path it runs on**, and that line is worth
reading: a missing `Metal` or `AppleAccelerate` is silent otherwise, and the
same command then measures something else entirely.
"""

using Vlasov
using Printf

# Optional, and loaded before anything is timed. `Metal` carries the GPU path;
# `AppleAccelerate` is worth ×1.31 on the CPU one — not only on the GEMMs, the
# particle loops gain 15–25 % because OpenBLAS's threads stop competing.
const METAL = try; @eval using Metal; true; catch; false; end
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end

const ROOT = dirname(@__DIR__)
const FS = 41.34137          # atomic units of time per femtosecond

function parse_args(argv)
    opts = Dict("particules" => "0", "nfine" => "0", "pas" => "700", "tous" => "50")
    for a in argv
        m = match(r"^--([a-z]+)=([0-9]+)$", a)
        m === nothing && error("unrecognised argument: $a (try --particules=2000000)")
        haskey(opts, m[1]) || error("unknown option: --$(m[1])")
        opts[m[1]] = m[2]
    end
    Dict(k => parse(Int, v) for (k, v) in opts)
end

"""Coarse grid derived from the fine one, so that both carry `2·nfine + 2`
basis functions."""
function grids(nfine)
    iseven(nfine) || error("--nfine must be even (got $nfine)")
    m = nfine ÷ 2 + 1
    n1 = m ÷ 2
    (ninner = 2n1, nouter = 2 * (m - n1) - 2)
end

function main(argv)
    o = parse_args(argv)
    # Defaults sized so that either path is about a minute and a half of wall
    # clock — 5×10⁵ on 90³ for the CPU, 8×10⁶ on 130³ for the GPU.
    npart = o["particules"] > 0 ? o["particules"] : (METAL ? 8_000_000 : 500_000)
    nfine = o["nfine"] > 0 ? o["nfine"] : (METAL ? 64 : 44)
    g = grids(nfine)
    dt = 0.5

    # ⚠️ `flush` after every line. Redirected to a file, Julia's stdout is
    # block-buffered: without this the run shows nothing for minutes and looks
    # hung, which is exactly what a first run must not do.
    @printf("path: %s%s, %d threads\n",
            METAL ? "GPU (Metal)" : "CPU",
            ACCELERATE ? " + AppleAccelerate" : "",
            Threads.nthreads())
    @printf("Na196 + Xe25+, 500 keV, b = 45 a0 | %.1e particles, grid %d^3\n",
            npart, 2nfine + 2)
    flush(stdout)

    p = SimulationParameters(nfine = nfine, ninner = g.ninner, nouter = g.nouter,
                             rcluster = 78.0, rbox = 235.0,
                             nions = 196.0, nelectrons = 196.0,
                             nparticles = npart, nsteps = 0, dt = dt)

    # The 1997 collision: Xe²⁵⁺ at v = 0.40 a.u., grazing at b = 45 a₀. The ball
    # softening of radius 5 is the Fortran's own for a multicharged projectile.
    mass = 131.3 * 1836.154
    v = 0.40
    proj = Projectile(mass = mass, charge = 25.0, energy = 0.5 * mass * v^2,
                      impact = 45.0, x0 = -70.0, dt = dt,
                      softening = BallSoftening(5.0))

    prof = read_radial_profile(joinpath(ROOT, "ref", "fortran", "hm1.dat"),
                               joinpath(ROOT, "ref", "fortran", "rhoinit.dat"))

    t0 = time()
    sim = METAL ?
          Simulation(p, prof; projectile = proj, backend = MetalBackend(),
                     precision = Float32, packed = true) :
          Simulation(p, prof; projectile = proj)
    @printf("built in %.1f s\n\n", time() - t0)

    println("   step     t (fs)    x_ion (a0)   q(<8 a0)    ms/step")
    flush(stdout)
    t0 = time()
    tprev = t0
    for s in 1:o["pas"]
        step!(sim; energy = false)
        if s % o["tous"] == 0 || s == o["pas"]
            now = time()
            @printf("%7d %10.2f %12.1f %10.3f %10.1f\n",
                    s, s * dt / FS, sim.projectile.position[1],
                    enclosed_charge(sim.cloud, sim.projectile, 8.0),
                    1000 * (now - tprev) / o["tous"])
            flush(stdout)
            tprev = now
        end
    end
    @printf("\n%d steps in %.1f s — %.1f ms/step\n",
            o["pas"], time() - t0, 1000 * (time() - t0) / o["pas"])
    @printf("charge carried away by the ion: %.3f electrons\n",
            enclosed_charge(sim.cloud, sim.projectile, 8.0))
end

main(ARGS)
