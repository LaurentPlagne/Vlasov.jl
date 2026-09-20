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

# Optional, and loaded before anything is timed. `AppleAccelerate` is worth
# ×1.31 on the CPU path — not only on the GEMMs, the particle loops gain 15–25 %
# because OpenBLAS's threads stop competing for the cores.
#
# ⚠️ **Loading the package is not the same as having the hardware.** `using
# Metal` *succeeds* on a machine that has no Apple GPU — it only logs an error —
# so a check that merely imported it announced the GPU path on a Linux box with
# an NVIDIA card, and failed later and elsewhere. Both vendors answer
# `functional()`, and that is the question being asked.
#
# ⚠️ Each probe is wrapped in its own `@eval`: inside one, the `using` and the
# call would be lowered together, against a world where the package's name is
# not yet bound.
const METAL_HERE = try; @eval using Metal; true; catch; false; end
const METAL = METAL_HERE && try; @eval Metal.functional(); catch; false; end
const CUDA_HERE = METAL ? false : try; @eval using CUDA; true; catch; false; end
const CUDA_OK = CUDA_HERE && try; @eval CUDA.functional(); catch; false; end
const GPU = METAL || CUDA_OK

# ⚠️ `Sys.isapple()` first, for the same reason as `functional()` above:
# `using AppleAccelerate` succeeds on Linux, and the path line then claimed a
# framework that does not exist on that machine. It was only a label, but the
# whole point of that line is to be believed.
const ACCELERATE = Sys.isapple() &&
                   try; @eval using AppleAccelerate; true; catch; false; end

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
    npart = o["particules"] > 0 ? o["particules"] : (GPU ? 8_000_000 : 500_000)
    nfine = o["nfine"] > 0 ? o["nfine"] : (GPU ? 64 : 44)
    g = grids(nfine)
    dt = 0.5

    # ⚠️ `flush` after every line. Redirected to a file, Julia's stdout is
    # block-buffered: without this the run shows nothing for minutes and looks
    # hung, which is exactly what a first run must not do.
    @printf("path: %s%s, %d threads\n",
            METAL ? "GPU (Metal)" : CUDA_OK ? "GPU (CUDA)" : "CPU",
            ACCELERATE ? " + AppleAccelerate" : "",
            Threads.nthreads())
    @printf("Na196 + Xe25+, 500 keV, b = 45 a0 | %.1e particles, grid %d^3\n",
            npart, 2nfine + 2)
    # A GPU package that is installed but has nothing to talk to is the most
    # confusing way to end up on the CPU — say so rather than let the wall
    # clock be the only clue.
    if !GPU && METAL_HERE
        println("  (Metal is installed but finds no Apple GPU here; " *
                "for an NVIDIA card, run with --project=cuda)")
    elseif !GPU && CUDA_HERE
        println("  (CUDA is installed but finds no device here)")
    end
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

    # ⚠️ `ref/fortran/data/`, and not `ref/fortran/`: the Fortran's Makefile
    # copies these two files to the directory above when it builds the oracle,
    # and those copies are gitignored. Reading them there worked on the machine
    # that had run the Fortran, and only there — a fresh clone got a
    # `SystemError` on the very first command the README gives.
    data = joinpath(ROOT, "ref", "fortran", "data")
    isfile(joinpath(data, "hm1.dat")) ||
        error("cannot find the cluster profile in $data — is this a complete " *
              "clone of the repository?")
    prof = read_radial_profile(data)

    t0 = time()
    # `@eval` again, for the same world-age reason: the backend type is bound
    # by the `using` above, which ran after this function was compiled.
    sim = if METAL
        Simulation(p, prof; projectile = proj, backend = @eval(MetalBackend()),
                   precision = Float32, packed = true)
    elseif CUDA_OK
        # ⚠️ Untested: no CUDA hardware where this was written. The kernels are
        # `KernelAbstractions` and the accelerator takes any backend, so this is
        # expected to work — `Float32` to stay on the path that is measured,
        # though CUDA would also give `Float64`.
        Simulation(p, prof; projectile = proj, backend = @eval(CUDABackend()),
                   precision = Float32, packed = true)
    else
        Simulation(p, prof; projectile = proj)
    end
    @printf("built in %.1f s\n\n", time() - t0)

    println("   step     t (fs)    x_ion (a0)   q(<8 a0)    ms/step")
    flush(stdout)
    t0 = time()
    tprev = t0
    for s in 1:o["pas"]
        step!(sim; energy = false)
        if s % o["tous"] == 0 || s == o["pas"]
            now = time()
            # ⚠️ The diagnostic below reads the cloud **on the host**, and on a
            # discrete GPU that is a different array from the one the kernels
            # move particles in. Unified memory hides this; CUDA would not, and
            # the capture curve would be whatever was uploaded at construction.
            # A no-op where the two halves are the same bytes.
            GPU && Vlasov.download!(sim.device.accelerator.particles,
                                    sim.device.backend)
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
