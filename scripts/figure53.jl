#!/usr/bin/env julia
"""
Reproduce the stopping-power curve of the thesis: Na₁₀₀₀, σ_ion = 1 a.u.

    julia --project=gpu -t auto scripts/figure53.jl [--particules=200000] [--kev=1,4,9,16,25]

Under `--project=gpu` it picks up **AppleAccelerate**; the trajectory is
unchanged, only faster. ⚠️ It does **not** use the GPU: that path is `Float32`,
and this figure is the one compared against published values.

The quantity plotted is the one the thesis defines — a **local slope at the
centre**, not the total loss:

    dE/dx ≃ [E_k(+Δx/2) − E_k(−Δx/2)] / Δx,   Δx = 4 a.u.

The published values are in [`ref/these/`](../ref/these/): `desdx.dat.1000`
gives the result, `Ekproj.dat.N` the trajectories it is drawn from.

The initial state comes from `rhorad.Na1000.dat`, the archived equilibrium
radial density (October 1998, 998.7 electrons when integrated): the `pot.dat`
that `initialise4` expected did not survive, but the density is enough — see
[`PotentialProfile`](@ref).

⚠️ The force is the **thesis's** (Gaussian), not the Fortran's (ball). That is
the whole subject of anomaly 10.
"""

using Vlasov
using Printf

# AppleAccelerate, when the environment carries it (`gpu/`). One line, ×1.31 on
# a time step — and not only on the GEMMs: the particle loops gain 15–25 %
# because OpenBLAS's thread pool stops competing with them for the cores.
# ⚠️ Never `BLAS.lbt_forward(libacc)` raw: that binds Accelerate's old LAPACK,
# `inv` returns garbage and the cluster explodes. `using` is enough.
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV        # 1 keV in hartree

# Grid: h = 2·rcluster/nfine ≈ 3.55, the resolution validated on Na₁₉₆, but
# widened to hold Na₁₀₀₀ (R = 40 a₀) and the projectile from the moment it enters.
const GRID = (nfine = 44, ncoarse = 22, rcluster = 78.0, rbox = 235.0)
const X0 = -65.0                        # start, as in the archived trajectories

function parse_args(argv)
    o = Dict("particules" => "200000", "kev" => "1,4,9,16,25")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("unrecognised argument: $a")
        o[m[1]] = m[2]
    end
    o
end

"""Run one crossing and return `(xs, eks)` — position and kinetic energy."""
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

    # Enough steps to come out the other side: the distance to cover divided by
    # the velocity, plus a margin — the projectile slows down.
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

"""Local slope at the centre over `Δx`, in eV/a₀ — the thesis's definition.

⚠️ Over four bohr only, this estimator is **very sensitive to sampling noise**:
with 20 000 pseudo-particles it returns negative values while the full
trajectory is correct to 3 %. That is why production runs used 800 000.
[`fitted_power`](@ref) acts as a guard rail.
"""
function stopping_power(xs, eks; Δx = 4.0)
    nearest(t) = argmin(abs.(xs .- t))
    i, j = nearest(-Δx / 2), nearest(Δx / 2)
    (eks[i] - eks[j]) * HARTREE_TO_EV / (xs[j] - xs[i])
end

"""The same slope, by least squares over a wider window — less faithful to the
thesis's recipe, but less noisy. If the two disagree, the statistics are not
sufficient."""
function fitted_power(xs, eks; half = 10.0)
    k = findall(x -> -half <= x <= half, xs)
    length(k) < 3 && return NaN
    x = view(xs, k); e = view(eks, k) .* HARTREE_TO_EV
    x̄ = sum(x) / length(x); ē = sum(e) / length(e)
    -sum((x .- x̄) .* (e .- ē)) / sum(abs2, x .- x̄)
end

"""Published values: `desdx.dat.1000`, a keV column then two measurements."""
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

    @printf("Na1000, σ_ion = 1, %d pseudo-particles, grid %d (h = %.2f a₀), BLAS: %s\n\n",
            npart, GRID.nfine, 2GRID.rcluster / GRID.nfine,
            ACCELERATE ? "Accelerate" : "OpenBLAS")
    @printf("%-5s %-7s %-9s %-9s %-18s %-8s %s\n",
            "keV", "v", "Δx=4", "fit ±10", "thesis", "error", "time")
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
