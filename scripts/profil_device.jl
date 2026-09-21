#!/usr/bin/env julia
"""
Profile of one device step, stage by stage, **in the real order**.

    julia --project=gpu  -t auto scripts/profil_device.jl              # Metal
    julia --project=cuda -t auto scripts/profil_device.jl              # CUDA
    julia --project=gpu  -t auto scripts/profil_device.jl --particules=80000000 --nfine=128

Options: `--particules`, `--nfine`, `--pas` (timed steps), `--chauffe`
(warm-up steps before timing).

This is the resident path's counterpart to `profil_pas.jl`, which profiles the
host one. It replays the body of `_update_forces_resident!` and of `step!` call
by call, with a `synchronize` after each, and accumulates.

⚠️ **A stage-by-stage clock is blind to two things, and says so at the end.**
Synchronising after every stage serialises work the driver would otherwise
overlap, so the sum is an upper bound on the step, not the step; the script
therefore prints a plain untimed `step!` beside it. And a sum of stages never
asks what the GPU was doing while the host worked — `Metal.@profile` once found
the GPU idle 61 % of a step that a table like this one accounted for in full.

⚠️ The cloud must be **warm**. A cloud that has not been advanced is uniformly
sorted and unrepresentatively cheap to deposit; the default 40 warm-up steps
are there for that, not for kernel compilation alone.
"""

using Printf

const OPTS = Dict(a[1] => a[2] for a in
                  (split(s, '='; limit = 2) for s in ARGS if startswith(s, "--"))
                  if length(a) == 2)
opt(k, d) = haskey(OPTS, "--$k") ? parse(Int, OPTS["--$k"]) : d

const NPART  = opt("particules", 8_000_000)
const NFINE  = opt("nfine", 64)
const NSTEPS = opt("pas", 10)
const NWARM  = opt("chauffe", 40)

# Same backend detection as `xenon.jl`: whichever GPU this machine has.
const BACKEND = if Sys.isapple()
    @eval using Metal
    Metal.functional() ? :metal : :cpu
else
    ok = try; @eval using CUDA; CUDA.functional(); catch; false; end
    ok ? :cuda : :cpu
end
BACKEND === :cpu && error("no GPU backend here — this script profiles the device path")
Sys.isapple() && (try; @eval using AppleAccelerate; catch; end)

using Vlasov
using Vlasov.KernelAbstractions: synchronize

devsync() = BACKEND === :cuda ? CUDA.synchronize() : Metal.synchronize()
backend() = BACKEND === :cuda ? CUDABackend() : MetalBackend()

# --- the same sodium-on-xenon setup the README's first run uses --------------
m = NFINE ÷ 2 + 1
n1 = m ÷ 2
params = SimulationParameters(nfine = NFINE, ninner = 2n1, nouter = 2 * (m - n1) - 2,
                              rcluster = 78.0, rbox = 235.0, nions = 196.0,
                              nelectrons = 196.0, nparticles = NPART, nsteps = 0,
                              dt = 0.5)
proj = Projectile(mass = 131.3 * 1836.154, charge = 25.0,
                  energy = 0.5 * 131.3 * 1836.154 * 0.16, impact = 45.0,
                  x0 = -70.0, dt = 0.5, softening = BallSoftening(5.0))
profile = read_radial_profile(joinpath(dirname(pathof(Vlasov)), "..", "ref", "fortran", "data"))

@printf("%s, %.1e particles, fine grid %d^3\n", uppercase(string(BACKEND)), NPART, NFINE)
flush(stdout)
sim = Simulation(params, profile; projectile = proj, backend = backend(),
                 precision = Float32, packed = true)
for _ in 1:NWARM
    Vlasov.step!(sim; energy = false)
end

# --- the step, replayed with a clock between the stages ----------------------
const ORDER = String[]
const MS = Dict{String,Float64}()
function stage(name, f)
    haskey(MS, name) || (MS[name] = 0.0; push!(ORDER, name))
    t0 = time()
    r = f()
    devsync()
    MS[name] += 1000 * (time() - t0)
    r
end

"""One step, each call of `_update_forces_resident!` and `step!` timed in place."""
function timed_step!(sim)
    dev = sim.device
    acc = dev.accelerator
    fine, coarse = sim.meshes[1], sim.meshes[2]
    dmf, dmc = dev.meshes
    w = sim.cloud.weight
    pos = sim.cloud.positions

    stage("pack + sort", () -> Vlasov._pack!(acc, pos))
    nout = stage("fill columns", () -> Vlasov._fill_columns!(acc, fine, sim.smoothing, pos))
    ncell = length(acc.sorter.occupied)
    stage("fine deposit", function ()
        fill!(dev.ρ[1], zero(eltype(dev.ρ[1])))
        Vlasov._deposit_sorted_kernel!(acc.backend, Vlasov.DEPOSIT_GROUPSIZE)(
            dev.ρ[1], acc.nodes, acc.cols.device, acc.cells.device,
            acc.bounds.device, Int32(acc.sorter.nknots);
            ndrange = ncell * Vlasov.DEPOSIT_GROUPSIZE)
        synchronize(acc.backend)
        q = Vlasov.total_charge(dev.ρ[1], acc.grid)
        E = eltype(dev.ρ[1])
        vec(dev.ρ[1]) .*= E((length(pos) - nout) * w / q)
    end)

    fine_ax = fine.axes[1]
    hf = (fine_ax.knots[end] - fine_ax.knots[1]) / (length(fine_ax.knots) - 1)
    stage("coarse deposit (CIC)", () ->
        Vlasov.deposit_cic!(dev.ρ[2], dmc, acc, length(pos), w, fine_ax.knots[1], hf))

    stage("poisson!", () -> Vlasov.poisson!(dev.φ, dev.ρ, dev.meshes))
    csolf = stage("csolf", () -> Vlasov.spline_coefficients!(dev.csol[1], dev.φ[1], dmf))
    csolc = stage("csolc", () -> Vlasov.spline_coefficients!(dev.csol[2], dev.φ[2], dmc))
    stage("mean field", function ()
        Vlasov.effective_potential!(csolf, dev.ρ[1], dmf, sim.jellium)
        Vlasov.effective_potential!(csolc, dev.ρ[2], dmc, sim.jellium)
    end)
    stage("csolc → host", () -> copyto!(dev.csolc_host, csolc))
    stage("forces + projectile", function ()
        Vlasov.forces!(sim.cloud, acc, fine.axes, csolf, coarse.axes,
                       dev.csolc_host, sim.smoothing; projectile = sim.projectile,
                       packed = true)
        Vlasov.advance_projectile!(sim, acc)
    end)
    stage("φ ← csol", function ()
        copyto!(dev.φ[1], csolf)
        copyto!(dev.φ[2], csolc)
    end)
    stage("Verlet", () -> Vlasov._advance_cloud!(sim))
    nothing
end

timed_step!(sim)                       # compile the closures
empty!(MS); empty!(ORDER)
for _ in 1:NSTEPS
    timed_step!(sim)
end

# The step as it really runs, with nothing synchronising between the stages.
Vlasov.step!(sim; energy = false)
devsync()
t0 = time()
for _ in 1:NSTEPS
    Vlasov.step!(sim; energy = false)
end
devsync()
plain = 1000 * (time() - t0) / NSTEPS

total = sum(values(MS)) / NSTEPS
println("\n stage                       ms      %")
for k in sort(ORDER; by = k -> -MS[k])
    @printf("  %-22s %7.2f  %5.1f\n", k, MS[k] / NSTEPS, 100 * MS[k] / NSTEPS / total)
end
@printf("  %-22s %7.2f  %5.1f\n", "— sum of the stages", total, 100.0)
@printf("\n  plain step!, unsynchronised  %7.2f ms   (the stages overstate it by %.0f %%)\n",
        plain, 100 * (total - plain) / plain)
