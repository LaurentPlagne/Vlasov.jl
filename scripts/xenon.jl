#!/usr/bin/env julia
"""
Na₁₉₆ + Xe²⁵⁺, 500 keV, b = 45 a₀ — the 1997 Springer collision, as a first run.

    julia --project=gpu  -t auto scripts/xenon.jl     # Apple Silicon
    julia --project=cuda -t auto scripts/xenon.jl     # an NVIDIA card
    julia --project=viz  -t auto scripts/xenon.jl     # no GPU at all

A multicharged xenon ion grazes a sodium cluster. Its field tears an electron
bridge out of the cloud, part of which it carries away. The run prints the
charge sitting within 8 a₀ of the ion — rising from zero and settling, which is
the capture — and then writes the film of it.

Three options:

  * `--particules=N`  pseudo-particles (default 8×10⁶ on a GPU, 6×10⁵ without);
  * `--nfine=n`       intervals of the fine grid — **even**, see below
                      (default 64, i.e. a 130³ grid, on either path);
  * `--cpu` / `--gpu` force the path instead of taking whatever is there.

It finds the vendor on its own: Metal, CUDA, or the processor.

Written at the root of the repository, all gitignored: `film_xenon.mp4`,
`film_xenon.gif`, `xenon_snapshots.png`, and `xenon_data_cache.jls` — the run
itself, which `scripts/render_xenon_glmakie.jl` redraws without recomputing.

⚠️ **The cache is written and never read back here.** An earlier version loaded
it whenever the file existed, which silently ignored `--particules` and redrew
somebody else's run. A cache that answers a question it was not asked is worse
than no cache.

⚠️ **The grid comes in pairs.** The coarse mesh must carry exactly as many
basis functions as the fine one, so `ninner` and `nouter` are derived from
`nfine` here rather than given. `nfine` must be **even**: an odd one leaves the
two meshes one function apart and the run stops on
`DimensionMismatch: ρ must cover the whole collocation grid`.

⚠️ **This script announces which path it runs on, and which renderer**, and
those lines are worth reading: a missing `Metal`, `AppleAccelerate` or `GLMakie`
is silent otherwise, and the same command then measures something else.

`--rendu=cairo`, `--rendu=gl` or `--rendu=aucun` forces the renderer, or skips
the film entirely. The default takes GLMakie where a display and the package
allow, because the primitive it can draw — a `heatmap`, one texture upload — is
worth ×7.8 over Cairo's `contourf`: 4.0 s against 31.0 for the same 176 frames.
"""

using Vlasov
using Printf
using Serialization

const ROOT = dirname(@__DIR__)
const FS = 41.34137          # atomic units of time per femtosecond
const GREY_RGB = (0.80, 0.80, 0.80)

# --- what this machine offers ------------------------------------------------
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
const FORCE_CPU = "--cpu" in ARGS
const FORCE_GPU = "--gpu" in ARGS
const METAL_HERE = !FORCE_CPU && try; @eval using Metal; true; catch; false; end
const METAL = METAL_HERE && try; @eval Metal.functional(); catch; false; end
const CUDA_HERE = (FORCE_CPU || METAL) ? false :
                  try; @eval using CUDA; true; catch; false; end
const CUDA_OK = CUDA_HERE && try; @eval CUDA.functional(); catch; false; end
const GPU = METAL || CUDA_OK

# ⚠️ `Sys.isapple()` first, for the same reason as `functional()` above:
# `using AppleAccelerate` succeeds on Linux, and the path line then claimed a
# framework that does not exist on that machine. It was only a label, but the
# whole point of that line is to be believed.
const ACCELERATE = Sys.isapple() &&
                   try; @eval using AppleAccelerate; true; catch; false; end

# --- which renderer ----------------------------------------------------------

"""Whether this machine can open the window GLMakie draws into.

macOS always can; elsewhere it takes an X11 or Wayland session, and without one
GLMakie stops at `GLFW: X11: The DISPLAY environment variable is missing`."""
has_display() = Sys.isapple() || haskey(ENV, "DISPLAY") || haskey(ENV, "WAYLAND_DISPLAY")

const RENDU = something(findfirst(a -> startswith(a, "--rendu="), ARGS), 0) == 0 ?
              "" : split(ARGS[findfirst(a -> startswith(a, "--rendu="), ARGS)], '=')[2]
const FILM = RENDU != "aucun"
const WANT_GL = RENDU == "gl" ||
                (RENDU == "" && has_display() && Base.find_package("GLMakie") !== nothing)

# ⚠️ The `using` goes through `@eval` because which one it is is a *runtime*
# decision, and both re-export the same Makie API — so everything below says
# `MK.`, and the two renderers differ only where they genuinely differ.
const MK = if !FILM
    nothing
elseif WANT_GL
    try
        @eval using GLMakie
        @eval GLMakie
    catch e
        @warn "GLMakie unavailable, falling back to CairoMakie" exception = e
        @eval using CairoMakie
        @eval CairoMakie.activate!(type = "png")
        @eval CairoMakie
    end
else
    @eval using CairoMakie
    @eval CairoMakie.activate!(type = "png")
    @eval CairoMakie
end

"""Whether the fast primitive is available — one texture upload per frame,
against a contour tessellation Makie does on the CPU whatever the backend."""
const HEATMAP = MK !== nothing && nameof(MK) === :GLMakie

# --- options -----------------------------------------------------------------

function parse_args(argv)
    opts = Dict("particules" => 0, "nfine" => 0)
    for a in argv
        a in ("--cpu", "--gpu") && continue
        startswith(a, "--rendu=") && continue
        m = match(r"^--([a-z]+)=([0-9]+)$", a)
        m === nothing && error("unrecognised argument: $a (try --particules=2000000)")
        haskey(opts, m[1]) || error("unknown option: --$(m[1])")
        opts[m[1]] = parse(Int, m[2])
    end
    opts
end

"""Coarse grid derived from the fine one, so that both carry `2·nfine + 2`
basis functions."""
function grids(nfine)
    iseven(nfine) || error("--nfine must be even (got $nfine)")
    m = nfine ÷ 2 + 1
    n1 = m ÷ 2
    (ninner = 2n1, nouter = 2 * (m - n1) - 2)
end

# --- drawing -----------------------------------------------------------------

"""Bilinear upsampling of one slice onto a regular `n_out` grid.

⚠️ **This belongs to Cairo, not to the physics.** It exists because `contourf`
on the raw slice shows the cells, and because Cairo cannot interpolate a
non-regular grid at all — the collocation points are not equally spaced. A GPU
backend's sampler does the same pass in the texture unit, so the heatmap path
never calls this."""
function resample_2d(xs, ys, V, n_out_x = 350, n_out_y = 350)
    xs_fine = range(xs[1], xs[end], length = n_out_x)
    ys_fine = range(ys[1], ys[end], length = n_out_y)
    V_fine = Matrix{Float32}(undef, n_out_x, n_out_y)
    nx, ny = length(xs), length(ys)
    for (j, y) in enumerate(ys_fine)
        jy = clamp(searchsortedlast(ys, y), 1, ny - 1)
        uy = clamp(Float32((y - ys[jy]) / (ys[jy+1] - ys[jy])), 0.0f0, 1.0f0)
        for (i, x) in enumerate(xs_fine)
            ix = clamp(searchsortedlast(xs, x), 1, nx - 1)
            tx = clamp(Float32((x - xs[ix]) / (xs[ix+1] - xs[ix])), 0.0f0, 1.0f0)
            V_fine[i, j] = (1 - tx) * (1 - uy) * V[ix, jy] + tx * (1 - uy) * V[ix+1, jy] +
                           (1 - tx) * uy * V[ix, jy+1] + tx * uy * V[ix+1, jy+1]
        end
    end
    (xs_fine, ys_fine, V_fine)
end

function render(data, out_mp4, out_gif, out_snapshots)
    GREY = MK.RGBf(GREY_RGB...)
    nframes = length(data.frames)
    rho_bulk = 0.00373f0
    c_min, c_max = 0.02f0 * rho_bulk, 1.45f0 * rho_bulk
    levels = range(c_min, c_max, length = 45)

    if HEATMAP
        xs_f, ys_f, frames = data.xs, data.ys, data.frames
    else
        println("Pre-resampling $nframes frames to 350×350 for Cairo...")
        flush(stdout)
        xs_f, ys_f, _ = resample_2d(data.xs, data.ys, data.frames[1])
        frames = [resample_2d(data.xs, data.ys, f)[3] for f in data.frames]
    end

    """The density panel, by whichever primitive this renderer can draw."""
    density!(ax, field) = HEATMAP ?
        MK.heatmap!(ax, xs_f, ys_f, field; colormap = :turbo,
                    colorrange = (c_min, c_max), lowclip = GREY, interpolate = true) :
        MK.contourf!(ax, xs_f, ys_f, field; levels = levels, colormap = :turbo,
                     extendlow = GREY)

    θ = range(0, 2π, length = 100)
    jellium!(ax, w) = MK.lines!(ax, data.r_jel .* cos.(θ), data.r_jel .* sin.(θ);
                                color = :white, linewidth = w, linestyle = :dash)

    # --- the six-panel strip, on the model of the 1997 figure ---------------
    fig = MK.Figure(size = (1100, 750), backgroundcolor = :white)
    MK.Label(fig[0, 1:3],
             "Snapshots of Electron Density during Na₁₉₆ + Xe²⁵⁺ Collision (500 keV, b = 45 a₀)\n" *
             (HEATMAP ? "(Interpolated Density Map)" : "(Smooth Iso-Contour Representation)"),
             fontsize = 18, font = :bold)
    for (k, tt) in enumerate((1.2, 2.5, 3.8, 4.8, 6.0, 7.5))
        fi = argmin(abs.(data.times_fs .- tt))
        row, col = (k - 1) ÷ 3 + 1, (k - 1) % 3 + 1
        ax = MK.Axis(fig[row, col],
                     title = @sprintf("t = %.2f fs (x_ion = %+.1f a₀)",
                                      data.times_fs[fi], data.proj_xs[fi]),
                     aspect = MK.DataAspect(), backgroundcolor = GREY,
                     xlabel = row == 2 ? "x (a₀)" : "",
                     ylabel = col == 1 ? "y (a₀)" : "")
        density!(ax, frames[fi])
        jellium!(ax, 1.5)
        MK.scatter!(ax, [data.proj_xs[fi]], [data.proj_ys[fi]];
                    color = :yellow, strokecolor = :black, strokewidth = 2,
                    markersize = 12)
        MK.xlims!(ax, -65, 65); MK.ylims!(ax, -35, 65)
    end
    MK.save(out_snapshots, fig, px_per_unit = 2)
    println("wrote $out_snapshots")
    flush(stdout)

    # --- the film -----------------------------------------------------------
    fig = MK.Figure(size = (900, 850), backgroundcolor = :white)
    idx = MK.Observable(1)

    ax1 = MK.Axis(fig[1, 1], aspect = MK.DataAspect(), backgroundcolor = GREY,
                  title = MK.lift(i -> @sprintf("Na₁₉₆ + Xe²⁵⁺ (500 keV, b = 45 a₀) — t = %.2f fs, x_ion = %+.1f a₀",
                                                data.times_fs[i], data.proj_xs[i]), idx),
                  xlabel = "x (a₀)", ylabel = "y (a₀)")
    density!(ax1, MK.lift(i -> frames[i], idx))
    jellium!(ax1, 2.0)
    MK.scatter!(ax1, MK.lift(i -> MK.Point2f(data.proj_xs[i], data.proj_ys[i]), idx);
                color = :yellow, strokecolor = :black, strokewidth = 2, markersize = 14)
    MK.xlims!(ax1, -65, 65); MK.ylims!(ax1, -35, 65)

    ax2 = MK.Axis(fig[2, 1], xlabel = "t (fs)", ylabel = "charge (e)",
                  title = "Dynamic charge transfer: capture and cluster ionization")
    MK.lines!(ax2, data.times_fs, data.q_caps, color = (:dodgerblue, 0.4), linewidth = 1.5)
    MK.lines!(ax2, data.times_fs, data.q_nets, color = (:darkorange, 0.4), linewidth = 1.5)
    MK.lines!(ax2, MK.lift(i -> data.times_fs[1:i], idx), MK.lift(i -> data.q_caps[1:i], idx);
              color = :dodgerblue, linewidth = 2.5, label = "Q_cap (R = 8 a₀)")
    MK.lines!(ax2, MK.lift(i -> data.times_fs[1:i], idx), MK.lift(i -> data.q_nets[1:i], idx);
              color = :darkorange, linewidth = 2.5, label = "Q_cluster")
    MK.scatter!(ax2, MK.lift(i -> [MK.Point2f(data.times_fs[i], data.q_caps[i])], idx);
                color = :dodgerblue, markersize = 10)
    MK.scatter!(ax2, MK.lift(i -> [MK.Point2f(data.times_fs[i], data.q_nets[i])], idx);
                color = :darkorange, markersize = 10)
    MK.xlims!(ax2, 0, data.times_fs[end])
    MK.ylims!(ax2, -0.5, max(14.5, 1.1 * maximum(data.q_nets)))
    MK.axislegend(ax2, position = :lt)

    t0 = time()
    MK.record(fig, out_mp4, 1:nframes; framerate = 25) do i
        idx[] = i
    end
    @printf("wrote %s — %d frames in %.1f s (%.1f frames/s)\n",
            out_mp4, nframes, time() - t0, nframes / (time() - t0))
    flush(stdout)
    MK.record(fig, out_gif, 1:2:nframes; framerate = 15) do i
        idx[] = i
    end
    println("wrote $out_gif")
end

# --- the run -----------------------------------------------------------------

function main(argv)
    o = parse_args(argv)
    FORCE_GPU && !GPU &&
        error("--gpu asked for, but no device answers `functional()` here. " *
              "Run with --project=gpu on Apple Silicon, --project=cuda on an " *
              "NVIDIA card, or drop the flag to take the processor.")
    # Defaults sized for the film: the same 130³ grid on either path, so the
    # picture is the same and only the noise differs.
    npart = o["particules"] > 0 ? o["particules"] : (GPU ? 8_000_000 : 600_000)
    nfine = o["nfine"] > 0 ? o["nfine"] : 64
    g = grids(nfine)
    dt = 0.5
    v = 0.40
    x0 = -70.0
    nsteps = ceil(Int, (abs(x0) + 70.0) / (v * dt))
    every = 50          # steps between two printed lines
    stride = 4          # steps between two captured frames

    # ⚠️ `flush` after every line. Redirected to a file, Julia's stdout is
    # block-buffered: without this the run shows nothing for minutes and looks
    # hung, which is exactly what a first run must not do.
    @printf("path: %s%s, %d threads | renderer: %s\n",
            METAL ? "GPU (Metal)" : CUDA_OK ? "GPU (CUDA)" : "CPU",
            ACCELERATE ? " + AppleAccelerate" : "",
            Threads.nthreads(),
            !FILM ? "none, --rendu=aucun" :
            string(nameof(MK), HEATMAP ? ", heatmap" : ", contourf"))
    @printf("Na196 + Xe25+, 500 keV, b = 45 a0 | %.1e particles, grid %d^3\n",
            npart, 2nfine + 2)
    # A GPU package that is installed but has nothing to talk to is the most
    # confusing way to end up on the CPU — say so rather than let the wall
    # clock be the only clue.
    if !GPU && !FORCE_CPU && METAL_HERE
        println("  (Metal is installed but finds no Apple GPU here; " *
                "for an NVIDIA card, run with --project=cuda)")
    elseif !GPU && !FORCE_CPU && CUDA_HERE
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
    proj = Projectile(mass = mass, charge = 25.0, energy = 0.5 * mass * v^2,
                      impact = 45.0, x0 = x0, dt = dt,
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
        Simulation(p, prof; projectile = proj, backend = @eval(CUDABackend()),
                   precision = Float32, packed = true)
    else
        Simulation(p, prof; projectile = proj)
    end
    @printf("built in %.1f s\n\n", time() - t0)

    fine = sim.meshes[1]
    xs = collocation_points(fine.axes[1].knots)
    ys = collocation_points(fine.axes[2].knots)
    iz1, iz2 = nfine, nfine + 1          # the reaction plane, z ≈ 0
    frames = Matrix{Float32}[]
    times_fs = Float64[]; proj_xs = Float64[]; proj_ys = Float64[]
    q_caps = Float64[]; q_nets = Float64[]

    # `out` is the number of particles whose 10³ stencil no longer fits in the
    # fine grid: they take the coarse path, on the host. It starts at zero and
    # grows as the cloud spreads — and it is the size of the only per-step
    # host/device traffic left, so it is worth watching on a discrete GPU.
    println("   step     t (fs)    x_ion (a0)   q(<8 a0)        out    ms/step")
    flush(stdout)
    t0 = time()
    tprev = t0
    tside = 0.0        # time spent reading the run back, since the last line
    tread = 0.0        # and in total
    for s in 1:nsteps
        step!(sim; energy = false)
        capture = FILM && (s % stride == 0 || s == 1)
        report = s % every == 0 || s == nsteps
        (capture || report) || continue

        # ⚠️ Everything here reads the simulation **on the host**: the density
        # grid, which a resident run keeps on the device, and the cloud, whose
        # host half is a different array on a discrete GPU. Both are no-ops on
        # unified memory, and neither belongs in the step time — which is why
        # they are timed out of it rather than left to inflate `ms/step`.
        tmark = time()
        if GPU
            capture && Vlasov.sync_host!(sim)
            Vlasov.download!(sim.device.accelerator.particles, sim.device.backend)
        end
        qcap = enclosed_charge(sim.cloud, sim.projectile, 8.0)
        if capture
            push!(frames, Float32.((sim.ρ[1][:, :, iz1] .+ sim.ρ[1][:, :, iz2]) ./ 2))
            push!(times_fs, s * dt / FS)
            push!(proj_xs, sim.projectile.position[1])
            push!(proj_ys, sim.projectile.position[2])
            push!(q_caps, qcap)
            push!(q_nets, 196.0 - sim.cloud.weight *
                          count(pt -> sum(abs2, pt) < 35.0^2, sim.cloud.positions))
        end
        tside += time() - tmark

        if report
            now = time()
            nout = GPU ? Int(sim.device.accelerator.outcount.host[1]) : 0
            @printf("%7d %10.2f %12.1f %10.3f %10d %10.1f\n",
                    s, s * dt / FS, sim.projectile.position[1], qcap, nout,
                    1000 * (now - tprev - tside) / every)
            flush(stdout)
            tprev = now
            tread += tside
            tside = 0.0
        end
    end
    elapsed = time() - t0
    @printf("\n%d steps in %.1f s — %.1f ms/step",
            nsteps, elapsed - tread, 1000 * (elapsed - tread) / nsteps)
    FILM && @printf(", plus %.1f s reading %d frames back", tread, length(frames))
    println()
    @printf("charge carried away by the ion: %.3f electrons\n",
            enclosed_charge(sim.cloud, sim.projectile, 8.0))
    flush(stdout)

    FILM || return
    cache = joinpath(ROOT, "xenon_data_cache.jls")
    d = (; frames, times_fs, proj_xs, proj_ys, q_caps, q_nets, xs, ys,
         r_jel = 196.0^(1 / 3) * 4.0)
    serialize(cache, d)
    println("wrote $cache")
    render(d, joinpath(ROOT, "film_xenon.mp4"), joinpath(ROOT, "film_xenon.gif"),
           joinpath(ROOT, "xenon_snapshots.png"))
end

main(ARGS)
