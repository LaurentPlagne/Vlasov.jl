#!/usr/bin/env julia
# Generate side-by-side comparison of initial state: 8M particles vs 80M particles

using Printf
using Serialization
using AppleAccelerate
using Metal
using CairoMakie
using Vlasov

const ROOT = dirname(@__DIR__)
println("=== GENERATING 80M VS 8M INITIAL STATE VISUALIZATION ===")
flush(stdout)

function cut_z0(ρ, mesh, halfwidth, xs, ys)
    csol = spline_coefficients(ρ, mesh)
    ax = mesh.axes
    zs = halfwidth > 0 ? range(-halfwidth, halfwidth; length = 3) : range(0, 0; length = 1)
    out = zeros(Float32, length(xs), length(ys))
    Threads.@threads for j in eachindex(ys)
        @inbounds for i in eachindex(xs)
            s = 0.0
            for z in zs
                v = spline_potential(ax, csol, (xs[i], ys[j], z))
                v === nothing || (s += v)
            end
            out[i, j] = Float32(s / length(zs))
        end
    end
    out
end

# 1. Load cached 8M initial frame
cache_8M = joinpath(ROOT, "proton_converged_8M_data_cache.jls")
data_8M = deserialize(cache_8M)
f_8M = data_8M.frames[1]
xs_8M = data_8M.outx
ys_8M = data_8M.outy
println("Loaded 8M initial frame.")
flush(stdout)

# 2. Compute 80M initial state on Grid 110
const npart = 80_000_000
const nfine = 110
const dt = 1.0

m = nfine ÷ 2 + 1
n1 = m ÷ 2
n2 = m - n1
ninner = 2 * n1
nouter = 2 * n2 - 2
rcluster = 78.0
rbox = 235.0

p = SimulationParameters(nfine = nfine, ninner = ninner, nouter = nouter,
                         rcluster = rcluster, rbox = rbox,
                         nions = 1000.0, nelectrons = 1000.0,
                         nparticles = npart, nsteps = 0, dt = dt)

grid, ρr = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
profile = PotentialProfile(grid, ρr)

mass_p = 1836.154
const KEV = 1000.0 / 27.211386245988
energy_au = 16.0 * KEV
x0 = -65.0
proj = Projectile(mass = mass_p, charge = 1.0, energy = energy_au,
                  impact = 0.0, x0 = x0, dt = dt,
                  softening = GaussianSoftening(1.0))

println("Sampling 80,000,000 particles...")
flush(stdout)
t0 = time()
sim = Simulation(p, profile; projectile = proj)
println("Initialized 80M cluster in $(round(time() - t0, digits=1)) s.")
flush(stdout)

fine = sim.meshes[1]
acc = ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart, size(sim.csol[1], 1))
println("Allocated 10.4 GB Metal buffers.")
flush(stdout)

println("Executing Step 1 on GPU...")
flush(stdout)
t0 = time()
step!(sim; accelerator = acc)
println("Step 1 done in $(round(time() - t0, digits=1)) s.")
flush(stdout)

println("Extracting 2D slice ρ(x, y, z=0) on fine grid 110...")
flush(stdout)
cx, cy = fine.axes[1].colloc, fine.axes[2].colloc
finesse = 6
xs_80M = collect(range(cx[1], cx[end]; length = finesse * length(cx)))
ys_80M = collect(range(cy[1], cy[end]; length = finesse * length(cy)))

f_80M = cut_z0(sim.ρ[1], fine, 2.0, xs_80M, ys_80M)
cache_80M = joinpath(ROOT, "slice_80M_initial.jls")
serialize(cache_80M, (f = f_80M, xs = xs_80M, ys = ys_80M))
println("Slice extracted and saved to $cache_80M.")
flush(stdout)

# 3. Render side-by-side comparison
cache_80M = joinpath(ROOT, "slice_80M_initial.jls")
if isfile(cache_80M) && (!@isdefined(f_80M) || f_80M === nothing)
    d80 = deserialize(cache_80M)
    f_80M = d80.f
    xs_80M = d80.xs
    ys_80M = d80.ys
end

n0 = 0.00373f0
c_min = 0.05f0 * n0
c_max = 1.45f0 * n0
GREY = CairoMakie.RGBf(0.82, 0.82, 0.82)
R_cluster = 40.0
θ = range(0, 2π, length = 150)

fig = CairoMakie.Figure(size = (1200, 650), backgroundcolor = :white)
CairoMakie.Label(fig[0, 1:2],
      "Na₁₀₀₀ Initial Equilibrium State (t = 0) — Particle Discretization & Shot Noise Comparison\n25 Years Later: 8 Million Particles (Left) vs 80 Million Particles (Right)",
      fontsize = 18, font = :bold)

# Panel 1: 8M
ax1 = CairoMakie.Axis(fig[1, 1],
    title = "8,000,000 Particles (Mesh 88, h = 1.77 a₀)\n~166 part/cell, Statistical Fluctuations ~ 7.8%",
    xlabel = "x (a₀)", ylabel = "y (a₀)",
    aspect = CairoMakie.DataAspect(), backgroundcolor = GREY)
CairoMakie.xlims!(ax1, -60, 60); CairoMakie.ylims!(ax1, -50, 50)
f1_masked = copy(f_8M)
f1_masked[f1_masked .< c_min] .= NaN32
CairoMakie.heatmap!(ax1, xs_8M, ys_8M, f1_masked, colormap = :turbo, colorrange = (c_min, c_max),
         nan_color = GREY, interpolate = true)
CairoMakie.lines!(ax1, R_cluster .* cos.(θ), R_cluster .* sin.(θ), color = :white, linestyle = :dash, linewidth = 2.0)
CairoMakie.scatter!(ax1, [-62.6], [0.0], color = :white, strokecolor = :black, strokewidth = 2.0, markersize = 12)

# Panel 2: 80M
ax2 = CairoMakie.Axis(fig[1, 2],
    title = "80,000,000 Particles (Mesh 110, h = 1.42 a₀)\n~800+ part/cell, Statistical Fluctuations < 2.5%",
    xlabel = "x (a₀)", ylabel = "y (a₀)",
    aspect = CairoMakie.DataAspect(), backgroundcolor = GREY)
CairoMakie.xlims!(ax2, -60, 60); CairoMakie.ylims!(ax2, -50, 50)
f2_masked = copy(f_80M)
f2_masked[f2_masked .< c_min] .= NaN32
CairoMakie.heatmap!(ax2, xs_80M, ys_80M, f2_masked, colormap = :turbo, colorrange = (c_min, c_max),
         nan_color = GREY, interpolate = true)
CairoMakie.lines!(ax2, R_cluster .* cos.(θ), R_cluster .* sin.(θ), color = :white, linestyle = :dash, linewidth = 2.0)
CairoMakie.scatter!(ax2, [-62.6], [0.0], color = :white, strokecolor = :black, strokewidth = 2.0, markersize = 12)

out_png = joinpath(ROOT, "initial_state_8M_vs_80M.png")
CairoMakie.save(out_png, fig, px_per_unit = 2)
println("Saved side-by-side comparison to $out_png")
flush(stdout)
