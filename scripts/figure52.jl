#!/usr/bin/env julia
"""
Reproduce figure 5.2 of the thesis: cross-sections of the electron density
during central Na₁₀₀₀ + H⁺ collisions, at four projectile energies.

    julia --project=gpu -t auto scripts/figure52.jl [--particules=3200000]
                                                    [--x=8] [--sortie=figure52.png]

Where [`figure53.jl`](figure53.jl) tests the *integral* of the response — the
stopping power — this one tests its **spatial structure**: the plasmon wake the
ion leaves behind, of wavelength `2πv/ω_p`.

That is a genuinely different check. A wrong mean field could still integrate to
a plausible `dE/dx`; it could not put the wake's nodes in the right places.

⚠️ **The energy is the point.** At 1 and 4 keV there is next to nothing to see —
the ion drags a compact clump and that is all. The wake appears at 9 keV and is
unmistakable at 16, because it is a *velocity* effect. Looking for it at 4 keV
is looking at the dullest of the four panels.

Display, matched to the thesis panels:

  * rainbow palette, vacuum **masked** to grey rather than coloured — from
    `0 → max` the cluster body sits at 89 % of full scale and comes out one flat
    colour, the whole range spent on empty space;
  * top of the scale at `1.45·ρ_bulk`, which lands the body in the orange and
    gives the wake's oscillation the green→red span;
  * `image!` with `interpolate`, not `heatmap!`: the density is a C¹ spline and
    the collocation values are point samples of it. ⚠️ This does not reduce the
    sampling noise, it only keeps the grid from showing through.

⚠️ **Statistics.** A fine-grid cell holds `npart/6000` pseudo-particles, so the
shot noise on the density goes as `1/√that`: 9.6 % at 800 000, 4.4 % at
3 200 000. The wake is a few per cent, so the default is 3.2 M — measured, not
guessed.
"""

using Vlasov
using Printf
using Serialization

# Optional accelerators, as in `film_images.jl`: present in `gpu/` only.
const ACCELERATE = try; @eval using AppleAccelerate; true; catch; false; end
const METAL = try; @eval using Metal; true; catch; false; end

using CairoMakie
CairoMakie.activate!(type = "png")

const ROOT = dirname(@__DIR__)
const KEV = 1000 / HARTREE_TO_EV
const ENERGIES = (1, 4, 9, 16)
const GREY = RGBf(0.78, 0.78, 0.78)

function parse_args(argv)
    o = Dict("particules" => "3200000", "x" => "8", "sortie" => "figure52.png",
             "epaisseur" => "2")
    for a in argv
        m = match(r"^--([a-z]+)=(.+)$", a)
        (m === nothing || !haskey(o, m[1])) && error("unrecognised argument: $a")
        o[m[1]] = m[2]
    end
    o
end

"""Density averaged over the planes with `|z| ≤ halfwidth` — see
`film_images.jl` for why the slab is thin: the wake fits inside one cell in `z`,
so a thicker one dilutes the signal faster than it kills the noise."""
function slab_z0(ρ, mesh, halfwidth)
    gz = mesh.axes[3].colloc
    ks = findall(z -> abs(z) <= halfwidth, gz)
    isempty(ks) && (ks = [argmin(abs.(gz))])
    out = zeros(Float32, size(ρ, 1), size(ρ, 2))
    @inbounds for k in ks, j in axes(ρ, 2), i in axes(ρ, 1)
        out[i, j] += Float32(ρ[i, j, k])
    end
    out ./ length(ks)
end

"""Run one crossing up to the instant the ion reaches `xsnap`, and return the
slab there. All four panels are taken at the **same ion position**, not at the
same time: that is what makes them comparable."""
function snapshot(profile, npart, keV, xsnap, halfwidth)
    p = SimulationParameters(nfine = 44, ninner = 22, nouter = 22, rcluster = 78.0,
                             rbox = 235.0, nions = 1000.0, nelectrons = 1000.0,
                             nparticles = npart, nsteps = 0, dt = 1.0)
    e = keV * KEV
    proj = Projectile(mass = 1836.154, charge = 1.0, energy = e, impact = 0.0,
                      x0 = -65.0, dt = 1.0, softening = GaussianSoftening(1.0))
    sim = Simulation(p, profile; projectile = proj)
    fine = sim.meshes[1]
    acc = METAL ? ForceAccelerator(MtlArray, fine.axes, sim.smoothing, npart,
                                   size(sim.csol[1], 1)) : nothing

    t0 = time(); n = 0
    while sim.projectile.position[1] < xsnap
        step!(sim; energy = false, accelerator = acc)
        n += 1
    end
    @printf("%2d keV (v = %.2f) : %3d steps, ion at x = %.1f, %.0f s\n",
            keV, sqrt(2e / 1836.154), n, sim.projectile.position[1], time() - t0)
    (rho = slab_z0(sim.ρ[1], fine, halfwidth),
     x = sim.projectile.position[1],
     gx = Float32.(fine.axes[1].colloc), gy = Float32.(fine.axes[2].colloc))
end

function render(snaps, out)
    # ρ_bulk from the upper quartile of the body: robust to the surface and to
    # the ion's own clump, which is three times the bulk.
    ref = snaps[last(ENERGIES)].rho
    bulk = let c = sort(filter(>(0), vec(ref))); c[round(Int, 0.75 * length(c))] end
    hi, void = Float32(1.45bulk), Float32(0.04bulk)

    fig = Figure(size = (900, 900), backgroundcolor = GREY)
    for (n, keV) in enumerate(ENERGIES)
        d = snaps[keV]
        r, c = fldmod1(n, 2)
        A = [a < void ? NaN32 : a for a in d.rho]
        ax = Axis(fig[r, c]; aspect = DataAspect(), backgroundcolor = GREY,
                  xticksvisible = false, yticksvisible = false,
                  xticklabelsvisible = false, yticklabelsvisible = false,
                  topspinevisible = false, rightspinevisible = false,
                  leftspinevisible = false, bottomspinevisible = false)
        image!(ax, extrema(d.gx), extrema(d.gy), A; colorrange = (0.0f0, hi),
               colormap = :jet, nan_color = GREY, interpolate = true)
        text!(ax, 44, -44; text = "$keV keV", align = (:right, :bottom),
              color = :black, fontsize = 19, font = :bold)
        limits!(ax, -50, 50, -50, 50)
    end
    colgap!(fig.layout, 4); rowgap!(fig.layout, 4)
    save(out, fig)
end

function main(argv)
    o = parse_args(argv)
    npart = parse(Int, o["particules"])
    xsnap = parse(Float64, o["x"])
    halfwidth = parse(Float64, o["epaisseur"])

    grid, ρr = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
    profile = PotentialProfile(grid, ρr)

    @printf("Na1000, %d pseudo-particles, snapshot at x = %g\n", npart, xsnap)
    @printf("BLAS: %s    forces: %s\n",
            ACCELERATE ? "Accelerate" : "OpenBLAS", METAL ? "GPU (Float32)" : "CPU")

    snaps = Dict(keV => snapshot(profile, npart, keV, xsnap, halfwidth)
                 for keV in ENERGIES)

    out = joinpath(ROOT, o["sortie"])
    render(snaps, out)
    serialize(joinpath(ROOT, "figure52.jls"), snaps)
    @printf("→ %s\n", out)
end

main(ARGS)
