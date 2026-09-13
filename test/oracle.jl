"""
Comparison against the original Fortran code.

These tests only run if the oracle has been produced:

    cd ref/fortran && make oracle

They are skipped otherwise — the binary dumps are not versioned, and the rest
of the suite already validates the same properties without an external
reference (exact second derivative on a cubic, fourth-order convergence). The
oracle brings something else: the certainty that we are solving *the same
problem* as the thesis, and not merely a correct one.
"""

const ORACLE_DIR = joinpath(@__DIR__, "..", "ref", "fortran")

"""Read a dumped vector: an `Int32` length, then `Float64`s."""
function read_dump_vector(name)
    open(joinpath(ORACLE_DIR, name)) do io
        n = Int(read(io, Int32))
        [read(io, Float64) for _ in 1:n]
    end
end

"""Read a dumped square matrix, column-major as in both Fortran and Julia."""
function read_dump_matrix(name)
    open(joinpath(ORACLE_DIR, name)) do io
        n = Int(read(io, Int32))
        A = Matrix{Float64}(undef, n, n)
        for j in 1:n, i in 1:n
            A[i, j] = read(io, Float64)
        end
        A
    end
end

oracle_available() = isfile(joinpath(ORACLE_DIR, "dump_gx.bin"))

"""Directory of the second oracle: the 1998-01-05 version, the port's target."""
const ORACLE98_DIR = joinpath(@__DIR__, "..", "ref", "fortran98")

oracle98_available() = isfile(joinpath(ORACLE98_DIR, "pot.dat"))

# Tolerated relative discrepancy. The port replaces building blocks (NAG f02agf
# → `eigen`, explicit inversion → factorisation), so bit-for-bit equality is not
# expected; beyond this threshold, on the other hand, it is a regression.
const ORACLE_TOL = 1e-13

reldiff(a, b) = norm(a - b) / norm(b)

@testset "Oracle Fortran" begin
    if oracle_available()
        @testset "ran2 generator" begin
            # The only bit-for-bit test in the suite: the slightest difference
            # in the integer arithmetic — the deliberate overflow included —
            # would shift the whole sequence, and the initial sampling with it.
            ref = read_dump_vector("rand2.bin")
            rng = Ran2(-1)
            @test all(i -> Float64(next!(rng)) === ref[i], eachindex(ref))
        end
    end

    if !oracle_available()
        @info "oracle missing — `cd ref/fortran && make oracle` to enable it"
        @test_skip false
    else
        # `vlas.inp`: 28 intervals on [-xclu, xclu] with xclu = 50.
        axf = uniform_axis(-50.0, 50.0, 28)

        @testset "Fine grid" begin
            @test reldiff(axf.knots, read_dump_vector("dump_gx.bin")) < ORACLE_TOL
            @test reldiff(axf.colloc, read_dump_vector("dump_gtx.bin")) < ORACLE_TOL
        end

        @testset "Collocation and operator" begin
            cm = CollocationMatrices(axf)
            @test reldiff(Matrix(cm.S), read_dump_matrix("dump_sx.bin")) < ORACLE_TOL
            @test reldiff(Matrix(cm.S″), read_dump_matrix("dump_s2x.bin")) < ORACLE_TOL

            D = laplacian1d(cm)
            @test reldiff(D, read_dump_matrix("dump_dex.bin")) < ORACLE_TOL

            λ = read_dump_vector("dump_lxr.bin")
            op = DiagonalizedOperator(D)
            @test reldiff(sort(op.λ), sort(λ)) < ORACLE_TOL
            # A property the Fortran computed and then threw away (`lxi`, `mxi`).
            @test all(<(0), op.λ)
        end

        @testset "Multipole moments" begin
            for (k, file) in ((0, "dump_psx.bin"), (1, "dump_pxx.bin"), (2, "dump_px2.bin"))
                @test reldiff(moments(axf, Val(k)), read_dump_vector(file)) < ORACLE_TOL
            end
        end

        @testset "Stretched coarse grid" begin
            # ⚠️ We start from the DUMPED knots, not from a rebuilt axis:
            # `findacc` stopped its bisection at 1e-10, whereas `stretched_axis`
            # solves to machine precision. Rebuilding the axis would pollute the
            # whole comparison at ~1e-12 and mask the real regressions.
            axb = SplineAxis(read_dump_vector("dumpb_gx.bin"),
                             read_dump_vector("dumpbgtx.bin"))
            cmb = CollocationMatrices(axb)
            @test reldiff(Matrix(cmb.S), read_dump_matrix("dumpb_sx.bin")) < ORACLE_TOL
            @test reldiff(Matrix(cmb.S″), read_dump_matrix("dumpbs2x.bin")) < ORACLE_TOL

            Db = laplacian1d(cmb)
            @test reldiff(Db, read_dump_matrix("dumpbdex.bin")) < ORACLE_TOL
            @test reldiff(sort(eigvals(Db) .|> real),
                          sort(read_dump_vector("dumpblxr.bin"))) < ORACLE_TOL

            for (k, file) in ((0, "dumpbpsx.bin"), (1, "dumpbpxx.bin"), (2, "dumpbpx2.bin"))
                @test reldiff(moments(axb, Val(k)), read_dump_vector(file)) < ORACLE_TOL
            end

            # The rebuilt stretched grid, for its part, need only agree to
            # `findacc`'s tolerance — check we stay in that range.
            axs = stretched_axis(50.0, 150.0, 7, 8)
            @test reldiff(axs.knots, read_dump_vector("dumpb_gx.bin")) < 1e-10
        end

        @testset "Charge deposit" begin
            # `makerho` dumps the particles AND the density of the same
            # invocation: the two therefore stay consistent, whatever moment of
            # the simulation the dump was taken at.
            qp = read_dump_vector("dumpqp.bin")
            npart = length(qp) ÷ 3
            positions = [(qp[3i-2], qp[3i-1], qp[3i]) for i in 1:npart]

            # The axis is rebuilt from the collocation dumped by `makerho`
            # itself, not borrowed from another routine: that is what makes the
            # comparison independent of the order of the calls.
            axb = axis_from_collocation(read_dump_vector("dumprho_gt.bin"))
            mesh = SplineMesh(axb, axb, axb)
            n = nbasis(axb)
            ρref = reshape(read_dump_vector("dumprho.bin"), n, n, n)

            ρ = similar(ρref)
            nout = deposit!(ρ, mesh, positions; charge = 196.0 / npart)
            @test nout == 0
            @test reldiff(ρ, ρref) < ORACLE_TOL

            # The original code's control observable: "sum of the charges".
            @test total_charge(ρ, mesh) ≈ 196.0 rtol = 1e-12
        end

        @testset "Poisson: density → potential" begin
            # `makerh2` dumps its density, its grid and its right-hand side; the
            # `solve` that follows dumps the potential, paired through a flag in
            # COMMON (`solve` is also called after `makerhsf`, on the other
            # grid, and its csol would not match this density).
            axb = SplineAxis(read_dump_vector("dumpb_gx.bin"),
                             read_dump_vector("dumpbgtx.bin"))
            @test read_dump_vector("dumprh2gt.bin") ≈ axb.colloc

            mesh = SplineMesh(axb, axb, axb)
            n = nbasis(axb)
            ρ = reshape(read_dump_vector("dumprh2.bin"), n, n, n)

            # Multipole moments. The quadrupole tolerates a little more: its
            # traceless form (2∫x² − ∫y² − ∫z²) cancels large numbers.
            mp = multipole(ρ, mesh)
            @test mp.charge ≈ read_dump_vector("dumpq.bin")[1] rtol = 1e-12
            @test reldiff(collect(mp.center), read_dump_vector("dumpbari.bin")) < 1e-11
            Q = reshape(read_dump_vector("dumpquad.bin"), 3, 3)
            @test reldiff(collect(mp.quadrupole[1:3]), [Q[1, 1], Q[2, 2], Q[3, 3]]) < 1e-11
            @test reldiff(collect(mp.quadrupole[4:6]), [Q[1, 2], Q[1, 3], Q[2, 3]]) < 1e-11

            # Boundary potential: only the faces are written by `makerh2`, the
            # interior of the `phi` array serving `solve` as a buffer.
            φbord = boundary_potential!(similar(ρ), mesh, mp)
            φref = reshape(read_dump_vector("dumpphi.bin"), n, n, n)
            faces = falses(n, n, n)
            faces[1, :, :] .= true; faces[end, :, :] .= true
            faces[:, 1, :] .= true; faces[:, end, :] .= true
            faces[:, :, 1] .= true; faces[:, :, end] .= true
            @test reldiff(φbord[faces], φref[faces]) < 1e-12

            @test reldiff(poisson_rhs(ρ, mesh),
                          reshape(read_dump_vector("dumprhs.bin"), size(mesh)...)) < 1e-11

            # The complete chain, and its spline coefficients (`csol`).
            φ = poisson(ρ, mesh)
            @test reldiff(φ, reshape(read_dump_vector("dumpphi2.bin"), n, n, n)) < 1e-11
            @test reldiff(spline_coefficients(φ, mesh),
                          reshape(read_dump_vector("dumpcsol.bin"), n, n, n)) < 1e-11
        end

        @testset "Verlet step" begin
            triplets(v) = [(v[3i-2], v[3i-1], v[3i]) for i in 1:length(v)÷3]
            flat(t) = collect(Iterators.flatten(t))

            q0 = triplets(read_dump_vector("mv_qp0.bin"))
            qold0 = triplets(read_dump_vector("mv_qo0.bin"))
            forces = triplets(read_dump_vector("mv_fp.bin"))
            dt = read_dump_vector("mv_par.bin")[1]

            # `move` also dumps `coef` = 1/2M: enough to recover the
            # pseudo-particle's mass without assuming it.
            M = 1 / (2 * read_dump_vector("mv_nb.bin")[1])
            cloud = ParticleCloud(q0, M / ELECTRON_MASS)
            @test mass(cloud) ≈ M
            copyto!(cloud.previous, qold0)
            copyto!(cloud.forces, forces)

            diag = step!(cloud, dt)
            @test reldiff(flat(cloud.positions), read_dump_vector("mv_qp1.bin")) < 1e-14
            @test flat(cloud.previous) == read_dump_vector("mv_qo1.bin")
            @test diag.kinetic ≈ read_dump_vector("mv_ekin.bin")[1] rtol = 1e-13
            @test reldiff(collect(diag.angular), read_dump_vector("mv_lcin.bin")) < 1e-12
        end

        @testset "Fields and forces" begin
            triplets(v) = [(v[3i-2], v[3i-1], v[3i]) for i in 1:length(v)÷3]
            flat(t) = collect(Iterators.flatten(t))

            gx = read_dump_vector("fg_gx.bin")
            gxb = read_dump_vector("fg_gxb.bin")
            fine = SplineAxis(gx, collocation_points(gx))
            coarse = SplineAxis(gxb, collocation_points(gxb))

            # Gaussian convolution tables.
            sm = GaussianSmoothing(fine)
            @test sm.spacing ≈ read_dump_vector("fg_pas.bin")[1]
            @test reldiff(sm.overlap, reshape(read_dump_vector("fg_it1.bin"), 10, :)) < 1e-11
            @test reldiff(sm.gradient, reshape(read_dump_vector("fg_it2.bin"), 10, :)) < 1e-11

            # Forces on the pseudo-particles.
            n = nbasis(fine)
            csol = reshape(read_dump_vector("fg_csol.bin"), n, n, n)
            csolb = reshape(read_dump_vector("fg_csolb.bin"), n, n, n)
            qp = triplets(read_dump_vector("fg_qp.bin"))
            cloud = ParticleCloud(qp, 196.0 / length(qp))
            nout = forces!(cloud, (fine, fine, fine), csol,
                           (coarse, coarse, coarse), csolb, sm;
                           escaped = Int(read_dump_vector("fg_n.bin")[1]))
            # On this dump every particle is inside the fine grid; the coarse
            # and monopole regimes are covered by the standalone suite, not
            # here.
            @test nout == 0
            @test reldiff(flat(cloud.forces), read_dump_vector("fg_fp.bin")) < 1e-11
        end

        @testset "Smoothed deposit" begin
            triplets(v) = [(v[3i-2], v[3i-1], v[3i]) for i in 1:length(v)÷3]

            gx = read_dump_vector("rg_gx.bin")
            ax = SplineAxis(gx, collocation_points(gx))
            mesh = SplineMesh(ax, ax, ax)
            sm = GaussianSmoothing(ax)
            n = nbasis(ax)

            @test reldiff(sm.nodes, reshape(read_dump_vector("rg_gtab.bin"), 8, :)) < 1e-11

            qp = triplets(read_dump_vector("rg_qp.bin"))
            ρ = Array{Float64,3}(undef, n, n, n)
            nout = deposit_smoothed!(ρ, mesh, sm, qp; charge = 196.0 / length(qp))
            @test nout == Int(read_dump_vector("rg_nout.bin")[1])
            @test reldiff(ρ, reshape(read_dump_vector("rg_rho.bin"), n, n, n)) < 1e-12
        end

        @testset "Initial sampling of the pseudo-particles" begin
            # `ran2` being reproduced bit for bit, the comparison is made
            # particle by particle and not on statistics — the only way to tell
            # a bug from sampling noise.
            flat(t) = collect(Iterators.flatten(t))
            prof = read_radial_profile(joinpath(ORACLE_DIR, "data"))
            @test length(prof.quantiles) == length(prof.density) == 1001
            @test prof.rmax == 35.0

            npart, nbelec = 20_000, 196.0
            pos, mom = sample_thomas_fermi(prof, npart, nbelec / npart)

            # ⚠️ Tolerance at 1e-11 and not 1e-13: the Fortran declares
            # `pi = 3.141592653589d0`, three decimals short, and the sampling
            # angles inherit it. With that π the agreement falls to 4e-17 — the
            # discrepancy comes from there and from nothing else.
            @test reldiff(flat(pos), read_dump_vector("in_rt.bin")) < 1e-11
            @test reldiff(flat(mom), read_dump_vector("in_pt.bin")) < 1e-11
        end

        @testset "Projectile" begin
            trip(v) = [(v[3i-2], v[3i-1], v[3i]) for i in 1:length(v)÷3]
            flat(t) = collect(Iterators.flatten(t))
            scal(f) = read_dump_vector(f)[1]

            npart = Int(scal("ip_np.bin"))
            cloud = ParticleCloud(trip(read_dump_vector("ip_qp.bin")),
                                  scal("ip_nbelec.bin") / npart)
            copyto!(cloud.forces, trip(read_dump_vector("ip_fp0.bin")))

            dt, cutoff = scal("ip_dt.bin"), scal("ip_cut.bin")
            pos0 = trip(read_dump_vector("ip_pos0.bin"))[1]
            proj = Projectile(; mass = scal("ip_par.bin"), charge = scal("ip_cha.bin"),
                              energy = 73.498, x0 = pos0[1], impact = pos0[2], dt, cutoff)
            # The dump's actual state, not the one `initpro` would have built:
            # the dump may come from any step of the trajectory.
            proj.position = pos0
            proj.previous = trip(read_dump_vector("ip_old0.bin"))[1]

            force, eel, ejel = projectile_forces!(cloud, proj, Jellium(scal("ip_nbion.bin")))
            @test eel ≈ scal("ip_eel.bin") rtol = 1e-13
            @test ejel ≈ scal("ip_ejel.bin") rtol = 1e-13
            # The back-reaction on the pseudo-particles, easily forgotten.
            @test reldiff(flat(cloud.forces), read_dump_vector("ip_fp1.bin")) < 1e-13

            step!(proj, force, dt)
            @test reldiff(collect(proj.position), read_dump_vector("ip_pos1.bin")) < 1e-13
            @test reldiff(collect(proj.velocity), read_dump_vector("ip_v.bin")) < 1e-13
            @test kinetic_energy(proj) ≈ scal("ip_ekin.bin") rtol = 1e-13
        end

        @testset "Energy budget" begin
            # `enertot2g` is called only one step in ten: everything it needs is
            # dumped ON ENTRY, otherwise the snapshots would come from different
            # iterations and would not be comparable.
            triplets(v) = [(v[3i-2], v[3i-1], v[3i]) for i in 1:length(v)÷3]
            gxf, gxb = read_dump_vector("et_gx.bin"), read_dump_vector("et_gxb.bin")
            axf = SplineAxis(gxf, collocation_points(gxf))
            axb = SplineAxis(gxb, collocation_points(gxb))
            sm = GaussianSmoothing(axf)
            n = nbasis(axf)

            nbelec = read_dump_vector("et_nbelec.bin")[1]
            npart = Int(read_dump_vector("et_np.bin")[1])
            cloud = ParticleCloud(triplets(read_dump_vector("et_qp.bin")), nbelec / npart)
            jel = Jellium(read_dump_vector("et_nbion.bin")[1])

            potel = interaction_energy(cloud,
                                       (axf, axf, axf), reshape(read_dump_vector("et_csol.bin"), n, n, n),
                                       (axb, axb, axb), reshape(read_dump_vector("et_csolb.bin"), n, n, n),
                                       sm)
            budget = energy_budget(jel, read_dump_vector("et_ekin.bin")[1],
                                   read_dump_vector("et_in.bin")[1], potel)

            @test budget.meanfield ≈ read_dump_vector("et_ejel.bin")[1] rtol = 1e-12
            @test budget.total ≈ read_dump_vector("et_out.bin")[1] rtol = 1e-11
            @test budget.ions ≈ 3 * 196.0^2 / (5 * jel.radius)
        end

        @testset "Mean field: exchange-correlation and jellium" begin
            ax = axis_from_collocation(read_dump_vector("ps_gt.bin"))
            mesh = SplineMesh(ax, ax, ax)
            n = nbasis(ax)
            ρ = reshape(read_dump_vector("ps_rho.bin"), n, n, n)
            jel = Jellium(read_dump_vector("ps_nbion.bin")[1])

            # The effective potential at the collocation points.
            g = ax.colloc
            ech = [xc_potential(ρ[i, j, k]) +
                   Vlasov.potential(jel, sqrt(g[i]^2 + g[j]^2 + g[k]^2))
                   for i in 1:n, j in 1:n, k in 1:n]
            @test reldiff(ech, reshape(read_dump_vector("ps_ech.bin"), n, n, n)) < 1e-13

            # Then its addition to the Hartree potential, in coefficients.
            csol = reshape(read_dump_vector("ps_csol0.bin"), n, n, n)
            before = copy(csol)
            effective_potential!(csol, ρ, mesh, jel)
            @test reldiff(csol, reshape(read_dump_vector("ps_csol.bin"), n, n, n)) < 1e-11
            # This term is not a correction: it weighs as much as Hartree does.
            @test norm(csol - before) / norm(before) > 0.5
        end

        @testset "Matching between grids" begin
            gxf = read_dump_vector("fg_gx.bin")
            gxc = read_dump_vector("sf_gxc.bin")
            axf = axis_from_collocation(read_dump_vector("sf_gt.bin"))
            axc = SplineAxis(gxc, collocation_points(gxc))
            fine = SplineMesh(axf, axf, axf)
            coarse = SplineMesh(axc, axc, axc)
            nested = NestedMeshes(fine, coarse)
            n = nbasis(axf)

            ρf = reshape(read_dump_vector("sf_rho.bin"), n, n, n)
            csolc = reshape(read_dump_vector("sf_csolc.bin"), n, n, n)

            # The boundary values read off the coarse solution.
            φ = Array{Float64,3}(undef, n, n, n)
            boundary_from_coarse!(φ, fine, coarse, csolc)
            φref = reshape(read_dump_vector("sf_phi.bin"), n, n, n)
            faces = falses(n, n, n)
            faces[1, :, :] .= true; faces[end, :, :] .= true
            faces[:, 1, :] .= true; faces[:, end, :] .= true
            faces[:, :, 1] .= true; faces[:, :, end] .= true
            @test reldiff(φ[faces], φref[faces]) < 1e-13

            @test reldiff(poisson_rhs!(Array{Float64,3}(undef, size(fine)), ρf, fine, φ),
                          reshape(read_dump_vector("sf_rhs.bin"), size(fine)...)) < 1e-12

            # The complete two-level chain, from both densities.
            ρc = reshape(read_dump_vector("dumprh2.bin"), n, n, n)
            φs = (Array{Float64,3}(undef, n, n, n), Array{Float64,3}(undef, n, n, n))
            poisson!(φs, (ρf, ρc), nested)
            @test reldiff(φs[1], reshape(read_dump_vector("sf_phi2.bin"), n, n, n)) < 1e-11
            @test reldiff(φs[2], reshape(read_dump_vector("dumpphi2.bin"), n, n, n)) < 1e-11
        end
    end
end

# ---------------------------------------------------------------------------
# "Late version" oracle (1998-01-05)
# ---------------------------------------------------------------------------
#
#   cd ref/fortran98 && make && python3 make_pot.py && ./vlas98 > run98.log
#
# The reference values below are copied from `run98.log`: the `initialise4`
# routine prints pseudo-particle 109 and the acceptance rate of its own accord,
# which spares us instrumenting the source.

@testset "1998 oracle — rejection sampling" begin
    if !oracle98_available()
        @info "1998 oracle missing (ref/fortran98/pot.dat) — tests skipped"
    else
        prof = read_potential_profile(joinpath(ORACLE98_DIR, "pot.dat"))
        @test prof.rmax == 35.0
        @test prof.fermi == 0.0
        @test prof.potential[1] ≈ -0.11552990 atol = 1e-9
        @test prof.potential[end] == 0.0

        npart, nbelec = 20_000, 196.0
        pos, mom = sample_thomas_fermi(prof, npart, nbelec / npart)

        # Pseudo-particle 109, as the oracle prints it.
        r = sqrt(sum(abs2, pos[109]))
        p = sqrt(sum(abs2, mom[109]))
        # `r` and `p` are **identical to the bit**: they pass through no
        # transcendental function, so their equality proves that the rejection
        # loop consumed the random stream exactly as the oracle did —
        # rejection for rejection.
        @test r == 18.535201405644546
        @test p == 3.7169869799140247e-3

        # The components go through cos/sin: a library discrepancy, at the
        # level already observed on the rest of the port.
        @test reldiff(collect(pos[109]),
                      [11.908010207778275, 13.479312950281439, 4.4789626508412743]) < 1e-12
        @test reldiff(collect(mom[109]),
                      [1.1166910985229179e-3, 9.4449849564513094e-4,
                       -3.4171502441441093e-3]) < 1e-12
        @test isapprox(pos[1][1], 20.103859205517573; rtol = 1e-12)
        @test isapprox(mom[1][1], -3.3382689538331450e-3; rtol = 1e-12)
    end
end
