using Vlasov
using LinearAlgebra
using Test
import SpecialFunctions
using KernelAbstractions: CPU, synchronize

"""Non-uniform test grid, in the spirit of the original code's."""
function testaxis(n = 8; L = 1.0)
    knots = [L * (t + 0.15 * sinpi(2t)) for t in range(0, 1; length = n + 1)]
    # Two collocation points per knot, as `SplineAxis` requires.
    colloc = similar(knots, 2length(knots))
    colloc[1] = knots[1]
    colloc[end] = knots[end]
    for j in 1:n
        a, b = knots[j], knots[j+1]
        colloc[2j]   = a + (b - a) / 3
        colloc[2j+1] = a + 2(b - a) / 3
    end
    SplineAxis(knots, colloc)
end

@testset "Vlasov.jl" begin

    include("oracle.jl")

    @testset "ran2 generator" begin
        rng = Ran2(-1)
        x = next!(rng)
        @test x isa Float32                  # the Fortran declared REAL
        @test 0 <= x < 1

        # Determinism: same seed, same sequence.
        @test [next!(Ran2(-1)) for _ in 1:50] == [next!(Ran2(-1)) for _ in 1:50]
        @test [next!(Ran2(-7)) for _ in 1:50] != [next!(Ran2(-1)) for _ in 1:50]

        # The thesis's variant and the corrected version diverge: these really
        # are two different generators, not a rounding detail.
        thesis = [next!(Ran2(-1)) for _ in 1:100]
        fixed = [next!(Ran2(-1; consistent = true)) for _ in 1:100]
        @test thesis != fixed

        # Neither of the two may be obviously biased.
        for r in (Ran2(-3), Ran2(-3; consistent = true))
            v = Float64[next!(r) for _ in 1:100_000]
            m = sum(v) / length(v)
            @test abs(m - 0.5) < 0.01
            @test abs(sum(x -> (x - m)^2, v) / length(v) - 1 / 12) < 0.002
            @test all(x -> 0 <= x < 1, v)
        end
    end

    @testset "BasisIndex" begin
        # Round trip linear index ↔ (knot, kind).
        for lin in 1:20
            @test linearindex(BasisIndex(lin)) == lin
        end
        @test BasisIndex(1) == BasisIndex(1, Value)
        @test BasisIndex(2) == BasisIndex(1, Slope)
        @test linearindex(BasisIndex(3, Slope)) == 6
    end

    @testset "Hermite basis" begin
        ax = testaxis()
        g = ax.knots

        for k in 2:(nknots(ax)-1)
            bv = BasisIndex(k, Value)
            bs = BasisIndex(k, Slope)

            # Defining properties of the Hermite basis at its carrying knot.
            @test value(ax, bv, g[k]) ≈ 1 atol = 1e-14
            @test value(ax, bs, g[k]) ≈ 0 atol = 1e-14
            @test derivative(ax, bv, g[k]) ≈ 0 atol = 1e-12
            @test derivative(ax, bs, g[k]) ≈ 1 atol = 1e-12

            # Vanishing at the neighbouring knots.
            @test value(ax, bv, g[k-1]) ≈ 0 atol = 1e-14
            @test value(ax, bv, g[k+1]) ≈ 0 atol = 1e-14
        end

        # Local support: two intervals only.
        b = BasisIndex(4, Value)
        lo, hi = support(ax, b)
        @test (lo, hi) == (g[3], g[5])
        @test value(ax, b, g[2]) == 0
        @test value(ax, b, g[6]) == 0
    end

    @testset "Reproduction of cubics" begin
        # The Hermite interpolant reproduces exactly any polynomial of degree ≤ 3.
        ax = testaxis(10)
        p(x)  = 2.0 - 0.7x + 1.3x^2 - 0.4x^3
        p′(x) = -0.7 + 2.6x - 1.2x^2

        function interpolate(x)
            s = 0.0
            for k in 1:nknots(ax)
                s += p(ax.knots[k])  * value(ax, BasisIndex(k, Value), x)
                s += p′(ax.knots[k]) * value(ax, BasisIndex(k, Slope), x)
            end
            s
        end

        for x in range(ax.knots[1], ax.knots[end]; length = 37)
            @test interpolate(x) ≈ p(x) atol = 1e-12
        end
    end

    @testset "Axis construction" begin
        ax = uniform_axis(-50.0, 50.0, 28)
        @test nknots(ax) == 29
        @test nbasis(ax) == 58
        @test ax.knots[1] == -50.0
        @test ax.knots[end] ≈ 50.0
        @test maximum(diff(ax.knots)) - minimum(diff(ax.knots)) < 1e-12

        # The collocation points are the 2-point Gauss nodes of each interval,
        # bounded by the ends of the domain.
        @test ax.colloc[1] == ax.knots[1]
        @test ax.colloc[end] == ax.knots[end]
        h = ax.knots[2] - ax.knots[1]
        @test ax.colloc[2] ≈ ax.knots[1] + h * (1 - 1 / sqrt(3)) / 2
        @test ax.colloc[3] ≈ ax.knots[1] + h * (1 + 1 / sqrt(3)) / 2

        # Geometric ratio: it must satisfy its defining equation.
        h1, L, n = 50.0 / 6, 100.0, 8
        a = stretch_ratio(h1, L, n)
        @test a > 1
        @test L * (1 - a) / (1 - a^n) ≈ h1 atol = 1e-13

        # The knots follow from the collocation points: two Gauss points
        # determine their interval unambiguously. That is what makes a dump
        # carrying its collocation usable on its own.
        for a in (uniform_axis(-50.0, 50.0, 28), uniform_axis(0.0, 1.0, 5))
            @test knots_from_collocation(a.colloc) ≈ a.knots
            b = axis_from_collocation(a.colloc)
            @test b.knots ≈ a.knots && b.colloc == a.colloc
        end

        axs = stretched_axis(50.0, 150.0, 7, 8)
        @test knots_from_collocation(axs.colloc) ≈ axs.knots   # stretched too
        @test nknots(axs) == 29
        @test axs.knots[1] ≈ -150.0
        @test axs.knots[end] ≈ 150.0
        @test axs.knots[15] ≈ 0 atol = 1e-12          # central knot
        @test axs.knots ≈ -reverse(axs.knots)          # symmetry
        # A seamless join: the first stretched step equals the constant step.
        d = diff(axs.knots)
        @test d[8] ≈ h1 rtol = 1e-9
        @test issorted(axs.knots)
    end

    @testset "Moments" begin
        # An exact identity: the Hermite interpolant of a polynomial of degree
        # ≤ 3 being that polynomial itself, ∫xᵏp = Σ_b c_b·moment(b,k) where the
        # c_b are the values and slopes of p at the knots. Holds for k = 0, 1, 2
        # and depends on no external reference.
        coeffs = (2.0, -0.7, 1.3, -0.4)                # p(x) = Σ coeffs[j+1]·xʲ
        p(x)  = sum(c * x^(j - 1) for (j, c) in enumerate(coeffs))
        p′(x) = sum((j - 1) * c * x^(j - 2) for (j, c) in enumerate(coeffs) if j > 1)
        exact(k, x0, xn) = sum(c * (xn^(j - 1 + k + 1) - x0^(j - 1 + k + 1)) / (j - 1 + k + 1)
                               for (j, c) in enumerate(coeffs))

        for ax in (uniform_axis(-2.0, 3.0, 9), stretched_axis(1.0, 4.0, 5, 6), testaxis(11))
            x0, xn = ax.knots[1], ax.knots[end]
            for k in 0:2
                m = moments(ax, Val(k))
                got = sum(1:nknots(ax)) do kn
                    g = ax.knots[kn]
                    p(g) * m[linearindex(BasisIndex(kn, Value))] +
                    p′(g) * m[linearindex(BasisIndex(kn, Slope))]
                end
                @test got ≈ exact(k, x0, xn) rtol = 1e-11
            end
        end

        # A moment depends only on the support of its basis function.
        ax = uniform_axis(0.0, 1.0, 6)
        b = BasisIndex(3, Value)
        lo, hi = support(ax, b)
        @test moment(ax, b, Val(0)) > 0
        @test moment(ax, b, Val(0)) < hi - lo
    end

    @testset "Collocation matrices" begin
        ax = testaxis(12)
        cm = CollocationMatrices(ax)
        m = nbasis(ax)
        @test size(cm.S) == (m, m)

        # Banded structure: nothing beyond the 2nd super-/sub-diagonal.
        for i in 1:m, j in 1:m
            abs(i - j) > 2 && @test cm.S[i, j] == 0
        end

        # Collocation matrices are not symmetric: rows = points,
        # columns = basis functions.
        @test !issymmetric(cm.S)

        # Consistency: S really does evaluate the basis at the collocation points.
        for k in 2:(m-1)
            for lin in 1:m
                @test cm.S[k, lin] ≈ value(ax, BasisIndex(lin), ax.colloc[k]) atol = 1e-14
            end
        end
    end

    @testset "1D operator" begin
        ax = testaxis(12)
        cm = CollocationMatrices(ax)
        D = laplacian1d(cm)
        @test size(D, 1) == nbasis(ax) - 2

        # Non-regression: `laplacian1d` must agree with the reference dense
        # path. On BandedMatrices v1.12.0, right division between two
        # `BandedMatrix` returns a WRONG result silently (residual ≈ 0.3) — this
        # test fails if anyone "simplifies" `Matrix(S″) / lu(S)` into `S″ / S`.
        Sdense, S2dense = Matrix(cm.S), Matrix(cm.S″)
        Dref = (S2dense / Sdense)[2:end-1, 2:end-1]
        @test norm(D - Dref) / norm(Dref) < 1e-12

        # The division must solve its defining equation.
        X = Matrix(cm.S″) / lu(cm.S)
        @test norm(X * Sdense - S2dense) / norm(S2dense) < 1e-12

        # An observed property, on which the whole tensor method rests:
        # a real and strictly negative spectrum.
        λ = eigvals(D)
        @test maximum(abs, imag.(λ)) < 1e-10 * maximum(abs, real.(λ))
        @test all(<(0), real.(λ))
    end

    @testset "D really is the second derivative" begin
        # The Hermite interpolant of a cubic being exact, D·f must return f″
        # EXACTLY at the interior collocation points. The two outermost rows
        # encode the boundary conditions, not a derivative: that is what
        # `laplacian1d` strips off.
        ax = uniform_axis(-2.0, 3.0, 10)
        cm = CollocationMatrices(ax)
        Dfull = Matrix(cm.S″) / lu(cm.S)
        f(x) = 2.0 - 0.7x + 1.3x^2 - 0.4x^3
        f″(x) = 2.6 - 2.4x
        got = Dfull * f.(ax.colloc)
        want = f″.(ax.colloc)
        @test maximum(abs, got[2:end-1] - want[2:end-1]) < 1e-11
    end

    @testset "Poisson 3D" begin
        L = 2.0
        φex(x, y, z) = sinpi(x / L) * sinpi(y / L) * sinpi(z / L)

        function err(n)
            a = uniform_axis(0.0, L, n)
            m = SplineMesh(a, a, a)
            cx, cy, cz = collocation_axes(m)
            Φ = [φex(x, y, z) for x in cx, y in cy, z in cz]
            norm(solve((-3 * (pi / L)^2) .* Φ, m) - Φ) / norm(Φ)
        end

        a = uniform_axis(0.0, L, 8)
        mesh = SplineMesh(a, a, a)
        @test ndims(mesh) == 3
        @test size(mesh) == (16, 16, 16)
        @test all(length.(collocation_axes(mesh)) .== 16)

        # The direct operator must be the inverse of the solver.
        ρ = randn(size(mesh))
        Φ = solve(ρ, mesh)
        @test norm(laplacian!(similar(Φ), Φ, mesh) - ρ) / norm(ρ) < 1e-10

        # Manufactured solution: fourth-order convergence, characteristic of
        # cubic collocation at the Gauss points. A lower order would signal an
        # error of discretisation, not of precision.
        errs = map(err, (4, 8, 16))
        @test issorted(errs; rev = true)
        for i in 1:(length(errs)-1)
            @test log2(errs[i] / errs[i+1]) > 3.8
        end
        @test errs[end] < 1e-6
    end

    @testset "Charge deposit" begin
        using Random
        rng = Random.MersenneTwister(1234)

        ax = uniform_axis(-50.0, 50.0, 28)
        mesh = SplineMesh(ax, ax, ax)
        n = nbasis(ax)

        # Dual lengths: their sum must cover the domain exactly.
        l = dual_lengths(ax)
        @test length(l) == n
        @test all(>(0), l)
        @test sum(l) ≈ ax.knots[end] - ax.knots[1]

        # Locating. Landing exactly on knot k returns `(k-1, 0)`: all the weight
        # goes to the right-hand knot, which IS knot k. The Fortran's convention.
        c, w = locate(ax, ax.colloc[5])
        @test (c, w) == (4, 0.0)
        # At the left edge there is no knot to the left: weight 1 on knot 1.
        @test locate(ax, ax.colloc[1]) == (1, 1.0)
        @test locate(ax, -60.0) === nothing
        @test locate(ax, 60.0) === nothing
        c, w = locate(ax, (ax.colloc[7] + ax.colloc[8]) / 2)
        @test c == 7 && w ≈ 0.5

        # Charge conservation: THE check on the deposit. Depositing N electrons
        # and reintegrating the density must give back N, up to rounding.
        npart, nbelec = 50_000, 196.0
        positions = [ntuple(_ -> 8.0 * randn(rng), 3) for _ in 1:npart]
        ρ = zeros(n, n, n)
        nout = deposit!(ρ, mesh, positions; charge = nbelec / npart)
        @test nout == 0
        @test total_charge(ρ, mesh) ≈ nbelec rtol = 1e-12

        # Particles outside the domain are counted and ignored, not wrapped.
        outside = [(200.0, 0.0, 0.0), (0.0, -300.0, 0.0)]
        ρ2 = zeros(n, n, n)
        @test deposit!(ρ2, mesh, outside; charge = 1.0) == 2
        @test all(iszero, ρ2)

        # Spline coefficients: S⁻¹ then S must give back the identity.
        coefs = spline_coefficients(ρ, mesh)
        back = similar(ρ)
        src = coefs
        for d in 1:3
            dst = similar(ρ)
            apply_mode!(dst, Matrix(mesh.collocation[d].S), src, d)
            src = dst
        end
        @test norm(src - ρ) / norm(ρ) < 1e-10

        # An inconsistent array size must be refused, not tolerated.
        @test_throws DimensionMismatch deposit!(zeros(n, n, n - 1), mesh,
                                                positions; charge = 1.0)
    end

    @testset "Multipole expansion" begin
        ax = uniform_axis(-20.0, 20.0, 16)
        mesh = SplineMesh(ax, ax, ax)
        n = nbasis(ax)

        # A centred spherically symmetric distribution has neither dipole nor
        # quadrupole: that is what makes the check discriminating.
        cx, cy, cz = map(a -> a.colloc, mesh.axes)
        σ = 3.0
        ρ = [exp(-(x^2 + y^2 + z^2) / 2σ^2) for x in cx, y in cy, z in cz]
        ρ .*= 10.0 / total_charge(ρ, mesh)      # normalised to 10 units

        mp = multipole(ρ, mesh)
        @test mp.charge ≈ 10.0 rtol = 1e-10
        @test all(c -> abs(c) < 1e-8, mp.center)
        @test all(q -> abs(q) < 1e-6, mp.quadrupole)

        # Far from the source, the potential tends to a point charge's.
        for r in (1e3, 1e4)
            @test potential(mp, r, 0.0, 0.0) ≈ 10.0 / r rtol = 1e-6
        end

        # Offsetting the distribution must move the barycentre by as much.
        shifted = [exp(-((x - 4)^2 + y^2 + z^2) / 2σ^2) for x in cx, y in cy, z in cz]
        mps = multipole(shifted, mesh)
        @test mps.center[1] ≈ 4.0 rtol = 1e-6
        @test abs(mps.center[2]) < 1e-8
    end

    @testset "Poisson with multipole boundaries" begin
        ax = uniform_axis(-20.0, 20.0, 16)
        mesh = SplineMesh(ax, ax, ax)
        cx, cy, cz = map(a -> a.colloc, mesh.axes)
        ρ = [exp(-((x - 1)^2 + (y + 2)^2 + z^2) / 8) for x in cx, y in cy, z in cz]

        φ = poisson(ρ, mesh)
        @test size(φ) == size(ρ)

        # On the faces, the potential IS the multipole expansion.
        mp = multipole(ρ, mesh)
        @test φ[1, 5, 7] ≈ potential(mp, cx[1], cy[5], cz[7])
        @test φ[end, 3, 9] ≈ potential(mp, cx[end], cy[3], cz[9])

        # Inside, the operator applied to the solution must give back the
        # right-hand side — the boundary lifting included.
        rhs = poisson_rhs(ρ, mesh)
        inner = φ[2:end-1, 2:end-1, 2:end-1]
        @test norm(laplacian!(similar(inner), inner, mesh) - rhs) / norm(rhs) < 1e-10

        # The right-hand side is −4πρ plus the boundary lifting. The latter
        # decays towards the interior but does not vanish exactly: the full
        # operator is not strictly local in support (bandwidth 19 out of 56
        # measured, with faint entries beyond).
        lifting = rhs .+ 4π .* @view ρ[2:end-1, 2:end-1, 2:end-1]
        c = size(mesh, 1) ÷ 2
        @test abs(lifting[c, c, c]) < abs(lifting[1, c, c])
        @test rhs[c, c, c] ≈ -4π * ρ[c+1, c+1, c+1] rtol = 1e-6

        @test_throws DimensionMismatch poisson_rhs!(zeros(3, 3, 3), ρ, mesh)
    end

    @testset "Verlet integration" begin
        w, dt = 0.5, 0.05
        q0, v = (1.0, -2.0, 0.5), (0.3, 0.1, -0.2)

        # Zero force: Verlet is exact, the position is affine in time.
        free = ParticleCloud([q0], w)
        free.previous[1] = q0 .- dt .* v
        local diag
        for _ in 1:20
            diag = step!(free, dt)
        end
        @test all(free.positions[1] .≈ q0 .+ (20dt) .* v)
        M = mass(free)
        @test M == ELECTRON_MASS * w
        @test diag.kinetic ≈ M * (v[1]^2 + v[2]^2 + v[3]^2) / 2

        # `rcmax`: splitting the kinetic energy into inside/escaped. Without a
        # radius, nothing is counted as escaped.
        @test step!(free, dt).escaped == 0
        # Three identical particles at increasing radii, with zero relative
        # motion: only those beyond the radius count.
        three = ParticleCloud([(1.0, 0.0, 0.0), (5.0, 0.0, 0.0), (9.0, 0.0, 0.0)], w)
        for i in 1:3
            three.previous[i] = three.positions[i] .- dt .* v
        end
        d = step!(three, dt; rcmax = 6.0)
        @test d.escaped ≈ d.kinetic / 3        # only one of the three is outside
        # The radius is measured on the position before the step, as `move` does.
        @test step!(three, dt; rcmax = 100.0).escaped == 0

        # Harmonic oscillator: Verlet does not conserve the energy exactly, but
        # without drift — it oscillates within a narrow band.
        k = 3.0
        osc = ParticleCloud([(1.0, 0.0, 0.0)], w)
        osc.previous[1] = osc.positions[1]           # starting at rest
        energies = Float64[]
        for _ in 1:4000
            osc.forces[1] = (-k) .* osc.positions[1]
            d = step!(osc, dt)
            # `d.kinetic` is the kinetic energy at t; after the step,
            # `previous` carries q(t). Mixing the times would make the energy
            # oscillate for the wrong reasons.
            q = osc.previous[1]
            push!(energies, d.kinetic + k * (q[1]^2 + q[2]^2 + q[3]^2) / 2)
        end
        amplitude = (maximum(energies) - minimum(energies)) / abs(first(energies))
        @test amplitude < 1e-2
        # No drift: the two halves of the trajectory have the same mean.
        half = length(energies) ÷ 2
        m1 = sum(energies[1:half]) / half
        m2 = sum(energies[half+1:end]) / half
        @test abs(m2 - m1) / abs(m1) < 1e-6

        # Central force: the angular momentum is conserved.
        orb = ParticleCloud([(1.0, 0.0, 0.0)], w)
        orb.previous[1] = (1.0, -0.4dt, 0.0)
        Ls = NTuple{3,Float64}[]
        for _ in 1:500
            r = orb.positions[1]
            n3 = sqrt(sum(abs2, r))
            orb.forces[1] = (-1.0 / n3^3) .* r
            push!(Ls, step!(orb, dt).angular)
        end
        Lz = [L[3] for L in Ls]
        @test maximum(abs, Lz .- first(Lz)) / abs(first(Lz)) < 1e-3
        @test all(L -> abs(L[1]) < 1e-14 && abs(L[2]) < 1e-14, Ls)
    end

    @testset "PackedPositions" begin
        x0, h, n = -78.0, 1.4181818181818182, 400
        # Deliberately past the fine grid on both sides: the representation has
        # to describe those too, which is why `knode` is never clamped.
        pos = [ntuple(d -> (d - 2.0) * 60 + 0.9i - 200, 3) for i in 1:n]
        data = Vector{Vlasov.PackedParticle{Float32}}(undef, n)
        fill!(data, zero(Vlasov.PackedParticle{Float32}))
        p = Vlasov.PackedPositions(data, x0, h, false)
        for i in 1:n
            p[i] = pos[i]
        end

        @test length(p) == n
        @test eltype(p) === NTuple{3,Float64}
        # The round trip loses only `Float32` on a bounded offset, never the
        # magnitude of the coordinate itself.
        @test maximum(i -> maximum(abs, p[i] .- pos[i]), 1:n) < 1e-7
        @test maximum(i -> maximum(abs, data[i].delta), 1:n) <= h / 2 + 1e-6
        # Unclamped: this sample reaches well outside [1, nknots].
        @test minimum(i -> minimum(data[i].knode), 1:n) < 1
        @test maximum(i -> maximum(data[i].knode), 1:n) > 100

        q = similar(p)
        @test q isa Vlasov.PackedPositions
        @test size(q) == size(p)
        q[7] = (1.5, -2.5, 0.25)
        @test all(q[7] .≈ (1.5, -2.5, 0.25))

        c = copy(p)
        @test all(c[i] == p[i] for i in 1:n)
        c[1] = (0.0, 0.0, 0.0)
        @test p[1] != c[1]          # a copy, not a view

        # A cloud built on it keeps plain triples for the forces.
        cloud = ParticleCloud(p, similar(p), fill((0.0, 0.0, 0.0), n), 0.5)
        @test cloud.positions isa Vlasov.PackedPositions
        @test cloud.forces isa Vector{NTuple{3,Float64}}
        @test length(cloud) == n
    end

    @testset "Verlet on packed positions" begin
        x0, h = -78.0, 1.4181818181818182
        w, dt = 0.5, 0.05
        q0, v = (1.0, -2.0, 0.5), (0.3, 0.1, -0.2)

        function packed_cloud(qs, prevs)
            m = length(qs)
            data = fill(zero(Vlasov.PackedParticle{Float64}), m)
            P = Vlasov.PackedPositions(data, x0, h, false)
            O = Vlasov.PackedPositions(data, x0, h, true)
            for i in 1:m
                P[i] = qs[i]; O[i] = prevs[i]
            end
            ParticleCloud(P, O, fill((0.0, 0.0, 0.0), m), w)
        end

        # Zero force: Verlet is exact, so the packed form must be exact too.
        free = packed_cloud([q0], [q0 .- dt .* v])
        local diag
        for _ in 1:20
            diag = step!(free, dt)
        end
        @test all(free.positions[1] .≈ q0 .+ (20dt) .* v)
        M = mass(free)
        @test diag.kinetic ≈ M * (v[1]^2 + v[2]^2 + v[3]^2) / 2

        # `rcmax` and the angular momentum read the absolute position, which the
        # packed form has to rebuild: same split as on plain coordinates.
        qs = [(1.0, 0.0, 0.0), (5.0, 0.0, 0.0), (9.0, 0.0, 0.0)]
        three_p = packed_cloud(qs, qs)
        three_v = ParticleCloud(copy(qs), copy(qs), fill((0.0, 0.0, 0.0), 3), w)
        @test step!(three_p, dt; rcmax = 7.0).escaped ==
              step!(three_v, dt; rcmax = 7.0).escaped

        # Against the reference integrator, on a force field that actually
        # curves the trajectory: the two must agree to the storage precision.
        start = [(1.0 + 0.1i, 0.5i, -0.3i) for i in 1:50]
        prev = [q .- dt .* (0.2, -0.1, 0.05) for q in start]
        pk = packed_cloud(start, prev)
        vc = ParticleCloud(copy(start), copy(prev), fill((0.0, 0.0, 0.0), 50), w)
        local dp, dv
        for _ in 1:40
            for i in 1:50
                r = vc.positions[i]
                g = (-0.5 / (1 + sum(abs2, r))) .* r
                vc.forces[i] = g
                pk.forces[i] = g          # same field, so only the arithmetic differs
            end
            dv = step!(vc, dt); dp = step!(pk, dt)
        end
        @test maximum(i -> maximum(abs, pk.positions[i] .- vc.positions[i]), 1:50) < 1e-10
        @test dp.kinetic ≈ dv.kinetic
        @test all(isapprox.(dp.angular, dv.angular; atol = 1e-10))
    end

    @testset "Leapfrog priming" begin
        M, dt = 0.25, 0.1
        q = [(1.0, 2.0, -1.0), (0.0, 0.5, 3.0)]
        p = [(0.4, -0.2, 0.1), (-0.3, 0.0, 0.2)]
        f = [(1.0, 0.0, -1.0), (0.5, 0.5, 0.5)]

        half = half_step_back(q, p, M, dt)
        @test all(half[1] .≈ q[1] .- (dt / 2M) .* p[1])

        # The two coefficients differ by the documented factor: the original
        # code applies dt/M where Taylor gives dt²/4M.
        faithful = full_step_back(q, half, f, M, dt)
        consistent = full_step_back(q, half, f, M, dt; consistent = true)
        gap = faithful[1] .- consistent[1]
        @test all(gap .≈ (dt / M - dt^2 / 4M) .* f[1])

        # The consistent variant must reproduce the Taylor expansion of a
        # uniformly accelerated motion, which the other one does not.
        a = (2.0, -1.0, 0.5)
        q0 = [(0.0, 0.0, 0.0)]
        v0 = (1.0, 2.0, -0.5)
        exact(t) = q0[1] .+ t .* v0 .+ (t^2 / 2) .* a
        half0 = [exact(-dt / 2)]
        F = [M .* a]
        @test all(full_step_back(q0, half0, F, M, dt; consistent = true)[1] .≈ exact(-dt))
    end

    @testset "Locating on an axis" begin
        ax = uniform_axis(0.0, 10.0, 10)
        @test nearest_knot(ax.knots, 0.4) == 1
        @test nearest_knot(ax.knots, 0.6) == 2
        @test nearest_knot(ax.knots, 5.0) == 6
        @test nearest_knot(ax.knots, 10.0) == 11
        @test cell_index(ax.knots, 0.0) == 1
        @test cell_index(ax.knots, 3.5) == 4
        @test cell_index(ax.knots, -0.1) === nothing
        @test cell_index(ax.knots, 10.1) === nothing
    end

    @testset "Electric fields" begin
        ax = uniform_axis(-50.0, 50.0, 28)
        n = nbasis(ax)
        axes3 = (ax, ax, ax)

        """1D spline coefficients of a polynomial, by Hermite interpolation."""
        function coefs1d(a, p, p′)
            c = zeros(nbasis(a))
            for k in 1:nknots(a)
                c[linearindex(BasisIndex(k, Value))] = p(a.knots[k])
                c[linearindex(BasisIndex(k, Slope))] = p′(a.knots[k])
            end
            c
        end

        # Potential Φ = x: the gradient is (1,0,0), hence E = (−1,0,0).
        cx = coefs1d(ax, identity, _ -> 1.0)
        c1 = coefs1d(ax, _ -> 1.0, _ -> 0.0)
        csol = [cx[i] * c1[j] * c1[k] for i in 1:n, j in 1:n, k in 1:n]
        pts = ((2.3, -1.7, 4.1), (-8.0, 0.5, -3.3), (11.2, 6.6, -9.9))

        # The unsmoothed field is the exact gradient of the interpolant: the
        # Hermite basis reproduces cubics, hence a fortiori affine functions.
        for p in pts
            E = spline_field(axes3, csol, p)
            @test all(E .≈ (-1.0, 0.0, 0.0)) || maximum(abs, E .- (-1.0, 0.0, 0.0)) < 1e-13
        end
        @test spline_field(axes3, csol, (200.0, 0.0, 0.0)) === nothing
        @test spline_potential(axes3, csol, (200.0, 0.0, 0.0)) === nothing

        # ⚠️ Performance non-regression. `cell_index` returns
        # `Union{Nothing,Int}`; letting that union into the construction of the
        # tuples propagates the instability, boxes everything and multiplies the
        # cost by a hundred. These two functions must stay frugal: the test
        # fails if the union reappears.
        q = (2.3, -1.7, 4.1)
        spline_potential(axes3, csol, q); spline_field(axes3, csol, q)
        loop(f, k) = for _ in 1:k; f(axes3, csol, q); end
        loop(spline_potential, 10); loop(spline_field, 10)
        @test @allocated(loop(spline_potential, 1000)) < 100_000
        @test @allocated(loop(spline_field, 1000)) < 100_000

        # The smoothed field is NOT exact: the tabulated kernel is not exactly
        # normalised (see `GaussianSmoothing`). We bound the known error — this
        # test would fail if it worsened.
        sm = GaussianSmoothing(ax; nbdt = 200, quadrature = 200)
        for p in pts
            E = smoothed_field(axes3, csol, sm, p)
            @test maximum(abs, E .- (-1.0, 0.0, 0.0)) < 1e-4
        end

        # Direct diagnostic of the kernel's normalisation.
        cst = [isodd(a) ? 1.0 : 0.0 for a in 1:10]
        S0 = [sum(@view(sm.overlap[:, i]) .* cst) for i in axes(sm.overlap, 2)]
        D0 = [sum(@view(sm.gradient[:, i]) .* cst) for i in axes(sm.gradient, 2)]
        @test maximum(abs, S0 .- 1) < 1e-5      # must be 1
        @test maximum(abs, D0) < 1e-4           # must be 0
        @test sm.σ ≈ (ax.knots[2] - ax.knots[1]) / 3
    end

    @testset "Forces on the cloud" begin
        fine = uniform_axis(-50.0, 50.0, 28)
        coarse = uniform_axis(-150.0, 150.0, 28)
        n = nbasis(fine)
        sm = GaussianSmoothing(fine; nbdt = 100, quadrature = 100)
        csol = zeros(n, n, n)

        # Three particles, one per regime: inside, between the two grids, and
        # outside both.
        w = 0.01
        cloud = ParticleCloud([(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (400.0, 0.0, 0.0)], w)
        nout = forces!(cloud, (fine, fine, fine), csol,
                       (coarse, coarse, coarse), csol, sm; escaped = 7)
        @test nout == 2

        # Zero potential: the first two regimes give no force at all.
        @test all(iszero, cloud.forces[1])
        @test all(iszero, cloud.forces[2])

        # The third falls back on the Coulomb monopole of the enclosed charge.
        r = 400.0
        @test cloud.forces[3][1] ≈ -w^2 * 7 / r^2
        @test cloud.forces[3][2] == 0
    end

    @testset "Lookup table" begin
        # A stretched grid, the one that was expensive: bisection there made up
        # half the coarse deposit.
        for ax in (uniform_axis(-10.0, 10.0, 8), stretched_axis(50.0, 150.0, 7, 8),
                   testaxis(9; L = 3.0))
            tbl = LocateTable(ax)
            gt = ax.colloc
            # The table must return **exactly** the same result, not
            # approximately: it is an optimisation, not an approximation.
            for u in range(gt[1], gt[end]; length = 997)
                @test locate(tbl, ax, u) == locate(ax, u)
            end
            # The bounds and the outside.
            @test locate(tbl, ax, gt[1]) == locate(ax, gt[1])
            @test locate(tbl, ax, gt[end]) == locate(ax, gt[end])
            @test locate(tbl, ax, gt[1] - 1e-9) === nothing
            @test locate(tbl, ax, gt[end] + 1e-9) === nothing
            # Fine enough that a one-notch correction suffices: each
            # collocation interval covers at least one bucket.
            @test length(tbl.cell) >= length(gt)
        end
    end

    @testset "Sort by cell" begin
        ax = uniform_axis(-10.0, 10.0, 8)          # h = 2.5; 9 knots
        n = 500
        rng = Ran2(-1)
        pos = [ntuple(_ -> 20 * (Float64(next!(rng)) - 0.5), 3) for _ in 1:n]

        cs = CellSort(ax, n, 4)
        cellsort!(cs, pos)

        # The permutation really is a permutation.
        @test sort(cs.perm) == 1:n
        # Particles are grouped by cell, and the cells are increasing.
        cells = [cs.keys[i] for i in cs.perm]
        @test issorted(cells)
        # The bounds describe exactly those groups.
        @test length(cs.bounds) == noccupied(cs) + 1
        @test cs.bounds[1] == 0 && cs.bounds[end] == n
        @test all(g -> all(==(cs.occupied[g]), cells[cs.bounds[g]+1:cs.bounds[g+1]]),
                  1:noccupied(cs))
        # No empty cell in the list, and they are increasing.
        @test issorted(cs.occupied) && allunique(cs.occupied)
        @test all(g -> cs.bounds[g+1] > cs.bounds[g], 1:noccupied(cs))

        # ⚠️ Two calls in a row must give the same result: the counters are
        # reset. Forgetting produced wrong offsets, hence an out-of-bounds
        # write.
        perm1 = copy(cs.perm); occ1 = copy(cs.occupied); b1 = copy(cs.bounds)
        cellsort!(cs, pos)
        @test cs.perm == perm1 && cs.occupied == occ1 && cs.bounds == b1

        # A non-uniform grid must be refused: the cell index is computed, not
        # searched for.
        @test_throws ArgumentError CellSort(testaxis(8), n, 2)
    end

    @testset "Smoothed deposit" begin
        using Random
        rng = Random.MersenneTwister(99)
        ax = uniform_axis(-50.0, 50.0, 28)
        mesh = SplineMesh(ax, ax, ax)
        sm = GaussianSmoothing(ax; nbdt = 100, quadrature = 100)
        n = nbasis(ax)

        npart, nbelec = 5_000, 196.0
        positions = [ntuple(_ -> 8.0 * randn(rng), 3) for _ in 1:npart]
        ρ = zeros(n, n, n)
        nout = deposit_smoothed!(ρ, mesh, sm, positions; charge = nbelec / npart)
        @test nout == 0

        # Renormalisation makes the conservation exact, where the tabulated
        # kernel alone would be off by a few 1e-4.
        @test total_charge(ρ, mesh) ≈ nbelec rtol = 1e-12

        # The smoothed deposit spreads ONE particle over 8³ collocation points,
        # where the trilinear one touches only 2³: that is its whole purpose, to
        # soften what an 8-point interpolation would make abrupt. Measuring it
        # on the whole cloud would say nothing — the stencils overlap there and
        # both saturate the occupied region.
        one = [(1.3, -0.7, 2.1)]
        ρ1, ρtri = zeros(n, n, n), zeros(n, n, n)
        deposit_smoothed!(ρ1, mesh, sm, one; charge = 1.0)
        deposit!(ρtri, mesh, one; charge = 1.0)
        @test count(!iszero, ρ1) == 8^3
        @test count(!iszero, ρtri) == 2^3

        # ⚠️ Non-regression: the parallel deposit must return EXACTLY what the
        # sequential one returns, up to rounding, and be deterministic. A
        # deposit is a scatter; if two threads share a slot the error is silent
        # and grows with the number of threads.
        for k in (1, 2, 4, Threads.nthreads())
            buffers = ScatterBuffers(mesh; nslots = k)
            runs = map(1:3) do _
                q = zeros(n, n, n)
                deposit_smoothed!(q, mesh, sm, positions; charge = nbelec / npart,
                                  buffers = buffers)
                q
            end
            @test all(r -> r == runs[1], runs)          # deterministic
            @test norm(runs[1] - ρ) / norm(ρ) < 1e-13   # identical to sequential
        end

        # The same check for the trilinear deposit, which carried the defect.
        for k in (1, 2, Threads.nthreads())
            buffers = ScatterBuffers(mesh; nslots = k)
            ref = zeros(n, n, n)
            deposit!(ref, mesh, positions; charge = nbelec / npart)
            runs = map(1:3) do _
                q = zeros(n, n, n)
                deposit!(q, mesh, positions; charge = nbelec / npart, buffers = buffers)
                q
            end
            @test all(r -> r == runs[1], runs)
            @test norm(runs[1] - ref) / norm(ref) < 1e-13
        end

        # Particles too close to the edge are rejected: their 8-point stencil
        # would overflow.
        h = ax.knots[2] - ax.knots[1]
        edge = [(ax.knots[1] + h / 4, 0.0, 0.0), (0.0, ax.knots[end] - h / 4, 0.0)]
        ρb = zeros(n, n, n)
        @test deposit_smoothed!(ρb, mesh, sm, [edge; positions];
                                charge = nbelec / npart) == 2
    end

    @testset "A complete time step" begin
        # Chains the four stages on a single grid: deposit, Poisson, forces,
        # Verlet. The fine/coarse matching (`makerhsf`) is not yet ported,
        # hence the single grid.
        using Random
        rng = Random.MersenneTwister(7)
        ax = uniform_axis(-50.0, 50.0, 28)
        mesh = SplineMesh(ax, ax, ax)
        sm = GaussianSmoothing(ax; nbdt = 100, quadrature = 100)
        n = nbasis(ax)

        npart, nbelec, dt = 2_000, 196.0, 1.0
        positions = [ntuple(_ -> 6.0 * randn(rng), 3) for _ in 1:npart]
        cloud = ParticleCloud(positions, nbelec / npart)
        copyto!(cloud.previous, cloud.positions)          # starting at rest

        ρ = zeros(n, n, n)
        @test deposit_smoothed!(ρ, mesh, sm, cloud.positions;
                                charge = cloud.weight) == 0
        @test total_charge(ρ, mesh) ≈ nbelec rtol = 1e-12

        φ = poisson(ρ, mesh)
        csol = spline_coefficients(φ, mesh)
        @test all(isfinite, csol)

        nout = forces!(cloud, mesh.axes, csol, mesh.axes, csol, sm)
        @test nout < npart ÷ 10                  # most of them are at the centre
        @test all(f -> all(isfinite, f), cloud.forces)

        diag = step!(cloud, dt)
        @test diag.kinetic > 0                   # the cloud has started moving
        @test all(p -> all(isfinite, p), cloud.positions)

        # The particles start from rest in a cloud with globally zero momentum:
        # the total angular momentum stays small next to the energy.
        @test maximum(abs, diag.angular) < 1e3
    end

    @testset "Nested grids" begin
        fine = uniform_axis(-10.0, 10.0, 8)
        coarse = uniform_axis(-30.0, 30.0, 8)
        mf = SplineMesh(fine, fine, fine)
        mc = SplineMesh(coarse, coarse, coarse)

        nested = NestedMeshes(mf, mc)
        @test length(nested) == 2
        @test finest(nested) === mf
        @test coarsest(nested) === mc
        @test nested[1] === mf
        @test collect(nested) == [mf, mc]

        # Nesting is a condition, not a calling convention: a fine grid that
        # overflows would have no boundary values to read.
        @test_throws ArgumentError NestedMeshes(mc, mf)

        # A single level remains legal: that is the single-grid case.
        @test length(NestedMeshes(mf)) == 1

        # Two-level solve. A density concentrated at the centre must produce a
        # potential that is continuous across the grid change.
        cx, cy, cz = map(a -> a.colloc, mf.axes)
        ρf = [exp(-(x^2 + y^2 + z^2) / 2) for x in cx, y in cy, z in cz]
        gx, gy, gz = map(a -> a.colloc, mc.axes)
        ρc = [exp(-(x^2 + y^2 + z^2) / 2) for x in gx, y in gy, z in gz]

        φs = (similar(ρf), similar(ρc))
        poisson!(φs, (ρf, ρc), nested)
        @test all(isfinite, φs[1])

        # The faces of the fine grid must equal the interpolation of the coarse
        # solution — that is the definition of the matching.
        coefs = spline_coefficients(φs[2], mc)
        @test φs[1][1, 4, 6] ≈ spline_potential(mc.axes, coefs, (cx[1], cy[4], cz[6]))
        @test φs[1][end, 2, 3] ≈ spline_potential(mc.axes, coefs, (cx[end], cy[2], cz[3]))

        # And the interior stays the solution of its own system.
        rhs = poisson_rhs(ρf, mf)   # multipole boundaries: a different problem
        inner = φs[1][2:end-1, 2:end-1, 2:end-1]
        lap = laplacian!(similar(inner), inner, mf)
        rhs_matched = poisson_rhs!(similar(rhs), ρf, mf, φs[1])
        @test norm(lap - rhs_matched) / norm(rhs_matched) < 1e-10
        @test norm(rhs - rhs_matched) / norm(rhs_matched) > 1e-6   # different boundaries
    end

    @testset "Mean field" begin
        N = 196.0
        jel = Jellium(N)
        r0 = jel.radius
        @test r0 ≈ WIGNER_SEITZ_NA * cbrt(N)
        @test Jellium(N; rs = 2.0).radius ≈ 2.0 * cbrt(N)

        # Uniformly charged sphere: potential and field continuous at r₀,
        # finite at the centre, Coulombic far away.
        @test Vlasov.potential(jel, 0.0) ≈ -3N / 2r0
        @test Vlasov.potential(jel, r0) ≈ -N / r0
        @test Vlasov.potential(jel, nextfloat(r0)) ≈ -N / r0
        @test Vlasov.potential(jel, 1e6) ≈ -N / 1e6
        dr = 1e-6
        inside = (Vlasov.potential(jel, r0) - Vlasov.potential(jel, r0 - dr)) / dr
        outside = (Vlasov.potential(jel, r0 + dr) - Vlasov.potential(jel, r0)) / dr
        @test inside ≈ outside rtol = 1e-4          # continuous field
        @test all(r -> Vlasov.potential(jel, r) < 0, (0.0, 1.0, r0, 100.0))

        # Exchange-correlation: attractive, and growing in magnitude with the
        # density — this is what binds the electrons to one another.
        @test xc_potential(0.0) == 0
        @test xc_potential(1e-3) < 0
        @test xc_potential(1e-2) < xc_potential(1e-3)
        # Exchange dominates at high density: Vxc ~ −(3/π)^⅓ ρ^⅓.
        @test xc_potential(1e3) / (-cbrt(3 / π) * cbrt(1e3)) ≈ 1 rtol = 0.2

        # The addition really happens in coefficient space.
        ax = uniform_axis(-30.0, 30.0, 10)
        mesh = SplineMesh(ax, ax, ax)
        n = nbasis(ax)
        g = ax.colloc
        ρ = [1e-3 * exp(-(x^2 + y^2 + z^2) / 200) for x in g, y in g, z in g]
        csol = zeros(n, n, n)
        effective_potential!(csol, ρ, mesh, jel)
        expected = [xc_potential(ρ[i, j, k]) +
                    Vlasov.potential(jel, sqrt(g[i]^2 + g[j]^2 + g[k]^2))
                    for i in 1:n, j in 1:n, k in 1:n]
        @test csol ≈ spline_coefficients(expected, mesh)

        # Called twice, it adds twice: this is a cumulative `!`.
        csol2 = copy(csol)
        effective_potential!(csol2, ρ, mesh, jel)
        @test csol2 ≈ 2 .* csol
    end

    @testset "Initial sampling" begin
        # Analytic profile: constant density inside a ball of radius R.
        R, nq = 10.0, 1001
        # r(q) for a uniform ball: q = (r/R)³, hence r = R·q^⅓.
        quantiles = [R * cbrt((j - 1) / (nq - 1)) for j in 1:nq]
        rmax = 12.0
        density = [r <= R ? 1e-3 : 0.0 for r in range(0, rmax; length = nq)]
        prof = RadialProfile(quantiles, density, rmax)

        npart, nbelec = 20_000, 100.0
        w = nbelec / npart
        pos, mom = sample_thomas_fermi(prof, npart, w)
        @test length(pos) == length(mom) == npart

        # The radii must fill the ball, and their cube be uniform — that is the
        # property the inverse transform must guarantee.
        r = [sqrt(sum(abs2, p)) for p in pos]
        @test maximum(r) <= R + 1e-9
        @test abs(sum(x -> (x / R)^3, r) / npart - 0.5) < 0.02

        # Uniform directions on the sphere: each reduced component has zero
        # mean, and the polar cosine is uniform on [−1, 1].
        μ = [p[3] / sqrt(sum(abs2, p)) for p in pos]
        @test abs(sum(μ) / npart) < 0.02
        @test abs(sum(abs2, μ) / npart - 1 / 3) < 0.02

        # Momenta inside the local Fermi sphere, scaled by the weight.
        pf = FERMI_COEFFICIENT * cbrt(1e-3)
        pnorm = [sqrt(sum(abs2, p)) / w for p in mom]
        @test maximum(pnorm) <= pf * (1 + 1e-9)
        # Uniform in volume: ⟨(p/p_F)³⟩ = 1/2.
        @test abs(sum(x -> (x / pf)^3, pnorm) / npart - 0.5) < 0.02

        # Reproducibility: same seed, same cloud.
        p2, m2 = sample_thomas_fermi(prof, 100, w; rng = Ran2(-1))
        p3, m3 = sample_thomas_fermi(prof, 100, w; rng = Ran2(-1))
        @test p2 == p3 && m2 == m3

        # `initial_cloud` primes the leapfrog half a step backwards.
        dt = 0.5
        cloud = initial_cloud(prof, 500, nbelec, dt)
        @test length(cloud) == 500
        @test cloud.weight ≈ nbelec / 500
        M = mass(cloud)
        pos4, mom4 = sample_thomas_fermi(prof, 500, nbelec / 500)
        @test cloud.positions == pos4
        @test all(i -> all(cloud.previous[i] .≈ pos4[i] .- (dt / 2M) .* mom4[i]), 1:500)
    end

    @testset "Rejection sampling (initialise4, 1998 version)" begin
        # Analytic profile: V(r) = −p_F²/2 constant inside a ball, so the
        # criterion `p²/2 + V < E_F` reduces to `p < p_F`. We then know what the
        # sampling must produce, without depending on a file.
        R, n = 10.0, 1001
        pf = 0.4
        grid = collect(range(0, R; length = n))
        potential = fill(-pf^2 / 2, n)
        prof = PotentialProfile(grid, potential, R, pf, 0.0)
        @test prof isa PhaseSpaceProfile

        npart, nbelec = 20_000, 100.0
        w = nbelec / npart
        pos, mom = sample_thomas_fermi(prof, npart, w)
        @test length(pos) == length(mom) == npart

        # The whole ball is accepted (V is constant), so r³ stays uniform…
        r = [sqrt(sum(abs2, p)) for p in pos]
        @test maximum(r) <= R + 1e-9
        @test abs(sum(x -> (x / R)^3, r) / npart - 0.5) < 0.02
        # …and p is bounded by p_F, uniform in volume.
        pn = [sqrt(sum(abs2, p)) / w for p in mom]
        @test maximum(pn) <= pf * (1 + 1e-9)
        @test abs(sum(x -> (x / pf)^3, pn) / npart - 0.5) < 0.02

        # The rejection must bite: with E_F below the bottom of the well,
        # nothing passes beyond a finite radius. Here we halve the Fermi sphere.
        prof2 = PotentialProfile(grid, potential, R, pf, -pf^2 / 2 + pf^2 / 8)
        _, mom2 = sample_thomas_fermi(prof2, 5_000, w)
        @test maximum(sqrt(sum(abs2, p)) / w for p in mom2) <= pf / 2 * (1 + 1e-9)

        # Reproducibility, including across the rejections.
        a = sample_thomas_fermi(prof, 200, w; rng = Ran2(-1))
        b = sample_thomas_fermi(prof, 200, w; rng = Ran2(-1))
        @test a == b

        # `initial_cloud` accepts both profiles: that is the point of the
        # `PhaseSpaceProfile` supertype.
        cloud = initial_cloud(prof, 300, nbelec, 0.5)
        @test length(cloud) == 300
    end

    @testset "Energy budget" begin
        jel = Jellium(196.0)

        # Self-energy of a uniformly charged ball: 3Q²/5R.
        @test ion_self_energy(jel) ≈ 3 * 196.0^2 / (5 * jel.radius)
        @test ion_self_energy(Jellium(1.0; rs = 1.0)) ≈ 0.6

        # The mean field is obtained by difference, and the total is the sum.
        b = energy_budget(jel, 10.0, 100.0, -500.0)
        @test b.meanfield ≈ -500.0 - 200.0
        @test b.kinetic == 10.0 && b.hartree == 100.0
        @test b.total ≈ b.ions + b.kinetic + b.hartree + b.meanfield

        # A constant potential Φ₀ gives Σ w·Φ₀ = N_elec·Φ₀: a direct check on
        # the sampling, independent of any reference.
        ax = uniform_axis(-50.0, 50.0, 28)
        sm = GaussianSmoothing(ax; nbdt = 100, quadrature = 100)
        n = nbasis(ax)
        mesh = SplineMesh(ax, ax, ax)
        φ0 = 2.5
        csol = spline_coefficients(fill(φ0, n, n, n), mesh)

        nbelec, npart = 196.0, 500
        pos = [(3.0 * cos(i), 3.0 * sin(i), 0.1i % 7 - 3) for i in 1:npart]
        cloud = ParticleCloud(pos, nbelec / npart)
        e = interaction_energy(cloud, (ax, ax, ax), csol, (ax, ax, ax), csol, sm)
        # ⚠️ 1e-4 and not 1e-12: the smoothing kernel is not exactly normalised
        # (see `GaussianSmoothing`), and it shows here.
        @test e ≈ nbelec * φ0 rtol = 1e-4
    end

    @testset "Simulation of an isolated cluster" begin
        oracle = joinpath(@__DIR__, "..", "ref", "fortran")
        if !isdir(joinpath(oracle, "data"))
            @info "radial profile missing — simulation not tested"
            @test_skip false
        else
            prof = read_radial_profile(joinpath(oracle, "data"))

            p = SimulationParameters(; nparticles = 2_000, nsteps = 6, dt = 1.0)
            sim = Simulation(p, prof)
            @test length(sim.cloud) == 2_000
            @test sim.jellium.radius ≈ WIGNER_SEITZ_NA * cbrt(196.0)

            # The priming really happened: `previous` is neither empty nor equal
            # to the positions, and the forces have been evaluated.
            @test sim.cloud.previous != sim.cloud.positions
            @test any(f -> any(!iszero, f), sim.cloud.forces)

            hist = run!(sim; nsteps = 6)
            @test length(hist) == 6
            @test all(b -> isfinite(b.total), hist)

            # The energy budget **observes**, it does not feed back: making it
            # periodic must leave the trajectory rigorously unchanged. That is
            # what licenses spacing it out — the Fortran called it only one step
            # in ten — and it is worth ×1.5 on the running time.
            a = Simulation(p, prof); run!(a; nsteps = 12, energy_every = 1)
            b = Simulation(p, prof); h = run!(b; nsteps = 12, energy_every = 4)
            @test a.cloud.positions == b.cloud.positions
            @test a.cloud.previous == b.cloud.previous
            @test length(h) == 3                       # steps 1, 5, 9
            @test all(x -> isfinite(x.total), h)

            # `step!(; energy = false)` advances without returning a budget.
            @test step!(b; energy = false) === nothing
            @test step!(b; energy = true) isa EnergyBudget
            @test_throws ArgumentError run!(b; nsteps = 1, energy_every = 0)

            # The total energy is a small difference of large terms: measuring
            # it against itself would exaggerate the drift. We report it against
            # the scale of the terms that make it up.
            b1 = hist[1]
            scale = abs(b1.hartree) + abs(b1.meanfield) + b1.ions
            tot = [b.total for b in hist]
            @test (maximum(tot) - minimum(tot)) / scale < 1e-3

            # Symplectic scheme: the energy oscillates, it does not drift.
            @test !all(diff(tot) .< 0) && !all(diff(tot) .> 0)

            # The deposited charge stays that of the electrons, step by step.
            @test total_charge(sim.ρ[1], sim.meshes[1]) ≈ 196.0 rtol = 1e-10

            # A cloud held as `(k, δ)` is the same simulation, not an
            # approximation of it: at equal storage precision the two
            # trajectories agree to the oracle's own level.
            ref = Simulation(p, prof)
            pk = Simulation(p, prof; packed = true, precision = Float64)
            @test pk.cloud.positions isa Vlasov.PackedPositions
            @test pk.cloud.forces isa Vector{NTuple{3,Float64}}
            local hr, hp
            for _ in 1:6
                hr = step!(ref); hp = step!(pk)
            end
            gap = maximum(i -> maximum(abs, ref.cloud.positions[i] .-
                                            pk.cloud.positions[i]), 1:length(ref.cloud))
            @test gap < 1e-11
            @test hp.total ≈ hr.total rtol = 1e-12

            # In `Float32` the offsets cost precision, but only on the offsets:
            # the cell index stays exact, so the drift stays far under the
            # physical dispersion.
            f32 = Simulation(p, prof; packed = true, precision = Float32)
            local h32
            for _ in 1:6
                h32 = step!(f32)
            end
            @test h32.total ≈ hr.total rtol = 1e-4

            # The resident path: cloud on the accelerator's own buffers, so the
            # integrator runs as a kernel and reads the forces where the field
            # kernel left them.
            #
            # ⚠️ Compared against the **resident** path on a host-held cloud,
            # not against the host path. Those two have never agreed on the
            # energy — see the note below — and the question here is only
            # whether holding the cloud as `(k, δ)` changes anything. It must
            # not — but note *what* "not" means now: the packed cloud is
            # physically **sorted**, so its particles sit at different indices
            # and its diagnostics are summed in a different order. The
            # comparison is therefore on the cloud as a **set**, and on the
            # energy to the last few bits rather than to the last one.
            res = Simulation(p, prof; backend = CPU(), precision = Float64,
                             packed = true)
            dev = Simulation(p, prof; backend = CPU(), precision = Float64)
            @test res.cloud.positions.data === res.device.accelerator.particles.host
            @test res.cloud.previous.data === res.device.accelerator.particles.host
            local hres, hdev
            for _ in 1:6
                hres = step!(res); hdev = step!(dev)
            end
            nc = length(dev.cloud)
            sorted_res = sort([res.cloud.positions[i] for i in 1:nc])
            sorted_dev = sort([dev.cloud.positions[i] for i in 1:nc])
            @test maximum(i -> maximum(abs, sorted_res[i] .- sorted_dev[i]),
                          1:nc) < 1e-12
            @test hres.total ≈ hdev.total rtol = 1e-11

            # ⚠️ Unrelated to the packed cloud, and worth knowing: the resident
            # path does **not** reproduce the host path's total energy, even at
            # `Float64` on `CPU()` where nothing is lost to precision. The two
            # differ by more than the total itself — the total being a small
            # difference of large terms, it magnifies whatever the two chains
            # do differently. The packed cloud reproduces its own path exactly,
            # which is what the tests above pin down.
            @test !isapprox(hdev.total, hr.total; rtol = 1e-3)
        end
    end

    @testset "Electron capture" begin
        dt = 1.0
        proj = Projectile(; mass = 1836.154, charge = 1.0, energy = 73.498,
                          x0 = 0.0, dt = dt, cutoff = 1.0)
        proj.position = (0.0, 0.0, 0.0)
        w = 0.02
        # Two inside (radius 10), two outside.
        pos = [(1.0, 0.0, 0.0), (0.0, 5.0, 0.0), (30.0, 0.0, 0.0), (0.0, 0.0, 40.0)]
        cloud = ParticleCloud(pos, w)
        cloud.forces[3] = (1.0, 2.0, 3.0)

        @test enclosed_charge(cloud, proj, 10.0) ≈ 2w
        @test enclosed_charge(cloud, proj, 2.0) ≈ w
        @test enclosed_charge(cloud, proj, 100.0) ≈ 4w

        q0 = proj.charge
        n, internal = capture!(cloud, proj; radius = 10.0)
        @test n == 2
        @test length(cloud) == 2
        @test proj.charge ≈ q0 - 2w            # the ion carries off two packets
        @test internal < 0                      # binding, hence negative energy

        # The survivors keep their state, not only their position.
        @test cloud.positions == [(30.0, 0.0, 0.0), (0.0, 0.0, 40.0)]
        @test cloud.forces[1] == (1.0, 2.0, 3.0)

        # No capture must change anything.
        q1 = proj.charge
        @test capture!(cloud, proj; radius = 1.0)[1] == 0
        @test proj.charge == q1 && length(cloud) == 2
    end

    @testset "Simulation with a projectile" begin
        oracle = joinpath(@__DIR__, "..", "ref", "fortran")
        if !isdir(joinpath(oracle, "data"))
            @test_skip false
        else
            prof = read_radial_profile(joinpath(oracle, "data"))
            p = SimulationParameters(; nparticles = 1_000, nsteps = 5, dt = 1.0)
            proj = Projectile(; mass = 1836.154, charge = 1.0, energy = 73.498,
                              impact = 0.0, x0 = -70.0, dt = 1.0, cutoff = 1.0)
            sim = Simulation(p, prof; projectile = proj)

            # ⚠️ The priming must NOT advance the projectile: it computes forces
            # at q(−dt/2), and letting it advance there would put it one step
            # ahead of the cloud.
            @test proj.position == (-70.0, 0.0, 0.0)

            v = proj.velocity[1]
            run!(sim; nsteps = 5)
            @test proj.position[1] ≈ -70.0 + 5 * v rtol = 1e-3
            @test proj.position[1] > -70.0          # it moves towards the cluster

            # Far from the cluster (radius ≈ 23), the slowing down is negligible.
            @test abs(energy_loss(proj)) * HARTREE_TO_EV < 1.0

            # Without a projectile the field carries it and the loop pays nothing.
            isolated = Simulation(p, prof)
            @test isolated.projectile === nothing
            @test run!(isolated; nsteps = 2) |> length == 2
        end
    end

    @testset "Simulation parameters" begin
        p = SimulationParameters()
        @test p.nfine == 28 && p.nelectrons == 196.0

        # Reading a `vlas.inp`: comment, value, alternating.
        inp = joinpath(@__DIR__, "..", "ref", "fortran", "vlas.inp")
        if isfile(inp)
            q = read_parameters(inp)
            @test q.nfine == 28
            @test q.rcluster == 50.0 && q.rbox == 150.0
            @test q.nions == 196.0 && q.nelectrons == 196.0
        end

        mktemp() do path, io
            write(io, "comment\n3\nanother\n4\n")
            close(io)
            @test_throws ArgumentError read_parameters(path)
        end
    end

    @testset "Exchange-correlation potential and energy" begin
        # `xc_potential` is the functional derivative of the
        # exchange-correlation energy, `xc_energy_density` is its density:
        # confusing the two is precisely the mistake the pair exists to prevent.
        # The test checks it on exchange alone, where the relation is exact:
        #   E_x = ∫ ε_x ρ  with ε_x = ¾·(−(3/π)^⅓ ρ^⅓)
        #   V_x = d(ε_x ρ)/dρ = −(3/π)^⅓ ρ^⅓ = (4/3)·ε_x
        # Correlation prevents global equality, but its weight is small.
        for ρ in (1e-3, 3.7e-3, 1e-2)
            εx = 3 / 4 * (-cbrt(3 / π)) * cbrt(ρ)
            vx = (-cbrt(3 / π)) * cbrt(ρ)
            @test vx ≈ 4 / 3 * εx
            # Each function must carry the right exchange term.
            @test xc_potential(ρ) < vx          # + correlation, negative
            @test xc_energy_density(ρ) < εx
            # …and stay clearly distinct from one another.
            @test xc_energy_density(ρ) > xc_potential(ρ)
        end
        # Zero density: the energy vanishes cleanly rather than diverging on
        # `rs → ∞`.
        @test xc_energy_density(0.0) == 0.0
        @test xc_potential(0.0) == 0.0
    end

    @testset "Uniformly charged sphere" begin
        Q, R = 3.0, 2.0
        # Continuity of the value and of the field at the surface.
        @test uniform_sphere_potential(Q, R, R) ≈ -Q / R
        @test uniform_sphere_potential(Q, R, nextfloat(R)) ≈ -Q / R
        @test uniform_sphere_potential(Q, R, 0.0) ≈ -3Q / 2R
        @test uniform_sphere_potential(Q, R, 100.0) ≈ -Q / 100
        h = 1e-6
        inside = (uniform_sphere_potential(Q, R, R) - uniform_sphere_potential(Q, R, R - h)) / h
        outside = (uniform_sphere_potential(Q, R, R + h) - uniform_sphere_potential(Q, R, R)) / h
        @test inside ≈ outside rtol = 1e-4

        # The jellium is a special case of it.
        jel = Jellium(196.0)
        @test Vlasov.potential(jel, 5.0) == uniform_sphere_potential(196.0, jel.radius, 5.0)
    end

    @testset "Projectile-electron softening" begin
        σ = 1.3
        g = GaussianSoftening(σ)
        b = BallSoftening(σ)

        # Direct form, the thesis's: [Erf(r/√2σ) − 2g(r)r]/r³ with g the
        # **one-dimensional** normalised Gaussian.
        gauss1d(r) = exp(-r^2 / 2σ^2) / (sqrt(2π) * σ)
        direct(r) = (SpecialFunctions.erf(r / (sqrt(2)σ)) - 2gauss1d(r) * r) / r^3

        # The series and the direct form must agree in the overlap — but it is
        # `direct` that sets the tolerance, not the series: at u = 0.49 the
        # subtraction has already lost a decimal and a half, and that is
        # precisely why the threshold exists.
        for u in (0.3, 0.45, 0.49)
            r = u * σ
            @test force_kernel(g, r^2) ≈ direct(r) rtol = 1e-9
        end
        # Beyond the threshold, `force_kernel` IS the direct form.
        for u in (0.6, 1.0, 2.0, 5.0)
            r = u * σ
            @test force_kernel(g, r^2) ≈ direct(r) rtol = 1e-14
        end

        # Limit at zero: √(2/π)/(3σ³), without catastrophic cancellation.
        @test force_kernel(g, 0.0) ≈ sqrt(2 / π) / (3σ^3) rtol = 1e-14
        # …and there the series is more accurate than the direct subtraction,
        # which loses its digits. That is the reason the threshold exists.
        @test isfinite(force_kernel(g, 1e-20))

        # Coulomb behaviour far away: k·r² → 1/r.
        for r in (8σ, 15σ)
            @test force_kernel(g, r^2) * r^3 ≈ 1 rtol = 1e-10
        end

        # Potential: Erf(r/√2σ)/r, with limit √(2/π)/σ.
        @test pair_potential(g, 1.0, 0.0) ≈ sqrt(2 / π) / σ rtol = 1e-14
        @test pair_potential(g, 2.0, 3σ) ≈ 2 * SpecialFunctions.erf(3 / sqrt(2)) / 3σ

        # The ball stays what it was: Coulomb outside, linear inside.
        @test force_kernel(b, (2σ)^2) ≈ 1 / (2σ)^3
        @test force_kernel(b, (σ / 2)^2) ≈ 1 / σ^3
        @test pair_potential(b, 1.0, 2σ) ≈ uniform_sphere_potential(1.0, σ, 2σ)

        # The point that motivates everything: at equal radius, the Gaussian is
        # markedly weaker at contact. That is where the stopping-power gap
        # between the code (ball) and the thesis (Gaussian) is born.
        @test force_kernel(g, σ^2) * σ < 0.25 * force_kernel(b, σ^2) * σ
        # …and the two meet again far away.
        @test force_kernel(g, (6σ)^2) ≈ force_kernel(b, (6σ)^2) rtol = 1e-6

        # The projectile carries its softening, and refuses ambiguity.
        kw = (mass = 1836.0, charge = 1.0, energy = 73.5, x0 = -70.0, dt = 1.0)
        @test Projectile(; kw..., cutoff = 1.0).softening === BallSoftening(1.0)
        @test Projectile(; kw..., softening = g).softening === g
        @test_throws ArgumentError Projectile(; kw...)
        @test_throws ArgumentError Projectile(; kw..., cutoff = 1.0, softening = g)
        @test Vlasov.cutoff(Projectile(; kw..., softening = g)) == σ
    end

    @testset "Projectile" begin
        dt = 1.0
        E0 = 73.498
        proj = Projectile(; mass = 1836.154, charge = 1.0, energy = E0,
                          impact = 0.0, x0 = -70.0, dt = dt, cutoff = 1.0)
        @test proj.position == (-70.0, 0.0, 0.0)
        # ⚠️ `2E0` is the literal 2.0 in Julia, not `2*E0`: write the product.
        @test proj.velocity[1] ≈ sqrt(2 * E0 / 1836.154)
        @test kinetic_energy(proj) ≈ E0
        @test energy_loss(proj) ≈ 0 atol = 1e-12
        # The previous step really is behind on the trajectory.
        @test proj.previous[1] < proj.position[1]

        # Zero force: uniform rectilinear motion, energy conserved.
        free = Projectile(; mass = 1836.154, charge = 1.0, energy = E0,
                          x0 = -70.0, dt = dt, cutoff = 1.0)
        v0 = free.velocity[1]
        for _ in 1:50
            step!(free, (0.0, 0.0, 0.0), dt)
        end
        @test free.position[1] ≈ -70.0 + 50 * dt * v0
        @test kinetic_energy(free) ≈ E0
        @test energy_loss(free) ≈ 0 atol = 1e-10

        # Back-reaction: the force returned to the projectile is the opposite of
        # the sum of those added to the particles, up to the jellium's share.
        w = 0.01
        cloud = ParticleCloud([(0.0, 0.0, 0.0), (2.0, 1.0, -1.0), (-3.0, 0.0, 2.0)], w)
        jel = Jellium(196.0)
        before = copy(cloud.forces)
        f, eel, ejel = projectile_forces!(cloud, proj, jel)
        # `sum` cannot add tuples: reduce explicitly.
        added = reduce((a, b) -> a .+ b, map((x, y) -> x .- y, cloud.forces, before))
        jellium_only = jel.nions * proj.charge / sum(abs2, proj.position)^1.5 .* proj.position
        @test all(isapprox.(f .+ added, jellium_only; rtol = 1e-10))

        # The softening bounds the force: at contact it does not diverge.
        contact = ParticleCloud([proj.position], w)
        fc, _, _ = projectile_forces!(contact, proj, jel)
        @test all(isfinite, fc)
        @test all(isfinite, contact.forces[1])
        @test all(iszero, contact.forces[1])       # zero force at the exact centre

        # …and the interaction energy stays finite there, equal to that at the
        # centre of a uniformly charged ball.
        _, e_contact, _ = projectile_forces!(ParticleCloud([proj.position], w), proj, jel)
        @test e_contact ≈ w * (-3 * proj.charge / (2 * Vlasov.cutoff(proj)))
    end

    @testset "Mode-d product" begin
        A = randn(4, 4)
        X = randn(4, 5, 6)

        # Mode-1 must agree with a slice-wise matrix product.
        Y = similar(X)
        apply_mode!(Y, A, X, 1)
        for j in 1:5, k in 1:6
            @test Y[:, j, k] ≈ A * X[:, j, k]
        end

        # Mode-2 acts on the second index.
        B = randn(5, 5)
        Z = similar(X)
        apply_mode!(Z, B, X, 2)
        for i in 1:4, k in 1:6
            @test Z[i, :, k] ≈ B * X[i, :, k]
        end

        # Mode-3 acts on the third.
        C = randn(6, 6)
        W = similar(X)
        apply_mode!(W, C, X, 3)
        for i in 1:4, j in 1:5
            @test W[i, j, :] ≈ C * X[i, j, :]
        end
    end

    @testset "Tensor solver" begin
        ax = testaxis(10)
        D = laplacian1d(CollocationMatrices(ax))
        op = DiagonalizedOperator(D)
        n = size(op)

        for N in (2, 3)
            s = TensorSolver(ntuple(_ -> op, N)...)
            B = randn(ntuple(_ -> n, N))
            X = solve(B, s)

            # Check through the direct operator: Σ_d D applied to mode d.
            T = zeros(size(B))
            tmp = similar(X)
            for d in 1:N
                apply_mode!(tmp, D, X, d)
                T .+= tmp
            end
            @test norm(T - B) / norm(B) < 1e-11
        end

        # `solve!` must accept X === B (in-place solve).
        s = TensorSolver(op, op, op)
        B = randn(n, n, n)
        ref = solve(B, s)
        inplace = copy(B)
        solve!(inplace, inplace, s)
        @test inplace ≈ ref

        # A non-real spectrum must be refused rather than silently truncated.
        rot = [0.0 -1.0; 1.0 0.0]
        @test_throws ArgumentError DiagonalizedOperator(rot)
    end

    # The kernels that carry the particle loops are written once, for every
    # backend. Run here on `CPU()` and in `Float64`, they can be held against
    # the scalar reference — which is the comparison no GPU can make, Metal
    # having no double precision.
    @testset "Portable particle kernels" begin
        npart = 5_000
        ax = uniform_axis(-78.0, 78.0, 44)
        sm = GaussianSmoothing(ax)
        mesh = SplineMesh(ax, ax, ax)
        n = nbasis(ax)
        knots = ax.knots
        nk = length(knots)
        h = (knots[end] - knots[1]) / (nk - 1)
        x0 = knots[1]
        backend = CPU()

        # Deterministic, and well inside: the stencils all fit, so what is
        # under test is the arithmetic and not the boundary fallback.
        pos = [(50cospi(0.021i), 50sinpi(0.013i), 45sinpi(0.031i)) for i in 1:npart]
        csol = [1e-3 * sinpi(0.01i + 0.02j + 0.03k) for i in 1:n, j in 1:n, k in 1:n]

        # --- packing --------------------------------------------------------
        parts = fill(zero(Vlasov.PackedParticle{Float64}), npart)
        Vlasov._pack_kd_kernel!(backend)(parts, pos, x0, h, Int32(nk);
                                         ndrange = npart)
        synchronize(backend)
        knode = [parts[i].knode[d] for d in 1:3, i in 1:npart]
        delta = [parts[i].delta[d] for d in 1:3, i in 1:npart]
        @test all(knode[d, i] == Vlasov.nearest_knot(knots, pos[i][d])
                  for i in 1:npart, d in 1:3)
        # ⚠️ The offset is taken from the knot **recomputed** as `x0 + (k−1)h`,
        # not from `knots[k]`. On a uniform axis the two agree mathematically,
        # and differ in the last bits because `knots` came out of a `range`.
        # The Fortran port did the same, and `δ` feeds a table column whose
        # width is thirty thousand times that difference.
        @test all(delta[d, i] == pos[i][d] - (x0 + (knode[d, i] - 1) * h)
                  for i in 1:npart, d in 1:3)
        @test maximum(abs(delta[d, i] - (pos[i][d] - knots[knode[d, i]]))
                      for i in 1:npart, d in 1:3) < 1e-12

        # --- the sort, which both kernels below are organised around ---------
        sorter = CellSort(ax, npart)
        cellsort!(sorter, pos)
        ncell = length(sorter.occupied)
        # ⚠️ The kernels below read the cloud **in order**: slot `s` is the
        # `s`-th sorted particle. So the particles are moved, not indexed
        # through `perm`, and the assertions map slot `s` back to `perm[s]`.
        sorted_parts = [parts[sorter.perm[s]] for s in 1:npart]

        # --- smoothed field: must reproduce `smoothed_field` exactly ---------
        # One group per occupied cell, each staging that cell's 10³ stencil. The
        # assertion is per particle and says nothing about the order they are
        # visited in — which is the point: the cell traversal is an optimisation
        # and must leave every value exactly where the scalar reference puts it.
        force = zeros(Float64, 3, npart)
        red = zeros(Float64, 4)
        w = 0.245
        GS = Vlasov.FIELD_GROUPSIZE
        Vlasov._smoothed_field_kernel!(backend, GS)(
            force, csol, sm.overlap, sm.gradient, sorted_parts,
            sorter.occupied, sorter.bounds, Int32(nk), x0, h,
            sm.spacing, Int32(sm.nbdt), w, Int32(size(sm.overlap, 2)),
            0.0, 0.0, 0.0, 0.0, 1.0, red;
            ndrange = ncell * GS)
        synchronize(backend)
        @test !any(isnan, force)
        @test all(Tuple(force[:, s]) ===
                  w .* smoothed_field((ax, ax, ax), csol, sm, pos[sorter.perm[s]])
                  for s in 1:npart)

        # --- deposition: sorted, one group per occupied cell -----------------
        half = sm.spacing / 2
        lo, hi = knots[2] + half, knots[end-1] - half
        ncol = size(sm.nodes, 2)
        # ⚠️ The columns are indexed by the **slot**, like everything the sorted
        # cloud feeds: `_columns_kernel!` runs after the sort and so produces
        # them in that order already.
        cols = Matrix{Int32}(undef, 3, npart)
        nout = 0
        for s in 1:npart
            i = sorter.perm[s]
            p = pos[i]
            if all(d -> lo <= p[d] <= hi, 1:3)
                for d in 1:3
                    cols[d, s] = clamp(floor(Int32, (delta[d, i] + half) / sm.spacing *
                                             sm.nbdt + 0.5) + Int32(1),
                                       Int32(1), Int32(ncol))
                end
            else
                cols[1, s] = cols[2, s] = cols[3, s] = Int32(1)
                nout += 1
            end
        end

        ρ = zeros(Float64, n, n, n)
        DGS = Vlasov.DEPOSIT_GROUPSIZE
        Vlasov._deposit_sorted_kernel!(backend, DGS)(
            ρ, sm.nodes, cols, sorter.occupied, sorter.bounds,
            Int32(nk); ndrange = ncell * DGS)
        synchronize(backend)
        ρ .*= (npart - nout) / total_charge(ρ, mesh)

        ρref = zeros(Float64, n, n, n)
        noutref = deposit_smoothed!(ρref, mesh, sm, pos; charge = 1.0)
        @test nout == noutref
        # Only the order of summation differs from the reference.
        @test maximum(abs, ρ .- ρref) / maximum(abs, ρref) < 1e-14
        @test total_charge(ρ, mesh) ≈ npart - nout rtol = 1e-12
    end

    # `_face_point` inverts the enumeration `foreach_face` performs: rather than
    # sweeping the volume and rejecting the interior, it sends a work-item index
    # straight to a surface point. The two must agree exactly — which is what
    # keeps `foreach_face` in the file: it is the oracle for that arithmetic.
    @testset "Surface enumeration" begin
        for (nx, ny, nz) in ((3, 4, 5), (5, 6, 7), (8, 8, 8), (134, 134, 134))
            ref = Set{NTuple{3,Int}}()
            Vlasov.foreach_face(nx, ny, nz) do i, j, k
                push!(ref, (i, j, k))
            end
            n = Vlasov.nface(nx, ny, nz)
            got = Set(Int.(Vlasov._face_point(Int32(t), Int32(nx), Int32(ny),
                                              Int32(nz))) for t in 1:n)
            @test length(ref) == n      # no point visited twice
            @test got == ref
        end
    end

    # Every function of the Poisson chain was split in two: one half taking the
    # tables, one taking the mesh that holds them. On `CPU()` in `Float64` the
    # device mirror therefore runs *the same code over the same numbers* as the
    # host mesh — so anything but bit-for-bit equality here is a real defect in
    # the split, not a rounding difference.
    @testset "Device mirror of a mesh" begin
        axf = uniform_axis(-78.0, 78.0, 16)
        axc = uniform_axis(-235.0, 235.0, 16)
        fine, coarse = SplineMesh(axf, axf, axf), SplineMesh(axc, axc, axc)
        nested = NestedMeshes(fine, coarse)
        nf, nc = nbasis(axf), nbasis(axc)

        gf, gc = axf.colloc, axc.colloc
        ρf = [1e-3 * exp(-(x^2 + y^2 + z^2) / 400) for x in gf, y in gf, z in gf]
        ρc = [1e-5 * exp(-(x^2 + y^2 + z^2) / 4000) for x in gc, y in gc, z in gc]

        dmf = DeviceMesh(CPU(), Float64, fine)
        dmc = DeviceMesh(CPU(), Float64, coarse)

        φref = (zeros(nf, nf, nf), zeros(nc, nc, nc))
        poisson!(φref, (ρf, ρc), nested)
        φdev = (zeros(nf, nf, nf), zeros(nc, nc, nc))
        poisson!(φdev, (ρf, ρc), (dmf, dmc))
        @test φdev[1] == φref[1]
        @test φdev[2] == φref[2]

        @test total_charge(ρf, dmf) == total_charge(ρf, fine)
        @test multipole(ρc, dmc).charge == multipole(ρc, coarse).charge

        @test spline_coefficients!(similar(φref[1]), φref[1], dmf) ==
              spline_coefficients!(similar(φref[1]), φref[1], fine)

        jel = Jellium(1000.0)
        er, ed = zeros(nf, nf, nf), zeros(nf, nf, nf)
        effective_potential!(er, ρf, fine, jel)
        effective_potential!(ed, ρf, dmf, jel)
        @test ed == er
    end

    @testset "Generic accelerator on CPU()" begin
        npart = 4_000
        ax = uniform_axis(-78.0, 78.0, 44)
        cax = uniform_axis(-235.0, 235.0, 20)
        sm = GaussianSmoothing(ax)
        mesh = SplineMesh(ax, ax, ax)
        n, ncb = nbasis(ax), nbasis(cax)
        w = 196 / npart
        # A tenth deliberately outside the fine grid, to exercise the fallback.
        pos = map(1:npart) do i
            r = i % 10 == 0 ? 95.0 : 50.0
            (r * cospi(0.021i), r * sinpi(0.013i), 0.8r * sinpi(0.031i))
        end
        csolf = [1e-3 * sinpi(0.01i + 0.02j + 0.03k) for i in 1:n, j in 1:n, k in 1:n]
        csolc = [1e-4 * cospi(0.02i + 0.01j + 0.03k)
                 for i in 1:ncb, j in 1:ncb, k in 1:ncb]

        acc = DeviceAccelerator(CPU(), Float64, (ax, ax, ax), sm, npart, n;
                                packed = true)

        # A cloud already held as `(k, δ)` deposits identically, with nothing
        # packed: same density, same out-of-stencil count. The tenth of the
        # sample that sits outside the fine grid is what makes this a real test
        # — those are the particles whose cell index is not clamped.
        pk = Vlasov.packed_cloud(ax, pos, w, Float64;
                                 storage = acc.particles.host)
        @test pk.positions.data === acc.particles.host
        @test pk.previous.data === acc.particles.host
        ρpk = zeros(n, n, n)
        npk = deposit_smoothed!(ρpk, acc, mesh, sm, pk.positions; charge = w)

        ρref = zeros(n, n, n)
        nref = deposit_smoothed!(ρref, mesh, sm, pos; charge = w)
        ρacc = zeros(n, n, n)
        nacc = deposit_smoothed!(ρacc, acc, mesh, sm, pos; charge = w)
        @test nacc == nref
        @test maximum(abs, ρacc .- ρref) / maximum(abs, ρref) < 1e-14
        @test total_charge(ρacc, mesh) ≈ total_charge(ρref, mesh) rtol = 1e-12
        # The packed route agrees with the reference to the same tolerance, and
        # agrees on which particles fell outside.
        @test npk == nref
        @test maximum(abs, ρpk .- ρref) / maximum(abs, ρref) < 1e-14

        # The two-stage device sort leaves the particles **physically ordered**,
        # so the test is on the storage itself rather than on a permutation:
        # read in order, the cell keys must be non-decreasing, and the cell
        # list must agree with a host sort of the same cloud. The order
        # *within* a cell is not specified by either.
        nk32 = Int32(acc.sorter.nknots)
        keys = [Vlasov._cell_key(acc.particles.host[i], nk32) for i in 1:npart]
        @test issorted(keys)
        @test length(acc.sorter.occupied) == length(unique(keys))
        @test acc.sorter.occupied == unique(keys)
        @test acc.sorter.bounds[end] == npart

        c1 = ParticleCloud(pos, w)
        c2 = ParticleCloud(pos, w)
        forces!(c1, (ax, ax, ax), csolf, (cax, cax, cax), csolc, sm)
        forces!(c2, acc, (ax, ax, ax), csolf, (cax, cax, cax), csolc, sm)

        # ⚠️ The two paths do not classify boundary particles alike, and never
        # have: `forces!` on the CPU asks whether the **position** clears a
        # two-knot margin, the kernel whether the 10³ **stencil** fits. So they
        # are compared where both chose the smoothed fine-grid field — and
        # there they must agree exactly, `Float64` on both sides.
        h = (ax.knots[end] - ax.knots[1]) / (length(ax.knots) - 1)
        lo, hi = ax.knots[3], ax.knots[end-2]
        both = map(1:npart) do i
            p = pos[i]
            cpu = all(d -> lo < p[d] < hi, 1:3)
            k = ntuple(d -> clamp(round(Int, (p[d] - ax.knots[1]) / h) + 1,
                                  1, length(ax.knots)), 3)
            cpu && all(d -> 2k[d] - 5 >= 1 && 2k[d] + 4 <= n, 1:3)
        end
        @test count(both) > 0.8npart
        @test all(c1.forces[i] === c2.forces[i] for i in 1:npart if both[i])
    end
end
