using Vlasov
using LinearAlgebra
using Test

"""Grille non uniforme de test, dans l'esprit de celle du code d'origine."""
function testaxis(n = 8; L = 1.0)
    knots = [L * (t + 0.15 * sinpi(2t)) for t in range(0, 1; length = n + 1)]
    # Deux points de collocation par nœud, comme l'exige `SplineAxis`.
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

    @testset "BasisIndex" begin
        # Aller-retour indice linéaire ↔ (nœud, nature).
        for lin in 1:20
            @test linearindex(BasisIndex(lin)) == lin
        end
        @test BasisIndex(1) == BasisIndex(1, Value)
        @test BasisIndex(2) == BasisIndex(1, Slope)
        @test linearindex(BasisIndex(3, Slope)) == 6
    end

    @testset "Base d'Hermite" begin
        ax = testaxis()
        g = ax.knots

        for k in 2:(nknots(ax)-1)
            bv = BasisIndex(k, Value)
            bs = BasisIndex(k, Slope)

            # Propriétés définissantes de la base d'Hermite au nœud porteur.
            @test value(ax, bv, g[k]) ≈ 1 atol = 1e-14
            @test value(ax, bs, g[k]) ≈ 0 atol = 1e-14
            @test derivative(ax, bv, g[k]) ≈ 0 atol = 1e-12
            @test derivative(ax, bs, g[k]) ≈ 1 atol = 1e-12

            # Annulation aux nœuds voisins.
            @test value(ax, bv, g[k-1]) ≈ 0 atol = 1e-14
            @test value(ax, bv, g[k+1]) ≈ 0 atol = 1e-14
        end

        # Support local : deux intervalles seulement.
        b = BasisIndex(4, Value)
        lo, hi = support(ax, b)
        @test (lo, hi) == (g[3], g[5])
        @test value(ax, b, g[2]) == 0
        @test value(ax, b, g[6]) == 0
    end

    @testset "Reproduction des cubiques" begin
        # L'interpolant d'Hermite reproduit exactement tout polynôme de degré ≤ 3.
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

    @testset "Construction d'axes" begin
        ax = uniform_axis(-50.0, 50.0, 28)
        @test nknots(ax) == 29
        @test nbasis(ax) == 58
        @test ax.knots[1] == -50.0
        @test ax.knots[end] ≈ 50.0
        @test maximum(diff(ax.knots)) - minimum(diff(ax.knots)) < 1e-12

        # Les points de collocation sont les nœuds de Gauss à 2 points de chaque
        # intervalle, bornés par les extrémités du domaine.
        @test ax.colloc[1] == ax.knots[1]
        @test ax.colloc[end] == ax.knots[end]
        h = ax.knots[2] - ax.knots[1]
        @test ax.colloc[2] ≈ ax.knots[1] + h * (1 - 1 / sqrt(3)) / 2
        @test ax.colloc[3] ≈ ax.knots[1] + h * (1 + 1 / sqrt(3)) / 2

        # Raison géométrique : elle doit vérifier son équation de définition.
        h1, L, n = 50.0 / 6, 100.0, 8
        a = stretch_ratio(h1, L, n)
        @test a > 1
        @test L * (1 - a) / (1 - a^n) ≈ h1 atol = 1e-13

        axs = stretched_axis(50.0, 150.0, 7, 8)
        @test nknots(axs) == 29
        @test axs.knots[1] ≈ -150.0
        @test axs.knots[end] ≈ 150.0
        @test axs.knots[15] ≈ 0 atol = 1e-12          # nœud central
        @test axs.knots ≈ -reverse(axs.knots)          # symétrie
        # Raccord sans rupture : le premier pas étiré vaut le pas constant.
        d = diff(axs.knots)
        @test d[8] ≈ h1 rtol = 1e-9
        @test issorted(axs.knots)
    end

    @testset "Moments" begin
        # Identité exacte : l'interpolant d'Hermite d'un polynôme de degré ≤ 3
        # étant ce polynôme lui-même, ∫xᵏp = Σ_b c_b·moment(b,k) où les c_b sont
        # les valeurs et pentes de p aux nœuds. Vaut pour k = 0, 1, 2 et ne
        # dépend d'aucune référence extérieure.
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

        # Un moment ne dépend que du support de sa fonction de base.
        ax = uniform_axis(0.0, 1.0, 6)
        b = BasisIndex(3, Value)
        lo, hi = support(ax, b)
        @test moment(ax, b, Val(0)) > 0
        @test moment(ax, b, Val(0)) < hi - lo
    end

    @testset "Matrices de collocation" begin
        ax = testaxis(12)
        cm = CollocationMatrices(ax)
        m = nbasis(ax)
        @test size(cm.S) == (m, m)

        # Structure bande : rien au-delà de la 2ᵉ sur-/sous-diagonale.
        for i in 1:m, j in 1:m
            abs(i - j) > 2 && @test cm.S[i, j] == 0
        end

        # Les matrices de collocation ne sont pas symétriques : lignes = points,
        # colonnes = fonctions de base.
        @test !issymmetric(cm.S)

        # Cohérence : S évalue bien la base aux points de collocation.
        for k in 2:(m-1)
            for lin in 1:m
                @test cm.S[k, lin] ≈ value(ax, BasisIndex(lin), ax.colloc[k]) atol = 1e-14
            end
        end
    end

    @testset "Opérateur 1D" begin
        ax = testaxis(12)
        cm = CollocationMatrices(ax)
        D = laplacian1d(cm)
        @test size(D, 1) == nbasis(ax) - 2

        # Non-régression : `laplacian1d` doit coïncider avec le chemin dense de
        # référence. Sur BandedMatrices v1.12.0, la division à droite entre deux
        # `BandedMatrix` rend un résultat FAUX en silence (résidu ≈ 0.3) — ce test
        # échoue si quelqu'un « simplifie » `Matrix(S″) / lu(S)` en `S″ / S`.
        Sdense, S2dense = Matrix(cm.S), Matrix(cm.S″)
        Dref = (S2dense / Sdense)[2:end-1, 2:end-1]
        @test norm(D - Dref) / norm(Dref) < 1e-12

        # La division doit résoudre son équation de définition.
        X = Matrix(cm.S″) / lu(cm.S)
        @test norm(X * Sdense - S2dense) / norm(S2dense) < 1e-12

        # Propriété constatée, sur laquelle repose toute la méthode tensorielle :
        # spectre réel et strictement négatif.
        λ = eigvals(D)
        @test maximum(abs, imag.(λ)) < 1e-10 * maximum(abs, real.(λ))
        @test all(<(0), real.(λ))
    end

    @testset "D est bien la dérivée seconde" begin
        # L'interpolant d'Hermite d'un cubique étant exact, D·f doit rendre f″
        # EXACTEMENT aux points de collocation intérieurs. Les deux lignes
        # extrêmes encodent les conditions au bord, pas une dérivée : c'est ce
        # que `laplacian1d` retire.
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

        function erreur(n)
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

        # L'opérateur direct doit être l'inverse du solveur.
        ρ = randn(size(mesh))
        Φ = solve(ρ, mesh)
        @test norm(laplacian!(similar(Φ), Φ, mesh) - ρ) / norm(ρ) < 1e-10

        # Solution manufacturée : convergence à l'ordre 4, propre à la
        # collocation cubique aux points de Gauss. Un ordre plus faible
        # signalerait une erreur de discrétisation, pas de précision.
        errs = map(erreur, (4, 8, 16))
        @test issorted(errs; rev = true)
        for i in 1:(length(errs)-1)
            @test log2(errs[i] / errs[i+1]) > 3.8
        end
        @test errs[end] < 1e-6
    end

    @testset "Dépôt de charge" begin
        using Random
        rng = Random.MersenneTwister(1234)

        ax = uniform_axis(-50.0, 50.0, 28)
        mesh = SplineMesh(ax, ax, ax)
        n = nbasis(ax)

        # Longueurs duales : leur somme doit recouvrir exactement le domaine.
        l = dual_lengths(ax)
        @test length(l) == n
        @test all(>(0), l)
        @test sum(l) ≈ ax.knots[end] - ax.knots[1]

        # Repérage. Tomber exactement sur le nœud k rend `(k-1, 0)` : tout le
        # poids va au nœud de droite, qui EST le nœud k. Convention du Fortran.
        c, w = locate(ax, ax.colloc[5])
        @test (c, w) == (4, 0.0)
        # Au bord gauche, il n'y a pas de nœud à gauche : poids 1 sur le nœud 1.
        @test locate(ax, ax.colloc[1]) == (1, 1.0)
        @test locate(ax, -60.0) === nothing
        @test locate(ax, 60.0) === nothing
        c, w = locate(ax, (ax.colloc[7] + ax.colloc[8]) / 2)
        @test c == 7 && w ≈ 0.5

        # Conservation de la charge : c'est LE contrôle du dépôt. Déposer N
        # électrons et réintégrer la densité doit rendre N, à l'arrondi près.
        npart, nbelec = 50_000, 196.0
        positions = [ntuple(_ -> 8.0 * randn(rng), 3) for _ in 1:npart]
        ρ = zeros(n, n, n)
        nout = deposit!(ρ, mesh, positions; charge = nbelec / npart)
        @test nout == 0
        @test total_charge(ρ, mesh) ≈ nbelec rtol = 1e-12

        # Les particules hors domaine sont comptées et ignorées, pas repliées.
        dehors = [(200.0, 0.0, 0.0), (0.0, -300.0, 0.0)]
        ρ2 = zeros(n, n, n)
        @test deposit!(ρ2, mesh, dehors; charge = 1.0) == 2
        @test all(iszero, ρ2)

        # Coefficients spline : S⁻¹ puis S doit rendre l'identité.
        coefs = spline_coefficients(ρ, mesh)
        back = similar(ρ)
        src = coefs
        for d in 1:3
            dst = similar(ρ)
            apply_mode!(dst, Matrix(mesh.collocation[d].S), src, d)
            src = dst
        end
        @test norm(src - ρ) / norm(ρ) < 1e-10

        # Une taille de tableau incohérente doit être refusée, pas tolérée.
        @test_throws DimensionMismatch deposit!(zeros(n, n, n - 1), mesh,
                                                positions; charge = 1.0)
    end

    @testset "Développement multipolaire" begin
        ax = uniform_axis(-20.0, 20.0, 16)
        mesh = SplineMesh(ax, ax, ax)
        n = nbasis(ax)

        # Une distribution à symétrie sphérique centrée n'a ni dipôle ni
        # quadrupôle : c'est ce qui rend le contrôle discriminant.
        cx, cy, cz = map(a -> a.colloc, mesh.axes)
        σ = 3.0
        ρ = [exp(-(x^2 + y^2 + z^2) / 2σ^2) for x in cx, y in cy, z in cz]
        ρ .*= 10.0 / total_charge(ρ, mesh)      # normalisée à 10 unités

        mp = multipole(ρ, mesh)
        @test mp.charge ≈ 10.0 rtol = 1e-10
        @test all(c -> abs(c) < 1e-8, mp.center)
        @test all(q -> abs(q) < 1e-6, mp.quadrupole)

        # Loin de la source, le potentiel tend vers celui d'une charge ponctuelle.
        for r in (1e3, 1e4)
            @test potential(mp, r, 0.0, 0.0) ≈ 10.0 / r rtol = 1e-6
        end

        # Décentrer la distribution doit déplacer le barycentre d'autant.
        shifted = [exp(-((x - 4)^2 + y^2 + z^2) / 2σ^2) for x in cx, y in cy, z in cz]
        mps = multipole(shifted, mesh)
        @test mps.center[1] ≈ 4.0 rtol = 1e-6
        @test abs(mps.center[2]) < 1e-8
    end

    @testset "Poisson avec bords multipolaires" begin
        ax = uniform_axis(-20.0, 20.0, 16)
        mesh = SplineMesh(ax, ax, ax)
        cx, cy, cz = map(a -> a.colloc, mesh.axes)
        ρ = [exp(-((x - 1)^2 + (y + 2)^2 + z^2) / 8) for x in cx, y in cy, z in cz]

        φ = poisson(ρ, mesh)
        @test size(φ) == size(ρ)

        # Sur les faces, le potentiel EST le développement multipolaire.
        mp = multipole(ρ, mesh)
        @test φ[1, 5, 7] ≈ potential(mp, cx[1], cy[5], cz[7])
        @test φ[end, 3, 9] ≈ potential(mp, cx[end], cy[3], cz[9])

        # À l'intérieur, l'opérateur appliqué à la solution doit rendre le
        # second membre — relèvement des bords compris.
        rhs = poisson_rhs(ρ, mesh)
        inner = φ[2:end-1, 2:end-1, 2:end-1]
        @test norm(laplacian!(similar(inner), inner, mesh) - rhs) / norm(rhs) < 1e-10

        # Le second membre est −4πρ plus le relèvement des bords. Celui-ci
        # décroît vers l'intérieur mais ne s'annule pas exactement : l'opérateur
        # complet n'est pas à support strictement local (largeur de bande 19
        # sur 56 mesurée, et des entrées ténues au-delà).
        releve = rhs .+ 4π .* @view ρ[2:end-1, 2:end-1, 2:end-1]
        c = size(mesh, 1) ÷ 2
        @test abs(releve[c, c, c]) < abs(releve[1, c, c])
        @test rhs[c, c, c] ≈ -4π * ρ[c+1, c+1, c+1] rtol = 1e-6

        @test_throws DimensionMismatch poisson_rhs!(zeros(3, 3, 3), ρ, mesh)
    end

    @testset "Intégration de Verlet" begin
        w, dt = 0.5, 0.05
        q0, v = (1.0, -2.0, 0.5), (0.3, 0.1, -0.2)

        # Force nulle : le Verlet est exact, la position est affine en temps.
        libre = ParticleCloud([q0], w)
        libre.previous[1] = q0 .- dt .* v
        local diag
        for _ in 1:20
            diag = step!(libre, dt)
        end
        @test all(libre.positions[1] .≈ q0 .+ (20dt) .* v)
        M = mass(libre)
        @test M == ELECTRON_MASS * w
        @test diag.kinetic ≈ M * (v[1]^2 + v[2]^2 + v[3]^2) / 2

        # Oscillateur harmonique : le Verlet ne conserve pas l'énergie
        # exactement, mais sans dérive — elle oscille dans une bande étroite.
        k = 3.0
        osc = ParticleCloud([(1.0, 0.0, 0.0)], w)
        osc.previous[1] = osc.positions[1]           # départ au repos
        énergies = Float64[]
        for _ in 1:4000
            osc.forces[1] = (-k) .* osc.positions[1]
            d = step!(osc, dt)
            # `d.kinetic` est l'énergie cinétique en t ; après le pas,
            # `previous` porte q(t). Mélanger les temps ferait osciller
            # l'énergie pour de mauvaises raisons.
            q = osc.previous[1]
            push!(énergies, d.kinetic + k * (q[1]^2 + q[2]^2 + q[3]^2) / 2)
        end
        amplitude = (maximum(énergies) - minimum(énergies)) / abs(first(énergies))
        @test amplitude < 1e-2
        # Pas de dérive : les deux moitiés de la trajectoire ont même moyenne.
        moitié = length(énergies) ÷ 2
        m1 = sum(énergies[1:moitié]) / moitié
        m2 = sum(énergies[moitié+1:end]) / moitié
        @test abs(m2 - m1) / abs(m1) < 1e-6

        # Force centrale : le moment cinétique se conserve.
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

    @testset "Amorçage du leapfrog" begin
        M, dt = 0.25, 0.1
        q = [(1.0, 2.0, -1.0), (0.0, 0.5, 3.0)]
        p = [(0.4, -0.2, 0.1), (-0.3, 0.0, 0.2)]
        f = [(1.0, 0.0, -1.0), (0.5, 0.5, 0.5)]

        demi = half_step_back(q, p, M, dt)
        @test all(demi[1] .≈ q[1] .- (dt / 2M) .* p[1])

        # Les deux coefficients diffèrent du facteur documenté : le code
        # d'origine applique dt/M là où Taylor donne dt²/4M.
        fidèle = full_step_back(q, demi, f, M, dt)
        homogène = full_step_back(q, demi, f, M, dt; consistent = true)
        écart = fidèle[1] .- homogène[1]
        @test all(écart .≈ (dt / M - dt^2 / 4M) .* f[1])

        # La variante homogène doit reproduire le développement de Taylor
        # d'un mouvement uniformément accéléré, ce que l'autre ne fait pas.
        a = (2.0, -1.0, 0.5)
        q0 = [(0.0, 0.0, 0.0)]
        v0 = (1.0, 2.0, -0.5)
        exact(t) = q0[1] .+ t .* v0 .+ (t^2 / 2) .* a
        demi0 = [exact(-dt / 2)]
        F = [M .* a]
        @test all(full_step_back(q0, demi0, F, M, dt; consistent = true)[1] .≈ exact(-dt))
    end

    @testset "Produit mode-d" begin
        A = randn(4, 4)
        X = randn(4, 5, 6)

        # Le mode-1 doit coïncider avec un produit matriciel par tranche.
        Y = similar(X)
        apply_mode!(Y, A, X, 1)
        for j in 1:5, k in 1:6
            @test Y[:, j, k] ≈ A * X[:, j, k]
        end

        # Le mode-2 agit sur le deuxième indice.
        B = randn(5, 5)
        Z = similar(X)
        apply_mode!(Z, B, X, 2)
        for i in 1:4, k in 1:6
            @test Z[i, :, k] ≈ B * X[i, :, k]
        end

        # Le mode-3 agit sur le troisième.
        C = randn(6, 6)
        W = similar(X)
        apply_mode!(W, C, X, 3)
        for i in 1:4, j in 1:5
            @test W[i, j, :] ≈ C * X[i, j, :]
        end
    end

    @testset "Solveur tensoriel" begin
        ax = testaxis(10)
        D = laplacian1d(CollocationMatrices(ax))
        op = DiagonalizedOperator(D)
        n = size(op)

        for N in (2, 3)
            s = TensorSolver(ntuple(_ -> op, N)...)
            B = randn(ntuple(_ -> n, N))
            X = solve(B, s)

            # Vérification par l'opérateur direct : Σ_d D appliqué au mode d.
            T = zeros(size(B))
            tmp = similar(X)
            for d in 1:N
                apply_mode!(tmp, D, X, d)
                T .+= tmp
            end
            @test norm(T - B) / norm(B) < 1e-11
        end

        # `solve!` doit accepter X === B (résolution en place).
        s = TensorSolver(op, op, op)
        B = randn(n, n, n)
        ref = solve(B, s)
        inplace = copy(B)
        solve!(inplace, inplace, s)
        @test inplace ≈ ref

        # Un spectre non réel doit être refusé plutôt que silencieusement tronqué.
        rot = [0.0 -1.0; 1.0 0.0]
        @test_throws ArgumentError DiagonalizedOperator(rot)
    end
end
