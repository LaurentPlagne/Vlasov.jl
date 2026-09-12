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
