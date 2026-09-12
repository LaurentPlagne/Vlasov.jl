"""
Comparaison au code Fortran d'origine.

Ces tests ne s'exécutent que si l'oracle a été produit :

    cd ref/fortran && make oracle

Ils sont ignorés sinon — les dumps binaires ne sont pas versionnés, et le
reste de la suite valide déjà les mêmes propriétés sans référence extérieure
(dérivée seconde exacte sur un cubique, convergence à l'ordre 4). L'oracle
apporte autre chose : la certitude qu'on résout *le même problème* que la
thèse, et pas seulement un problème correct.
"""

const ORACLE_DIR = joinpath(@__DIR__, "..", "ref", "fortran")

"""Lit un vecteur dumpé : un `Int32` de taille, puis des `Float64`."""
function read_dump_vector(name)
    open(joinpath(ORACLE_DIR, name)) do io
        n = Int(read(io, Int32))
        [read(io, Float64) for _ in 1:n]
    end
end

"""Lit une matrice carrée dumpée, en ordre colonne comme Fortran et Julia."""
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

# Écart relatif toléré. Le portage remplace des briques (NAG f02agf → `eigen`,
# inversion explicite → factorisation), l'égalité bit-à-bit n'est donc pas
# attendue ; au-delà de ce seuil, en revanche, c'est une régression.
const ORACLE_TOL = 1e-13

reldiff(a, b) = norm(a - b) / norm(b)

@testset "Oracle Fortran" begin
    if oracle_available()
        @testset "Générateur ran2" begin
            # Le seul test bit-à-bit de la suite : la moindre différence dans
            # l'arithmétique entière — y compris le débordement volontaire —
            # décalerait toute la séquence, et avec elle le tirage initial.
            ref = read_dump_vector("rand2.bin")
            rng = Ran2(-1)
            @test all(i -> Float64(next!(rng)) === ref[i], eachindex(ref))
        end
    end

    if !oracle_available()
        @info "oracle absent — `cd ref/fortran && make oracle` pour l'activer"
        @test_skip false
    else
        # `vlas.inp` : 28 intervalles sur [-xclu, xclu] avec xclu = 50.
        axf = uniform_axis(-50.0, 50.0, 28)

        @testset "Grille fine" begin
            @test reldiff(axf.knots, read_dump_vector("dump_gx.bin")) < ORACLE_TOL
            @test reldiff(axf.colloc, read_dump_vector("dump_gtx.bin")) < ORACLE_TOL
        end

        @testset "Collocation et opérateur" begin
            cm = CollocationMatrices(axf)
            @test reldiff(Matrix(cm.S), read_dump_matrix("dump_sx.bin")) < ORACLE_TOL
            @test reldiff(Matrix(cm.S″), read_dump_matrix("dump_s2x.bin")) < ORACLE_TOL

            D = laplacian1d(cm)
            @test reldiff(D, read_dump_matrix("dump_dex.bin")) < ORACLE_TOL

            λ = read_dump_vector("dump_lxr.bin")
            op = DiagonalizedOperator(D)
            @test reldiff(sort(op.λ), sort(λ)) < ORACLE_TOL
            # Propriété que le Fortran calculait puis jetait (`lxi`, `mxi`).
            @test all(<(0), op.λ)
        end

        @testset "Moments multipolaires" begin
            for (k, file) in ((0, "dump_psx.bin"), (1, "dump_pxx.bin"), (2, "dump_px2.bin"))
                @test reldiff(moments(axf, Val(k)), read_dump_vector(file)) < ORACLE_TOL
            end
        end

        @testset "Grille grossière étirée" begin
            # ⚠️ On part des nœuds DUMPÉS, pas d'un axe reconstruit : `findacc`
            # arrêtait sa dichotomie à 1e-10, et `stretched_axis` résout à la
            # précision machine. Reconstruire l'axe polluerait toute la
            # comparaison à ~1e-12 et masquerait les vraies régressions.
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

            # La grille étirée reconstruite, elle, ne doit coller qu'à la
            # tolérance de `findacc` — vérifier qu'on reste dans cet ordre.
            axs = stretched_axis(50.0, 150.0, 7, 8)
            @test reldiff(axs.knots, read_dump_vector("dumpb_gx.bin")) < 1e-10
        end

        @testset "Dépôt de charge" begin
            # `makerho` dumpe les particules ET la densité de la même
            # invocation : les deux restent donc cohérentes, quel que soit le
            # moment de la simulation où le dump a été pris.
            qp = read_dump_vector("dumpqp.bin")
            npart = length(qp) ÷ 3
            positions = [(qp[3i-2], qp[3i-1], qp[3i]) for i in 1:npart]

            axb = SplineAxis(read_dump_vector("dumpb_gx.bin"),
                             read_dump_vector("dumpbgtx.bin"))
            mesh = SplineMesh(axb, axb, axb)
            n = nbasis(axb)
            ρref = reshape(read_dump_vector("dumprho.bin"), n, n, n)

            ρ = similar(ρref)
            nout = deposit!(ρ, mesh, positions; charge = 196.0 / npart)
            @test nout == 0
            @test reldiff(ρ, ρref) < ORACLE_TOL

            # L'observable de contrôle du code d'origine : « somme des charges ».
            @test total_charge(ρ, mesh) ≈ 196.0 rtol = 1e-12
        end

        @testset "Poisson : densité → potentiel" begin
            # `makerh2` dumpe sa densité, sa grille et son second membre ; le
            # `solve` qui la suit dumpe le potentiel, apparié par un drapeau en
            # COMMON (`solve` est aussi appelée après `makerhsf`, sur l'autre
            # grille, et son csol ne correspondrait pas à cette densité-ci).
            axb = SplineAxis(read_dump_vector("dumpb_gx.bin"),
                             read_dump_vector("dumpbgtx.bin"))
            @test read_dump_vector("dumprh2gt.bin") ≈ axb.colloc

            mesh = SplineMesh(axb, axb, axb)
            n = nbasis(axb)
            ρ = reshape(read_dump_vector("dumprh2.bin"), n, n, n)

            # Moments multipolaires. Le quadrupôle tolère un peu plus : sa
            # forme sans trace (2∫x² − ∫y² − ∫z²) compense de grands nombres.
            mp = multipole(ρ, mesh)
            @test mp.charge ≈ read_dump_vector("dumpq.bin")[1] rtol = 1e-12
            @test reldiff(collect(mp.center), read_dump_vector("dumpbari.bin")) < 1e-11
            Q = reshape(read_dump_vector("dumpquad.bin"), 3, 3)
            @test reldiff(collect(mp.quadrupole[1:3]), [Q[1, 1], Q[2, 2], Q[3, 3]]) < 1e-11
            @test reldiff(collect(mp.quadrupole[4:6]), [Q[1, 2], Q[1, 3], Q[2, 3]]) < 1e-11

            # Potentiel de bord : seules les faces sont écrites par `makerh2`,
            # l'intérieur du tableau `phi` sert de tampon à `solve`.
            φbord = boundary_potential!(similar(ρ), mesh, mp)
            φref = reshape(read_dump_vector("dumpphi.bin"), n, n, n)
            faces = falses(n, n, n)
            faces[1, :, :] .= true; faces[end, :, :] .= true
            faces[:, 1, :] .= true; faces[:, end, :] .= true
            faces[:, :, 1] .= true; faces[:, :, end] .= true
            @test reldiff(φbord[faces], φref[faces]) < 1e-12

            @test reldiff(poisson_rhs(ρ, mesh),
                          reshape(read_dump_vector("dumprhs.bin"), size(mesh)...)) < 1e-11

            # La chaîne complète, et ses coefficients spline (`csol`).
            φ = poisson(ρ, mesh)
            @test reldiff(φ, reshape(read_dump_vector("dumpphi2.bin"), n, n, n)) < 1e-11
            @test reldiff(spline_coefficients(φ, mesh),
                          reshape(read_dump_vector("dumpcsol.bin"), n, n, n)) < 1e-11
        end

        @testset "Pas de Verlet" begin
            triplets(v) = [(v[3i-2], v[3i-1], v[3i]) for i in 1:length(v)÷3]
            flat(t) = collect(Iterators.flatten(t))

            q0 = triplets(read_dump_vector("mv_qp0.bin"))
            qold0 = triplets(read_dump_vector("mv_qo0.bin"))
            forces = triplets(read_dump_vector("mv_fp.bin"))
            dt = read_dump_vector("mv_par.bin")[1]

            # `move` dumpe aussi `coef` = 1/2M : de quoi retrouver la masse de
            # la pseudo-particule sans la supposer.
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

        @testset "Champs et forces" begin
            triplets(v) = [(v[3i-2], v[3i-1], v[3i]) for i in 1:length(v)÷3]
            flat(t) = collect(Iterators.flatten(t))

            gx = read_dump_vector("fg_gx.bin")
            gxb = read_dump_vector("fg_gxb.bin")
            fine = SplineAxis(gx, collocation_points(gx))
            coarse = SplineAxis(gxb, collocation_points(gxb))

            # Tables de convolution gaussienne.
            sm = GaussianSmoothing(fine)
            @test sm.spacing ≈ read_dump_vector("fg_pas.bin")[1]
            @test reldiff(sm.overlap, reshape(read_dump_vector("fg_it1.bin"), 10, :)) < 1e-11
            @test reldiff(sm.gradient, reshape(read_dump_vector("fg_it2.bin"), 10, :)) < 1e-11

            # Forces sur les pseudo-particules.
            n = nbasis(fine)
            csol = reshape(read_dump_vector("fg_csol.bin"), n, n, n)
            csolb = reshape(read_dump_vector("fg_csolb.bin"), n, n, n)
            qp = triplets(read_dump_vector("fg_qp.bin"))
            cloud = ParticleCloud(qp, 196.0 / length(qp))
            nout = forces!(cloud, (fine, fine, fine), csol,
                           (coarse, coarse, coarse), csolb, sm;
                           escaped = Int(read_dump_vector("fg_n.bin")[1]))
            # Sur ce dump, toutes les particules sont dans la grille fine ; les
            # régimes grossier et monopolaire sont couverts par la suite
            # autonome, pas ici.
            @test nout == 0
            @test reldiff(flat(cloud.forces), read_dump_vector("fg_fp.bin")) < 1e-11
        end

        @testset "Dépôt lissé" begin
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

        @testset "Raccord entre grilles" begin
            gxf = read_dump_vector("fg_gx.bin")
            gxc = read_dump_vector("sf_gxc.bin")
            axf = SplineAxis(gxf, read_dump_vector("sf_gt.bin"))
            axc = SplineAxis(gxc, collocation_points(gxc))
            fine = SplineMesh(axf, axf, axf)
            coarse = SplineMesh(axc, axc, axc)
            nested = NestedMeshes(fine, coarse)
            n = nbasis(axf)

            ρf = reshape(read_dump_vector("sf_rho.bin"), n, n, n)
            csolc = reshape(read_dump_vector("sf_csolc.bin"), n, n, n)

            # Les valeurs de bord lues dans la solution grossière.
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

            # La chaîne complète à deux niveaux, depuis les deux densités.
            ρc = reshape(read_dump_vector("dumprh2.bin"), n, n, n)
            φs = (Array{Float64,3}(undef, n, n, n), Array{Float64,3}(undef, n, n, n))
            poisson!(φs, (ρf, ρc), nested)
            @test reldiff(φs[1], reshape(read_dump_vector("sf_phi2.bin"), n, n, n)) < 1e-11
            @test reldiff(φs[2], reshape(read_dump_vector("dumpphi2.bin"), n, n, n)) < 1e-11
        end
    end
end
