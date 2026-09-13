#!/usr/bin/env julia
"""
Dépôt de charge sur GPU : les deux voies, mesurées côte à côte.

    julia --project=gpu -t auto scripts/depot_gpu.jl

Le dépôt est un **scatter** : chaque particule écrit dans 8³ = 512 points de la
grille, et les particules voisines écrivent aux mêmes endroits. C'est le poste
le plus lourd du pas une fois les forces sur GPU, et le plus difficile à porter.

Deux voies, et elles ne se valent pas du tout :

  * **atomique** — une particule par fil, 512 additions atomiques. Simple, sans
    préparation, et **trois fois plus lent que le CPU** : 800 000 × 512 = 410
    millions d'atomiques en conflit.
  * **triée** — les particules sont rangées par maille, puis un groupe de 512
    fils traite une maille et **chaque fil possède un point du pochoir**. Il
    parcourt toutes les particules de la maille en accumulant dans un registre
    et ne fait qu'**une** atomique à la fin. Le renversement de boucle divise
    les atomiques par le nombre de particules par maille — ici 108.

C'est le tri de la thèse, et pour la même raison qu'en 1997 : la localité des
données. Il profite d'ailleurs aussi au CPU, qui gagne ×1,34 sur ce dépôt sans
rien changer d'autre.

Ce fichier est un **prototype mesuré**, pas le chemin de production : il
n'est pas encore branché dans `update_forces!`.
"""

using Vlasov, Metal, Printf, LinearAlgebra

const ROOT = dirname(@__DIR__)
const NPART = 800_000

# ---------------------------------------------------------------------------
# Tri par comptage, parallèle
# ---------------------------------------------------------------------------

"""Indice linéaire de la maille — le nœud le plus proche, comme le dépôt.
La grille fine étant uniforme, il se **calcule** : pas de dichotomie."""
@inline function cell_of(p, x0, h, nk)
    kx = clamp(round(Int32, (p[1] - x0) / h) + Int32(1), Int32(1), Int32(nk))
    ky = clamp(round(Int32, (p[2] - x0) / h) + Int32(1), Int32(1), Int32(nk))
    kz = clamp(round(Int32, (p[3] - x0) / h) + Int32(1), Int32(1), Int32(nk))
    kx + Int32(nk) * (ky - Int32(1) + Int32(nk) * (kz - Int32(1)))
end

"""Comptage par tranche. ⚠️ Remet les compteurs à zéro : sans cela une mesure
répétée les cumule, et le tri produit des indices hors bornes."""
function count_cells!(keys, partial, pos, chunks, x0, h, nk)
    Threads.@threads for t in eachindex(chunks)
        cnt = partial[t]
        fill!(cnt, Int32(0))
        @inbounds for i in chunks[t]
            c = cell_of(pos[i], x0, h, nk)
            keys[i] = c
            cnt[c] += Int32(1)
        end
    end
end

"""Liste des mailles **occupées**. Sur 91 125 mailles, 7 413 le sont : l'agrégat
de rayon 40 n'occupe qu'une fraction de la boîte de ±78."""
function occupied_cells(total::Vector{Int32})
    occ = Vector{Int32}(undef, count(>(Int32(0)), total))
    j = 0
    @inbounds for c in eachindex(total)
        total[c] > Int32(0) && (j += 1; occ[j] = Int32(c))
    end
    occ
end

"""Où chaque tranche écrit, pour chaque maille : le tri par comptage parallèle.

⚠️ Ne balayer que les mailles **occupées**. La version qui parcourait les
91 125 mailles coûtait 1,86 ms ; restreinte, elle en coûte 0,13, et dresser la
liste 0,06 — un gain net de 1,67 ms sur une préparation de 6,7."""
function slice_offsets!(offsets, partial, total, occ)
    acc = Int32(0)
    @inbounds for c in occ
        a = acc
        for t in eachindex(partial)
            offsets[t][c] = a
            a += partial[t][c]
        end
        acc += total[c]
    end
    offsets
end

"""Placement. Chaque fil écrit dans sa propre zone de chaque maille : aucune
synchronisation. ⚠️ Un `Dict` à la place de `offsets` coûtait dix millisecondes
à lui seul — vingt fois cette version."""
function place!(perm, keys, offsets, chunks)
    Threads.@threads for t in eachindex(chunks)
        cur = offsets[t]
        @inbounds for i in chunks[t]
            c = keys[i]
            cur[c] += Int32(1)
            perm[cur[c]] = Int32(i)
        end
    end
end

# ---------------------------------------------------------------------------
# Noyaux Metal
# ---------------------------------------------------------------------------

"""Version **atomique** : une particule par fil, 512 additions atomiques."""
function kernel_atomic!(ρ, nodes, pos, x0, h, nknots, spacing, nbdt, nc, lo, hi, npart)
    i = thread_position_in_grid_1d()
    i > npart && return nothing
    @inbounds begin
        px = pos[1, i]; py = pos[2, i]; pz = pos[3, i]
        (px < lo || px > hi || py < lo || py > hi || pz < lo || pz > hi) && return nothing
        kx = min(max(round(Int32, (px - x0) / h) + Int32(1), Int32(1)), nknots)
        ky = min(max(round(Int32, (py - x0) / h) + Int32(1), Int32(1)), nknots)
        kz = min(max(round(Int32, (pz - x0) / h) + Int32(1), Int32(1)), nknots)
        half = spacing * 0.5f0
        col(u, k) = min(max(floor(Int32, (u - (x0 + (k - Int32(1)) * h) + half) /
                            spacing * nbdt + 0.5f0) + Int32(1), Int32(1)), nc)
        cx = col(px, kx); cy = col(py, ky); cz = col(pz, kz)
        bx = Int32(2) * kx - Int32(5); by = Int32(2) * ky - Int32(5); bz = Int32(2) * kz - Int32(5)
        n = Int32(size(ρ, 1))
        (bx < Int32(0) || by < Int32(0) || bz < Int32(0) ||
         bx + Int32(8) > n || by + Int32(8) > n || bz + Int32(8) > n) && return nothing
        for kk in Int32(1):Int32(8), jj in Int32(1):Int32(8)
            c = nodes[jj, cy] * nodes[kk, cz]
            j = by + jj; k = bz + kk
            for ii in Int32(1):Int32(8)
                Metal.@atomic ρ[bx + ii, j, k] += nodes[ii, cx] * c
            end
        end
    end
    nothing
end

"""Colonnes de table, sur GPU.

C'est le plus gros morceau de la préparation du tri — 3,63 ms sur CPU — et il
n'a rien à y faire : il est purement particulaire, et les positions sont déjà
montées pour les forces. Seule la **permutation** doit monter, ce qui est un
vecteur d'entiers.

⚠️ **Ce noyau n'est pas retenu.** Calculé en `Float32`, il fait basculer 0,05 %
des colonnes sur leur voisine et porte l'écart de densité de 1,5e-07 à 9,0e-05.
La cause est structurelle : l'ULP de `Float32` à 78 a₀ vaut 7,6e-06, soit 0,22 %
de la largeur d'une colonne (0,00355 a₀). Une particule sur cinq cents est à
moins d'un ULP d'une frontière, et le mauvais échantillon de gaussienne lui est
appliqué — un choix discret faux, pas un arrondi qui se moyenne.

Il est gardé ici pour que la mesure soit rejouable. Voir `docs/gpu.md` pour ce
qu'il faudrait faire à la place : monter `(k, δ)` calculés en `Float64` sur
l'hôte, plutôt que les positions absolues.
"""
function kernel_cols!(cols, pos, perm, x0, h, nk, spacing, nbdt, nc, npart)
    s = thread_position_in_grid_1d()
    s > npart && return nothing
    @inbounds begin
        i = perm[s]
        half = spacing * 0.5f0
        for d in Int32(1):Int32(3)
            u = pos[d, i]
            k = min(max(round(Int32, (u - x0) / h) + Int32(1), Int32(1)), nk)
            cols[d, s] = min(max(floor(Int32, (u - (x0 + (k - Int32(1)) * h) + half) /
                                 spacing * nbdt + 0.5f0) + Int32(1), Int32(1)), nc)
        end
    end
    nothing
end

"""Version **triée** : un groupe de 512 fils par maille occupée, chaque fil
propriétaire d'un point du pochoir.

Les indices de table passent par la mémoire du groupe, chargés par tranches de
64 : sans cela les 512 fils reliraient chacun les données de chaque particule,
et multiplieraient le trafic mémoire par autant."""
function kernel_sorted!(ρ, nodes, cols, cellids, offs, nk, ncell_occ)
    g = threadgroup_position_in_grid_1d()
    g > ncell_occ && return nothing
    t = thread_index_in_threadgroup()

    @inbounds begin
        c0 = cellids[g] - Int32(1)
        kx = c0 % nk + Int32(1); c0 ÷= nk
        ky = c0 % nk + Int32(1); c0 ÷= nk
        kz = c0 + Int32(1)
        bx = Int32(2) * kx - Int32(5); by = Int32(2) * ky - Int32(5); bz = Int32(2) * kz - Int32(5)

        t0 = t - Int32(1)
        ii = t0 % Int32(8) + Int32(1); t0 ÷= Int32(8)
        jj = t0 % Int32(8) + Int32(1); t0 ÷= Int32(8)
        kk = t0 + Int32(1)

        n = Int32(size(ρ, 1))
        inside = bx >= Int32(0) && by >= Int32(0) && bz >= Int32(0) &&
                 bx + Int32(8) <= n && by + Int32(8) <= n && bz + Int32(8) <= n

        shared = MtlThreadGroupArray(Int32, 3 * 64)
        lo = offs[g]; hi = offs[g + Int32(1)]
        acc = 0.0f0
        p = lo
        while p < hi
            m = min(Int32(64), hi - p)
            if t <= m
                b = 3 * (t - Int32(1))
                shared[b + Int32(1)] = cols[1, p + t]
                shared[b + Int32(2)] = cols[2, p + t]
                shared[b + Int32(3)] = cols[3, p + t]
            end
            threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            if inside
                for q in Int32(1):m
                    b = 3 * (q - Int32(1))
                    acc += nodes[ii, shared[b + Int32(1)]] *
                           nodes[jj, shared[b + Int32(2)]] *
                           nodes[kk, shared[b + Int32(3)]]
                end
            end
            threadgroup_barrier(Metal.MemoryFlagThreadGroup)
            p += m
        end
        inside && acc != 0.0f0 && (Metal.@atomic ρ[bx + ii, by + jj, bz + kk] += acc)
    end
    nothing
end

# ---------------------------------------------------------------------------

chrono(f; k = 8) = (f(); t0 = time(); for _ in 1:k; f(); end; 1000(time() - t0) / k)

function main()
    grid, ρr = read_radial_density(joinpath(ROOT, "ref", "these", "rhorad.Na1000.dat"))
    p = SimulationParameters(nfine = 44, ninner = 22, nouter = 22, rcluster = 78.0,
                             rbox = 235.0, nions = 1000.0, nelectrons = 1000.0,
                             nparticles = NPART, nsteps = 0, dt = 1.0)
    sim = Simulation(p, PotentialProfile(grid, ρr);
                     projectile = Projectile(mass = 1836.154, charge = 1.0, energy = 147.0,
                                             x0 = -30.0, dt = 1.0,
                                             softening = GaussianSoftening(1.0)))
    step!(sim; energy = false)

    fine, sm, w = sim.meshes[1], sim.smoothing, sim.cloud.weight
    knots = fine.axes[1].knots
    x0 = knots[1]; h = (knots[end] - knots[1]) / (length(knots) - 1); nk = length(knots)
    halfsp = sm.spacing / 2
    pos = sim.cloud.positions

    ρref = similar(sim.ρ[1])
    deposit_smoothed!(ρref, fine, sm, pos; charge = w, buffers = sim.scatter[1])

    # --- tri ---------------------------------------------------------------
    nch = Threads.nthreads()
    chunks = [round(Int, NPART * (t - 1) / nch) + 1 : round(Int, NPART * t / nch) for t in 1:nch]
    ncell = nk^3
    keys = Vector{Int32}(undef, NPART)
    partial = [Vector{Int32}(undef, ncell) for _ in 1:nch]
    offsets = [Vector{Int32}(undef, ncell) for _ in 1:nch]
    perm = Vector{Int32}(undef, NPART)
    cols = Matrix{Int32}(undef, 3, NPART)
    # ⚠️ Tampon distinct : écrire les totaux dans `partial[1]` détruirait les
    # compteurs de la première tranche, dont `slice_offsets!` a besoin. Le coût
    # de l'oubli est une écriture hors bornes, donc une faute de segmentation.
    total = Vector{Int32}(undef, ncell)

    function sort!()
        count_cells!(keys, partial, pos, chunks, x0, h, nk)
        copyto!(total, partial[1])
        for t in 2:nch; total .+= partial[t]; end
        occ = occupied_cells(total)
        slice_offsets!(offsets, partial, total, occ)
        place!(perm, keys, offsets, chunks)
        occ
    end

    """Les colonnes, à l'ancienne : sur CPU, pour comparaison."""
    function cols_cpu!()
        Threads.@threads for s in 1:NPART
            @inbounds begin
                q = pos[perm[s]]
                for d in 1:3
                    k = clamp(round(Int32, (q[d] - x0) / h) + Int32(1), Int32(1), Int32(nk))
                    cols[d, s] = clamp(floor(Int32, (q[d] - (x0 + (k - 1) * h) + halfsp) /
                                       sm.spacing * sm.nbdt + 0.5) + Int32(1),
                                       Int32(1), Int32(size(sm.nodes, 2)))
                end
            end
        end
    end
    occ = sort!()
    offs = Int32[0]; acc = Int32(0)
    for c in occ; acc += total[c]; push!(offs, acc); end

    # --- GPU ----------------------------------------------------------------
    gnodes = MtlArray(Float32.(sm.nodes))
    gρ = MtlArray(zeros(Float32, size(ρref)...))
    gpos = MtlArray(zeros(Float32, 3, NPART))
    hostpos = Matrix{Float32}(undef, 3, NPART)
    @inbounds for i in 1:NPART
        q = pos[i]; hostpos[1, i] = q[1]; hostpos[2, i] = q[2]; hostpos[3, i] = q[3]
    end
    copyto!(gpos, hostpos)
    cols_cpu!()
    gcols = MtlArray(cols); gcells = MtlArray(occ); goffs = MtlArray(offs)
    gperm = MtlArray(perm)

    run_cols() = Metal.@sync @metal threads=256 groups=cld(NPART, 256) kernel_cols!(
        gcols, gpos, gperm, Float32(x0), Float32(h), Int32(nk), Float32(sm.spacing),
        Int32(sm.nbdt), Int32(size(sm.nodes, 2)), Int32(NPART))

    gs = 256
    run_atomic() = (fill!(gρ, 0f0); Metal.@sync @metal threads=gs groups=cld(NPART, gs) kernel_atomic!(
        gρ, gnodes, gpos, Float32(x0), Float32(h), Int32(nk), Float32(sm.spacing),
        Int32(sm.nbdt), Int32(size(sm.nodes, 2)), Float32(knots[2] + halfsp),
        Float32(knots[end-1] - halfsp), Int32(NPART)))
    run_sorted() = (fill!(gρ, 0f0); Metal.@sync @metal threads=512 groups=length(occ) kernel_sorted!(
        gρ, gnodes, gcols, gcells, goffs, Int32(nk), Int32(length(occ))))

    function normalised(k)
        k(); ρ = Float64.(Array(gρ)); ρ .*= w
        ρ .*= NPART * w / Vlasov.total_charge(ρ, fine)
        ρ
    end

    sorted_pos = [pos[i] for i in perm]
    buf = Vlasov.ScatterBuffers(fine)
    ρs = similar(ρref)

    @printf("Na1000, %d particules, grille %d — %d mailles occupées sur %d, %.0f par maille\n\n",
            NPART, p.nfine, length(occ), ncell, NPART / length(occ))
    @printf("%-34s %9s %14s\n", "version", "ms", "écart à réf.")
    for (nom, t, ρ) in (("CPU, ordre courant",
                         chrono(() -> deposit_smoothed!(ρref, fine, sm, pos;
                                                        charge = w, buffers = sim.scatter[1])), nothing),
                        ("CPU, ordre trié",
                         chrono(() -> deposit_smoothed!(ρs, fine, sm, sorted_pos;
                                                        charge = w, buffers = buf)), ρs),
                        ("GPU atomique", chrono(run_atomic), normalised(run_atomic)),
                        ("GPU trié", chrono(run_sorted), normalised(run_sorted)))
        e = ρ === nothing ? "" : @sprintf("%.1e", norm(ρ - ρref) / norm(ρref))
        @printf("%-34s %9.1f %14s\n", nom, t, e)
    end
    @printf("\n%-34s %9.2f\n", "tri : permutation (CPU)", chrono(sort!; k = 4))
    @printf("%-34s %9.2f\n", "colonnes de table (CPU)", chrono(cols_cpu!))
    @printf("%-34s %9.2f\n", "colonnes de table (GPU)", chrono(run_cols))

    # Les colonnes calculées en Float32 désignent-elles les mêmes ?
    cols_cpu!(); ref_cols = copy(cols)
    run_cols(); gpu_cols = Array(gcols)
    diff = count(!=(0), gpu_cols .- ref_cols)
    ρ_gpu = normalised(run_sorted)
    @printf("\ncolonnes différentes : %d sur %d (%.3f %%)\n", diff, 3NPART, 100diff / (3NPART))
    @printf("écart sur la densité : %.1e\n", norm(ρ_gpu - ρref) / norm(ρref))
end

main()
