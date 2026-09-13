"""
Rangement des particules par maille.

Le dépôt de charge est un *scatter* : chaque particule écrit dans 8³ points de
la grille, et les particules voisines écrivent aux mêmes endroits. Les ranger
par maille sert deux choses à la fois — la **localité** sur CPU, et sur GPU la
possibilité de traiter une maille par groupe de fils, ce qui divise les
additions atomiques par le nombre de particules qu'elle contient.

C'est le tri de la thèse, pour la raison de 1997. Elle employait PSRS, dont
l'intérêt est de minimiser la **redistribution entre processeurs** : sur le
T3E, trier est un problème de communication. En mémoire partagée il n'y a pas
d'échange à équilibrer, et un tri par comptage suffit — `O(N)`, insensible à
l'ordre de départ. Voir `docs/gpu.md`.
"""

"""
    CellSort(axis, npart, nthreads = Threads.nthreads())

Tampons d'un tri par comptage sur la maille de la grille fine.

Tout est alloué une fois : à 800 000 particules et 91 125 mailles, allouer à
chaque pas coûterait plus que le tri lui-même.

⚠️ Suppose la grille **uniforme** — l'indice de maille se calcule alors au lieu
de se chercher.
"""
struct CellSort{T<:AbstractFloat}
    x0::T
    h::T
    nknots::Int
    "Maille de chaque particule, dans l'ordre courant."
    keys::Vector{Int32}
    "Compteurs par tranche, puis leur somme."
    partial::Vector{Vector{Int32}}
    total::Vector{Int32}
    "Où chaque tranche écrit, pour chaque maille."
    offsets::Vector{Vector{Int32}}
    "Permutation : `perm[s]` est l'indice courant de la `s`-ième particule triée."
    perm::Vector{Int32}
    "Mailles contenant au moins une particule, et bornes `[bounds[g]+1, bounds[g+1]]`."
    occupied::Vector{Int32}
    bounds::Vector{Int32}
    chunks::Vector{UnitRange{Int}}
end

function CellSort(axis::SplineAxis{T}, npart::Integer,
                  nthreads::Integer = Threads.nthreads()) where {T}
    k = axis.knots
    h = (k[end] - k[1]) / (length(k) - 1)
    maximum(abs, diff(k) .- h) <= 1e-9 * abs(h) ||
        throw(ArgumentError("`CellSort` suppose une grille uniforme"))
    nk = length(k)
    ncell = nk^3
    CellSort{T}(k[1], h, nk,
                Vector{Int32}(undef, npart),
                [Vector{Int32}(undef, ncell) for _ in 1:nthreads],
                Vector{Int32}(undef, ncell),
                [Vector{Int32}(undef, ncell) for _ in 1:nthreads],
                Vector{Int32}(undef, npart),
                Int32[], Int32[], chunks(npart, nthreads))
end

"""Indice linéaire de la maille d'un point — le nœud le plus proche, comme le
dépôt. Calculé, la grille étant uniforme."""
@inline function cell_of(p, x0, h, nk)
    i(u) = clamp(round(Int32, (u - x0) / h) + Int32(1), Int32(1), Int32(nk))
    i(p[1]) + Int32(nk) * (i(p[2]) - Int32(1) + Int32(nk) * (i(p[3]) - Int32(1)))
end

"""
    cellsort!(cs, positions) -> cs

Range les particules par maille. Remplit `perm`, `occupied` et `bounds`.

Quatre passes : maille de chaque particule, fusion des compteurs, décalages,
placement. Les deux premières et la dernière sont parallèles ; la troisième ne
visite que les mailles **occupées**, qui sont dix fois moins nombreuses que les
autres — l'agrégat n'occupe qu'une fraction de la boîte.
"""
function cellsort!(cs::CellSort, positions)
    length(positions) == length(cs.perm) ||
        throw(DimensionMismatch("tri dimensionné pour $(length(cs.perm)) particules"))
    x0, h, nk = cs.x0, cs.h, cs.nknots

    # 1. Maille de chaque particule, et comptage par tranche.
    # ⚠️ Remettre les compteurs à zéro : sans cela deux appels successifs les
    # cumulent, les décalages deviennent faux et le placement écrit hors bornes.
    Threads.@threads for t in eachindex(cs.chunks)
        cnt = cs.partial[t]
        fill!(cnt, Int32(0))
        @inbounds for i in cs.chunks[t]
            c = cell_of(positions[i], x0, h, nk)
            cs.keys[i] = c
            cnt[c] += Int32(1)
        end
    end

    # 2. Fusion. ⚠️ Dans `total`, jamais dans `partial[1]` : les décalages ont
    # besoin des compteurs de chaque tranche, y compris la première.
    copyto!(cs.total, cs.partial[1])
    for t in 2:length(cs.partial)
        cs.total .+= cs.partial[t]
    end

    # 3. Mailles occupées, puis décalages — restreints à celles-là.
    empty!(cs.occupied); empty!(cs.bounds); push!(cs.bounds, Int32(0))
    acc = Int32(0)
    @inbounds for c in eachindex(cs.total)
        cs.total[c] == 0 && continue
        push!(cs.occupied, Int32(c))
        a = acc
        for t in eachindex(cs.partial)
            cs.offsets[t][c] = a
            a += cs.partial[t][c]
        end
        acc += cs.total[c]
        push!(cs.bounds, acc)
    end

    # 4. Placement. Chaque tranche écrit dans sa propre zone de chaque maille,
    # donc sans synchronisation.
    Threads.@threads for t in eachindex(cs.chunks)
        cur = cs.offsets[t]
        @inbounds for i in cs.chunks[t]
            c = cs.keys[i]
            cur[c] += Int32(1)
            cs.perm[cur[c]] = Int32(i)
        end
    end
    cs
end

"""Nombre de mailles occupées — dix fois moins que de mailles, en pratique."""
noccupied(cs::CellSort) = length(cs.occupied)
