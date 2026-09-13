"""
Backend Metal : évaluation du champ lissé sur GPU Apple.

Chargée automatiquement dès que `Metal` l'est. Voir `src/gpu.jl` pour le
contrat, et `docs/gpu.md` pour ce que la mesure en dit.
"""
module VlasovMetalExt

using Vlasov
using Metal

import Vlasov: ForceAccelerator, forces!, deposit_smoothed!, smoothed_field,
               spline_field, nearest_knot, table_column, GaussianSmoothing,
               SplineAxis, SplineMesh, ParticleCloud, CellSort, cellsort!,
               total_charge

"""
Tables et tampons résidents sur le GPU.

`csol` est réécrit à chaque pas — le potentiel change —, les tables ne le sont
jamais. La grille fine étant **uniforme**, l'indice du nœud le plus proche se
calcule au lieu de se chercher : le noyau n'a ni recherche dichotomique ni
branche, ce qui est exactement ce qu'un GPU demande.
"""
struct MetalForceAccelerator <: ForceAccelerator
    csol::MtlArray{Float32,3}
    overlap::MtlArray{Float32,2}
    gradient::MtlArray{Float32,2}
    "Table du dépôt : la gaussienne aux huit points de collocation voisins."
    nodes::MtlArray{Float32,2}
    pos::MtlArray{Float32,2}
    force::MtlArray{Float32,2}
    "Tampons hôtes réutilisés : convertir `csol` ou empaqueter les positions
     à chaque pas allouerait plusieurs mégaoctets par pas, et un ramasse-miettes
     à l'arrivée."
    hostcsol::Array{Float32,3}
    hostpos::Matrix{Float32}
    hostforce::Matrix{Float32}
    """Tri par maille et ce qu'il produit : colonnes de table dans l'ordre
    trié, mailles occupées, bornes. Les colonnes restent calculées sur **hôte**,
    en `Float64` : les former en `Float32` ferait basculer une colonne sur deux
    mille (l'ULP à 78 a₀ vaut 0,22 % de leur largeur) et porterait l'écart de
    densité de 1,5e-07 à 9,0e-05."""
    sorter::CellSort{Float64}
    hostcols::Matrix{Int32}
    cols::MtlArray{Int32,2}
    """⚠️ Redimensionnés en cours de route : le nombre de mailles occupées
    change à mesure que l'agrégat évolue. D'où les `Ref`, et `_ensure!`."""
    cells::Base.RefValue{MtlVector{Int32,Metal.PrivateStorage}}
    bounds::Base.RefValue{MtlVector{Int32,Metal.PrivateStorage}}
    rho::MtlArray{Float32,3}
    hostrho::Array{Float32,3}
    x0::Float32                   # premier nœud de la grille fine
    h::Float32                    # pas (grille uniforme)
    nknots::Int32
    spacing::Float32
    nbdt::Int32
    ncol::Int32
    npart::Int
end

"""Vérifie qu'un axe est bien uniforme — le noyau en dépend."""
function _uniform_step(ax::SplineAxis)
    k = ax.knots
    h = (k[end] - k[1]) / (length(k) - 1)
    maximum(abs, diff(k) .- h) <= 1e-9 * abs(h) ||
        throw(ArgumentError("le backend Metal suppose une grille fine uniforme"))
    h
end

function Vlasov.ForceAccelerator(::Type{MtlArray}, fine::NTuple{3,SplineAxis{T}},
                                 sm::GaussianSmoothing{T}, npart::Integer,
                                 n::Integer) where {T}
    h = _uniform_step(fine[1])
    for d in 2:3
        isapprox(_uniform_step(fine[d]), h; rtol = 1e-12) ||
            throw(ArgumentError("le backend Metal suppose les trois axes identiques"))
    end
    MetalForceAccelerator(
        MtlArray(zeros(Float32, n, n, n)),
        MtlArray(Float32.(sm.overlap)), MtlArray(Float32.(sm.gradient)),
        MtlArray(Float32.(sm.nodes)),
        MtlArray(zeros(Float32, 3, npart)), MtlArray(zeros(Float32, 3, npart)),
        Array{Float32,3}(undef, n, n, n),
        Matrix{Float32}(undef, 3, npart), Matrix{Float32}(undef, 3, npart),
        CellSort(fine[1], npart), Matrix{Int32}(undef, 3, npart),
        MtlArray(zeros(Int32, 3, npart)), Ref(MtlArray(zeros(Int32, 1))),
        Ref(MtlArray(zeros(Int32, 1))), MtlArray(zeros(Float32, n, n, n)),
        Array{Float32,3}(undef, n, n, n),
        Float32(fine[1].knots[1]), Float32(h), Int32(length(fine[1].knots)),
        Float32(sm.spacing), Int32(sm.nbdt), Int32(size(sm.overlap, 2)), npart)
end

"""Empaquette `Vector{NTuple{3,T}}` en une matrice `3×N` de `Float32`.

Séparée dans sa propre fonction pour que la boucle soit typée : écrite en
place dans `forces!`, elle capturerait des variables dont le type n'est connu
qu'à l'exécution."""
function _pack!(dest::Matrix{Float32}, src::Vector{NTuple{3,T}}) where {T}
    @inbounds for i in eachindex(src)
        p = src[i]
        dest[1, i] = p[1]; dest[2, i] = p[2]; dest[3, i] = p[3]
    end
    dest
end

"""
Noyau : une particule par fil, contraction 10×10×10 contre `csol`.

Environ 3000 opérations pour 4 Ko lus : le noyau est **borné par la mémoire**,
pas par le calcul. Deux particules d'une même maille lisent le même pavé, d'où
l'intérêt (ici non exploité) de trier les particules — le tri de la thèse,
pour la même raison qu'en 1997.

Les particules dont le pochoir déborde de la grille écrivent un `NaN` : elles
sont reprises par le CPU. Signaler vaut mieux que tronquer en silence.
"""
function _field_kernel!(force, csol, ovl, grad, pos, x0, h, nknots,
                        spacing, nbdt, w, nc, npart)
    i = thread_position_in_grid_1d()
    i > npart && return nothing

    @inbounds begin
        px = pos[1, i]; py = pos[2, i]; pz = pos[3, i]

        # Nœud le plus proche, calculé (grille uniforme) et non cherché.
        kx = min(max(round(Int32, (px - x0) / h) + Int32(1), Int32(1)), nknots)
        ky = min(max(round(Int32, (py - x0) / h) + Int32(1), Int32(1)), nknots)
        kz = min(max(round(Int32, (pz - x0) / h) + Int32(1), Int32(1)), nknots)

        bx = Int32(2) * kx - Int32(5)
        by = Int32(2) * ky - Int32(5)
        bz = Int32(2) * kz - Int32(5)

        nn = Int32(size(csol, 1))
        if bx < Int32(1) || by < Int32(1) || bz < Int32(1) ||
           bx + Int32(9) > nn || by + Int32(9) > nn || bz + Int32(9) > nn
            force[1, i] = NaN32; force[2, i] = NaN32; force[3, i] = NaN32
            return nothing
        end

        half = spacing * 0.5f0
        cx = min(max(floor(Int32, (px - (x0 + (kx - Int32(1)) * h) + half) /
                          spacing * nbdt + 0.5f0) + Int32(1), Int32(1)), nc)
        cy = min(max(floor(Int32, (py - (x0 + (ky - Int32(1)) * h) + half) /
                          spacing * nbdt + 0.5f0) + Int32(1), Int32(1)), nc)
        cz = min(max(floor(Int32, (pz - (x0 + (kz - Int32(1)) * h) + half) /
                          spacing * nbdt + 0.5f0) + Int32(1), Int32(1)), nc)

        fx = 0.0f0; fy = 0.0f0; fz = 0.0f0
        for kk in Int32(1):Int32(10)
            k = bz + kk - Int32(1)
            oz = ovl[kk, cz]; gz = grad[kk, cz]
            for jj in Int32(1):Int32(10)
                j = by + jj - Int32(1)
                oy = ovl[jj, cy]; gy = grad[jj, cy]
                dxp = 0.0f0; val = 0.0f0
                for ii in Int32(1):Int32(10)
                    c = csol[bx + ii - Int32(1), j, k]
                    dxp = fma(c, grad[ii, cx], dxp)
                    val = fma(c, ovl[ii, cx], val)
                end
                fx = fma(oy * oz, dxp, fx)
                fy = fma(gy * oz, val, fy)
                fz = fma(oy * gz, val, fz)
            end
        end
        force[1, i] = -w * fx
        force[2, i] = -w * fy
        force[3, i] = -w * fz
    end
    nothing
end


"""
Noyau de dépôt, version **triée**.

Un groupe de 512 fils par maille occupée, et chaque fil possède **un** des 512
points du pochoir 8³. Il parcourt toutes les particules de la maille en
accumulant dans un registre, et ne fait qu'**une seule** addition atomique à la
fin.

C'est le renversement de boucle qui paie : la voie naïve fait 512 atomiques par
particule, celle-ci en fait une par point de pochoir et par maille — cent fois
moins, puisque chaque maille occupée contient une centaine de particules.

Les colonnes passent par la mémoire du groupe, chargées par tranches de 64 :
sans cela les 512 fils reliraient chacun les données de chaque particule.
"""
function _deposit_kernel!(ρ, nodes, cols, cellids, bounds, nk, ncell)
    g = threadgroup_position_in_grid_1d()
    g > ncell && return nothing
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
        lo = bounds[g]; hi = bounds[g + Int32(1)]
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

"""Agrandit les tampons de mailles si le tri en a trouvé davantage. On ne
rétrécit jamais : la taille se stabilise en quelques pas."""
function _ensure!(acc::MetalForceAccelerator, ncell::Integer)
    if length(acc.cells[]) < ncell
        acc.cells[] = MtlArray(zeros(Int32, ncell))
    end
    if length(acc.bounds[]) < ncell + 1
        acc.bounds[] = MtlArray(zeros(Int32, ncell + 1))
    end
    nothing
end

"""Colonnes de table dans l'ordre trié, et particules hors domaine.

Calculées en `Float64` sur l'hôte, délibérément : voir le champ `hostcols`.
Les particules hors du domaine utile reçoivent la colonne 1 et sont exclues du
dépôt par un poids nul — plus simple qu'une liste à part, et sans branche dans
le noyau."""
function _fill_columns!(acc::MetalForceAccelerator, mesh, sm, positions)
    knots = mesh.axes[1].knots
    x0 = acc.sorter.x0; h = acc.sorter.h; nk = acc.sorter.nknots
    half = sm.spacing / 2
    lo = knots[2] + half; hi = knots[end-1] - half
    sp = sm.spacing; nbdt = sm.nbdt; ncol = size(sm.nodes, 2)
    perm = acc.sorter.perm
    cols = acc.hostcols
    nout = Threads.Atomic{Int}(0)

    # ⚠️ Indice de nœud **calculé**, pas cherché : la grille est uniforme, et
    # une dichotomie par particule et par direction coûtait quinze millisecondes
    # de plus que tout le reste du dépôt réuni.
    Threads.@threads for s in eachindex(perm)
        @inbounds begin
            p = positions[perm[s]]
            if lo <= p[1] <= hi && lo <= p[2] <= hi && lo <= p[3] <= hi
                for d in 1:3
                    u = p[d]
                    k = clamp(round(Int, (u - x0) / h) + 1, 1, nk)
                    # En `Float64`, délibérément : voir le champ `hostcols`.
                    δ = u - (x0 + (k - 1) * h)
                    cols[d, s] = clamp(floor(Int32, (δ + half) / sp * nbdt + 0.5) + Int32(1),
                                       Int32(1), Int32(ncol))
                end
            else
                cols[1, s] = cols[2, s] = cols[3, s] = Int32(1)
                Threads.atomic_add!(nout, 1)
            end
        end
    end
    nout[]
end

function Vlasov.deposit_smoothed!(ρ::Array{T,3}, acc::MetalForceAccelerator,
                                  mesh::SplineMesh{3,T}, sm::GaussianSmoothing{T},
                                  positions; charge::T) where {T}
    cellsort!(acc.sorter, positions)
    nout = _fill_columns!(acc, mesh, sm, positions)

    ncell = length(acc.sorter.occupied)
    _ensure!(acc, ncell)
    copyto!(acc.cols, acc.hostcols)
    copyto!(acc.cells[], 1:ncell, acc.sorter.occupied, 1:ncell)
    copyto!(acc.bounds[], 1:(ncell + 1), acc.sorter.bounds, 1:(ncell + 1))
    fill!(acc.rho, 0f0)

    Metal.@sync @metal threads=512 groups=ncell _deposit_kernel!(
        acc.rho, acc.nodes, acc.cols, acc.cells[], acc.bounds[],
        Int32(acc.sorter.nknots), Int32(ncell))

    copyto!(acc.hostrho, acc.rho)
    ρ .= acc.hostrho
    ρ .*= charge
    ρ .*= (length(positions) - nout) * charge / total_charge(ρ, mesh)
    nout
end

function Vlasov.forces!(cloud::ParticleCloud{T}, acc::MetalForceAccelerator,
                        fine::NTuple{3,SplineAxis{T}}, csol_fine::Array{T,3},
                        coarse::NTuple{3,SplineAxis{T}}, csol_coarse::Array{T,3},
                        sm::GaussianSmoothing{T}; escaped::Integer = 0) where {T}
    npart = length(cloud.positions)
    npart == acc.npart || throw(DimensionMismatch("accélérateur dimensionné pour $(acc.npart)"))
    w = Float32(cloud.weight)

    acc.hostcsol .= csol_fine
    copyto!(acc.csol, acc.hostcsol)
    _pack!(acc.hostpos, cloud.positions)
    copyto!(acc.pos, acc.hostpos)

    groupsize = 256
    Metal.@sync @metal threads = groupsize groups = cld(npart, groupsize) _field_kernel!(
        acc.force, acc.csol, acc.overlap, acc.gradient, acc.pos,
        acc.x0, acc.h, acc.nknots, acc.spacing, acc.nbdt, w, acc.ncol, Int32(npart))

    copyto!(acc.hostforce, acc.force)

    # Reprise CPU de ce que le GPU a refusé — les bords, et eux seuls.
    n = 0
    ww = T(cloud.weight); w2 = ww * ww
    @inbounds for i in 1:npart
        if isnan(acc.hostforce[1, i])
            n += 1
            p = cloud.positions[i]
            E = spline_field(coarse, csol_coarse, p)
            cloud.forces[i] = if E === nothing
                r3 = (p[1]^2 + p[2]^2 + p[3]^2)^T(1.5)
                (-w2 * escaped / r3) .* p
            else
                ww .* E
            end
        else
            cloud.forces[i] = (T(acc.hostforce[1, i]), T(acc.hostforce[2, i]),
                               T(acc.hostforce[3, i]))
        end
    end
    n
end

end # module
