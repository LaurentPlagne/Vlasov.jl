"""
Backend Metal : évaluation du champ lissé sur GPU Apple.

Chargée automatiquement dès que `Metal` l'est. Voir `src/gpu.jl` pour le
contrat, et `docs/gpu.md` pour ce que la mesure en dit.
"""
module VlasovMetalExt

using Vlasov
using Metal

import Vlasov: ForceAccelerator, forces!, deposit_smoothed!, projectile_forces!,
               smoothed_field, Projectile, GaussianSoftening, Jellium, erf,
               spline_field, nearest_knot, table_column, GaussianSmoothing,
               SplineAxis, SplineMesh, ParticleCloud, CellSort, cellsort!,
               total_charge, uniform_sphere_potential

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
    force::MtlArray{Float32,2}
    "Tampons hôtes réutilisés : convertir `csol` ou empaqueter les positions
     à chaque pas allouerait plusieurs mégaoctets par pas, et un ramasse-miettes
     à l'arrivée."
    hostcsol::Array{Float32,3}
    hostforce::Matrix{Float32}
    """Positions sous la forme `(k, δ)` — voir [`_pack_kd!`](@ref). C'est cette
    représentation, et non la position absolue, qui permet au GPU de choisir la
    bonne colonne de table : `δ` est majoré par un demi-pas, donc codé en
    `Float32` avec trente mille fois la finesse d'une colonne."""
    hostknode::Matrix{Int32}
    hostdelta::Matrix{Float32}
    knode::MtlArray{Int32,2}
    delta::MtlArray{Float32,2}
    "Tri par maille : permutation, mailles occupées, bornes."
    sorter::CellSort{Float64}
    hostcols::Matrix{Int32}
    cols::MtlArray{Int32,2}
    """⚠️ Redimensionnés en cours de route : le nombre de mailles occupées
    change à mesure que l'agrégat évolue. D'où les `Ref`, et `_ensure!`."""
    cells::Base.RefValue{MtlVector{Int32,Metal.PrivateStorage}}
    bounds::Base.RefValue{MtlVector{Int32,Metal.PrivateStorage}}
    rho::MtlArray{Float32,3}
    hostrho::Array{Float32,3}
    hostreduction::Vector{Float32}
    """Réduction du projectile : les trois composantes de la force qu'il subit,
    puis l'énergie d'interaction avec les pseudo-électrons. Fusionnée dans le
    noyau des forces, elle ne coûte que quelques opérations sur des données déjà
    chargées — un second passage sur 800 000 particules en coûterait dix fois
    plus."""
    reduction::MtlArray{Float32,1}
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
        MtlArray(zeros(Float32, 3, npart)),
        Array{Float32,3}(undef, n, n, n), Matrix{Float32}(undef, 3, npart),
        Matrix{Int32}(undef, 3, npart), Matrix{Float32}(undef, 3, npart),
        MtlArray(zeros(Int32, 3, npart)), MtlArray(zeros(Float32, 3, npart)),
        CellSort(fine[1], npart), Matrix{Int32}(undef, 3, npart),
        MtlArray(zeros(Int32, 3, npart)), Ref(MtlArray(zeros(Int32, 1))),
        Ref(MtlArray(zeros(Int32, 1))), MtlArray(zeros(Float32, n, n, n)),
        Array{Float32,3}(undef, n, n, n),
        Vector{Float32}(undef, 4), MtlArray(zeros(Float32, 4)),
        Float32(fine[1].knots[1]), Float32(h), Int32(length(fine[1].knots)),
        Float32(sm.spacing), Int32(sm.nbdt), Int32(size(sm.overlap, 2)), npart)
end

"""Empaquette les positions sous la forme `(k, δ)` : indice du nœud le plus
proche, et **écart à ce nœud**.

⚠️ C'est le point qui décide de la justesse du portage. Une position vaut
jusqu'à 78 a₀, où l'ULP de `Float32` est 7,6e-06 — soit 0,22 % de la largeur
d'une colonne de table (0,00355 a₀). Former `x − knot` sur le GPU fait donc
basculer une particule sur cinq cents sur la colonne voisine, ce qui n'est pas
un arrondi qui se moyenne mais un **échantillon de gaussienne faux**.

`δ` est majoré par un demi-pas, 1,8 a₀ : codé en `Float32`, sa résolution est
1,2e-07, trente mille fois plus fine qu'une colonne. La soustraction se fait
ici, en `Float64`, et une seule fois — les deux noyaux s'en servent.

La position absolue se reconstruit au besoin par `x₀ + (k−1)h + δ`, ce qui ne
perd rien de plus que ne perdait l'ancien empaquetage.

Séparée dans sa propre fonction pour que la boucle soit typée : écrite en place
dans `forces!`, elle capturerait des variables de type inconnu à la compilation.
"""
function _pack_kd!(knode::Matrix{Int32}, delta::Matrix{Float32},
                   src::Vector{NTuple{3,T}}, x0::T, h::T, nk::Int) where {T}
    Threads.@threads for i in eachindex(src)
        @inbounds begin
            p = src[i]
            for d in 1:3
                k = clamp(round(Int32, (p[d] - x0) / h) + Int32(1), Int32(1), Int32(nk))
                knode[d, i] = k
                delta[d, i] = Float32(p[d] - (x0 + (k - 1) * h))
            end
        end
    end
    nothing
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
function _field_kernel!(force, csol, ovl, grad, knode, delta, x0, h,
                        spacing, nbdt, w, nc, npart,
                        px0, py0, pz0, coef, σ, red)
    i = thread_position_in_grid_1d()
    tid = thread_index_in_threadgroup()
    nthr = Int32(256)

    # ⚠️ Tampon de réduction du groupe. Tous les fils doivent atteindre chaque
    # barrière : pas de `return` anticipé dans ce noyau, seulement des drapeaux.
    sh = MtlThreadGroupArray(Float32, 4 * 256)
    @inbounds for c in Int32(0):Int32(3)
        sh[c * nthr + tid] = 0.0f0
    end

    @inbounds if i <= npart
        # `(k, δ)` viennent de l'hôte, calculés en `Float64` : le noyau ne forme
        # jamais `x − knot`, ce qui serait sa plus grosse perte de précision.
        kx = knode[1, i]; ky = knode[2, i]; kz = knode[3, i]
        dx0 = delta[1, i]; dy0 = delta[2, i]; dz0 = delta[3, i]
        bx = Int32(2) * kx - Int32(5)
        by = Int32(2) * ky - Int32(5)
        bz = Int32(2) * kz - Int32(5)

        nn = Int32(size(csol, 1))
        ok = bx >= Int32(1) && by >= Int32(1) && bz >= Int32(1) &&
             bx + Int32(9) <= nn && by + Int32(9) <= nn && bz + Int32(9) <= nn

        fxp = NaN32; fyp = NaN32; fzp = NaN32
        if ok
            # La colonne ne dépend que de `δ`, donc elle est **exacte** ici :
            # plus aucune grande soustraction.
            half = spacing * 0.5f0
            cx = min(max(floor(Int32, (dx0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)
            cy = min(max(floor(Int32, (dy0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)
            cz = min(max(floor(Int32, (dz0 + half) / spacing * nbdt + 0.5f0) +
                         Int32(1), Int32(1)), nc)

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
            fxp = -w * fx; fyp = -w * fy; fzp = -w * fz
        end

        # --- projectile ↔ pseudo-électron, fusionné -------------------------
        # Calculée pour **toutes** les particules, y compris celles que le CPU
        # reprendra : la réduction doit les compter. `coef = −poids·charge`.
        if coef != 0.0f0
            # Position absolue reconstruite : `x₀ + (k−1)h + δ`. Elle porte la
            # même précision que l'ancien empaquetage direct, et ne sert qu'ici
            # — les colonnes, elles, n'en dépendent plus.
            px = x0 + Float32(kx - Int32(1)) * h + dx0
            py = x0 + Float32(ky - Int32(1)) * h + dy0
            pz = x0 + Float32(kz - Int32(1)) * h + dz0
            dx = px0 - px; dy = py0 - py; dz = pz0 - pz
            d2 = dx * dx + dy * dy + dz * dz
            u2 = d2 / (σ * σ)
            # Près de zéro, les deux termes de la force gaussienne s'annulent à
            # l'ordre dominant : la série évite la soustraction.
            kf = if u2 <= 0.25f0
                    q = 1.0f0 / 685440.0f0
                    q = -1.0f0 / 49920.0f0 + u2 * q
                    q =  1.0f0 / 4224.0f0  + u2 * q
                    q = -1.0f0 / 432.0f0   + u2 * q
                    q =  1.0f0 / 56.0f0    + u2 * q
                    q = -1.0f0 / 10.0f0    + u2 * q
                    q =  1.0f0 / 3.0f0     + u2 * q
                    0.7978845608f0 * q / (σ * σ * σ)
                else
                    r = sqrt(d2)
                    (erf(r / 1.4142135624f0 / σ) -
                     0.7978845608f0 * (r / σ) * exp(-u2 * 0.5f0)) / (d2 * r)
                end
            m = coef * kf
            fx2 = m * dx; fy2 = m * dy; fz2 = m * dz
            if ok
                fxp -= fx2; fyp -= fy2; fzp -= fz2       # réaction
            end
            r = sqrt(d2)
            sh[tid]            = fx2
            sh[nthr + tid]     = fy2
            sh[2 * nthr + tid] = fz2
            sh[3 * nthr + tid] = r < 1.0f-4 * σ ? 0.7978845608f0 / σ :
                                 erf(r / 1.4142135624f0 / σ) / r
        end

        force[1, i] = fxp; force[2, i] = fyp; force[3, i] = fzp
    end

    # Réduction en arbre, puis une seule atomique par groupe.
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    stride = nthr ÷ Int32(2)
    while stride > Int32(0)
        if tid <= stride
            @inbounds for c in Int32(0):Int32(3)
                sh[c * nthr + tid] += sh[c * nthr + tid + stride]
            end
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        stride ÷= Int32(2)
    end
    if tid == Int32(1)
        @inbounds for c in Int32(0):Int32(3)
            Metal.@atomic red[c + Int32(1)] += sh[c * nthr + Int32(1)]
        end
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
    half = sm.spacing / 2
    lo = knots[2] + half; hi = knots[end-1] - half
    sp = sm.spacing; nbdt = sm.nbdt; ncol = size(sm.nodes, 2)
    perm = acc.sorter.perm
    cols = acc.hostcols
    delta = acc.hostdelta
    nout = Threads.Atomic{Int}(0)

    # `δ` a déjà été calculé en `Float64` par `_pack_kd!` : il ne reste qu'à le
    # relire dans l'ordre trié. La colonne n'en dépend que de lui.
    Threads.@threads for s in eachindex(perm)
        @inbounds begin
            i = perm[s]
            p = positions[i]
            if lo <= p[1] <= hi && lo <= p[2] <= hi && lo <= p[3] <= hi
                for d in 1:3
                    cols[d, s] = clamp(floor(Int32, (Float64(delta[d, i]) + half) /
                                             sp * nbdt + 0.5) + Int32(1),
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
    # Le dépôt ouvre le pas : c'est lui qui empaquette `(k, δ)`, dont `forces!`
    # se resservira.
    _pack_kd!(acc.hostknode, acc.hostdelta, positions,
              acc.sorter.x0, acc.sorter.h, acc.sorter.nknots)
    copyto!(acc.knode, acc.hostknode)
    copyto!(acc.delta, acc.hostdelta)
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
                        sm::GaussianSmoothing{T}; escaped::Integer = 0,
                        projectile = nothing, packed::Bool = false) where {T}
    npart = length(cloud.positions)
    npart == acc.npart || throw(DimensionMismatch("accélérateur dimensionné pour $(acc.npart)"))
    w = Float32(cloud.weight)

    acc.hostcsol .= csol_fine
    copyto!(acc.csol, acc.hostcsol)
    # `packed = true` dit que le dépôt vient de le faire pour les mêmes
    # positions — c'est le cas dans `update_forces!`, où il ouvre le pas.
    # Refaire l'empaquetage coûterait deux millisecondes pour rien.
    if !packed
        _pack_kd!(acc.hostknode, acc.hostdelta, cloud.positions,
                  acc.sorter.x0, acc.sorter.h, acc.sorter.nknots)
        copyto!(acc.knode, acc.hostknode)
        copyto!(acc.delta, acc.hostdelta)
    end

    # Le projectile est fusionné dans ce noyau : ses arguments valent zéro
    # quand il n'y en a pas, et la branche disparaît.
    pp = projectile === nothing ? (0f0, 0f0, 0f0) : Float32.(projectile.position)
    coef = projectile === nothing ? 0f0 : Float32(-cloud.weight * projectile.charge)
    σ = projectile === nothing ? 1f0 : Float32(Vlasov.scale(projectile.softening))
    fill!(acc.reduction, 0f0)

    groupsize = 256
    Metal.@sync @metal threads = groupsize groups = cld(npart, groupsize) _field_kernel!(
        acc.force, acc.csol, acc.overlap, acc.gradient, acc.knode, acc.delta,
        acc.x0, acc.h, acc.spacing, acc.nbdt, w, acc.ncol, Int32(npart),
        pp[1], pp[2], pp[3], coef, σ, acc.reduction)

    copyto!(acc.hostforce, acc.force)
    copyto!(acc.hostreduction, acc.reduction)

    # Reprise CPU de ce que le GPU a refusé — les bords, et eux seuls. ⚠️ Il
    # faut y **réinjecter la réaction du projectile** : le noyau l'a comptée
    # dans la réduction mais n'a pas pu l'ajouter à une force qu'il n'a pas
    # calculée.
    n = 0
    ww = T(cloud.weight); w2 = ww * ww
    @inbounds for i in 1:npart
        if isnan(acc.hostforce[1, i])
            n += 1
            p = cloud.positions[i]
            E = spline_field(coarse, csol_coarse, p)
            f = if E === nothing
                r3 = (p[1]^2 + p[2]^2 + p[3]^2)^T(1.5)
                (-w2 * escaped / r3) .* p
            else
                ww .* E
            end
            if projectile !== nothing
                d = projectile.position .- p
                m = -cloud.weight * projectile.charge *
                    Vlasov.force_kernel(projectile.softening, sum(abs2, d))
                f = f .- m .* d
            end
            cloud.forces[i] = f
        else
            cloud.forces[i] = (T(acc.hostforce[1, i]), T(acc.hostforce[2, i]),
                               T(acc.hostforce[3, i]))
        end
    end
    n
end

"""
    projectile_forces!(cloud, acc, proj, jel) -> (force, e_electrons, e_jellium)

Ne calcule **rien** sur les particules : la somme a déjà été faite par le noyau
des forces, qui la fusionne au lieu d'ouvrir un second passage sur 800 000
particules. Ne reste ici que la part jellium, qui est un scalaire.

⚠️ Suppose donc que [`forces!`](@ref) vient d'être appelée sur le **même**
accélérateur, avec ce projectile.
"""
function Vlasov.projectile_forces!(cloud::ParticleCloud{T}, acc::MetalForceAccelerator,
                                   proj::Projectile{T}, jel::Jellium{T}) where {T}
    p = proj.position
    q = proj.charge
    r2 = sum(abs2, p)
    modf = r2 > jel.radius^2 ? jel.nions * q / r2^T(1.5) : q / Vlasov.WIGNER_SEITZ_NA^3
    fjel = modf .* p
    fel = ntuple(d -> T(acc.hostreduction[d]), 3)
    (fjel .+ fel, T(cloud.weight) * q * T(acc.hostreduction[4]),
     q * Vlasov.potential(jel, sqrt(r2)))
end

end # module
