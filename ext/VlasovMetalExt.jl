"""
Backend Metal : évaluation du champ lissé sur GPU Apple.

Chargée automatiquement dès que `Metal` l'est. Voir `src/gpu.jl` pour le
contrat, et `docs/gpu.md` pour ce que la mesure en dit.
"""
module VlasovMetalExt

using Vlasov
using Metal

import Vlasov: ForceAccelerator, forces!, smoothed_field, spline_field,
               nearest_knot, GaussianSmoothing, SplineAxis, ParticleCloud

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
    pos::MtlArray{Float32,2}
    force::MtlArray{Float32,2}
    "Tampons hôtes réutilisés : convertir `csol` ou empaqueter les positions
     à chaque pas allouerait plusieurs mégaoctets par pas, et un ramasse-miettes
     à l'arrivée."
    hostcsol::Array{Float32,3}
    hostpos::Matrix{Float32}
    hostforce::Matrix{Float32}
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
        MtlArray(zeros(Float32, 3, npart)), MtlArray(zeros(Float32, 3, npart)),
        Array{Float32,3}(undef, n, n, n),
        Matrix{Float32}(undef, 3, npart), Matrix{Float32}(undef, 3, npart),
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
