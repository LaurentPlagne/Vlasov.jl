"""
Deposition of the pseudo-particles onto the grid, and the values ↔ coefficients
conversion.

Two representations of the same field coexist in this method, and confusing them
is *the* classic bug:

  * **values at the collocation points** — what deposition produces;
  * **spline coefficients** — what the moments integrate against.

One goes from one to the other by applying `S⁻¹` in each direction.
"""

"""
    dual_lengths(ax) -> Vector

Length of the dual cell of each collocation point: the share of the domain it
"owns". Used to normalise a deposition into a density.

The four outermost points are treated separately — their cell is bounded by the
edge of the domain, not by a neighbouring collocation point.
"""
function dual_lengths(ax::SplineAxis)
    gt, g = ax.colloc, ax.knots
    m = length(gt)
    map(1:m) do j
        j == 1     ? (gt[2] - g[1]) / 2 :
        j == 2     ? (gt[3] - g[1]) / 2 :
        j == m     ? (g[end] - gt[m-1]) / 2 :
        j == m - 1 ? (g[end] - gt[m-2]) / 2 :
                     (gt[j+1] - gt[j-1]) / 2
    end
end

"""
    locate(ax, x) -> (cell, weight) or `nothing`

Locates `x` between two consecutive collocation points: returns the index of the
left point and the weight that falls to it (the right one receives
`1 - weight`). Returns `nothing` outside the domain.
"""
@inline function locate(ax::SplineAxis{T}, x) where {T}
    gt = ax.colloc
    (x < gt[1] || x > gt[end]) && return nothing
    c = max(1, searchsortedfirst(gt, x) - 1)
    (c, (gt[c+1] - x) / (gt[c+1] - gt[c]))
end

"""
    scatter!(kernel, ρ, mesh, positions, buffers) -> nout

Engine shared by both depositions. `kernel(dest, ax, ay, az, p)` deposits one
particle and returns `false` if it lies outside the domain.

Without `buffers` the loop is sequential. With them, each thread accumulates
into its own array and the arrays are then summed: the only way to parallelise a
scatter without losing contributions or paying for atomics.
"""
function scatter!(kernel, ρ::Array{T,3}, mesh::SplineMesh{3,T},
                  positions, buffers) where {T}
    mx, my, mz = mesh.axes
    size(ρ) == (nbasis(mx), nbasis(my), nbasis(mz)) ||
        throw(DimensionMismatch("ρ must cover the whole collocation grid"))

    if buffers === nothing
        fill!(ρ, zero(T))
        nout = 0
        for p in positions
            kernel(ρ, mx, my, mz, p) || (nout += 1)
        end
        return nout
    end

    parts = chunks(length(positions), min(length(buffers.slots), Threads.nthreads()))
    nouts = zeros(Int, length(parts))
    Threads.@threads for c in eachindex(parts)
        dest = buffers.slots[c]
        fill!(dest, zero(T))
        n = 0
        @inbounds for i in parts[c]
            kernel(dest, mx, my, mz, positions[i]) || (n += 1)
        end
        nouts[c] = n
    end
    scatter_reduce!(ρ, buffers, length(parts))
    sum(nouts)
end

"""
    ScatterBuffers(mesh)

One density array **per thread**, to parallelise the deposition.

Deposition is a *scatter*: several particles write into the same slot, and
naively splitting the loop would lose contributions. Each thread therefore
accumulates into its own array, and they are summed at the end — the sum costs
`nthreads × N³` additions, negligible next to the deposition itself.

⚠️ **The memory cost grows as the cube of the grid.** At 58³ and eight threads
that is 12 MB; at 128³ it would be 134 MB. The buffers are therefore created
explicitly by the caller, never behind its back.

Known avenue for going further once the particles have lost their order: **sort
them by cell** before depositing, so that neighbouring particles write into
neighbouring slots. That is what the thesis code did, in parallel (the PSRS
method). Now available as [`CellSort`](@ref).
"""
struct ScatterBuffers{T,N}
    slots::Vector{Array{T,N}}
end

function ScatterBuffers(mesh::SplineMesh{N,T}; nslots = Threads.nthreads()) where {N,T}
    dims = map(nbasis, mesh.axes)
    ScatterBuffers{T,N}([Array{T,N}(undef, dims) for _ in 1:nslots])
end

"""
    scatter_reduce!(ρ, buffers, nused) -> ρ

Sums the first `nused` buffers into `ρ`.
"""
function scatter_reduce!(ρ::Array{T,N}, b::ScatterBuffers{T,N}, nused::Integer) where {T,N}
    copyto!(ρ, b.slots[1])
    for c in 2:nused
        ρ .+= b.slots[c]
    end
    ρ
end

"""
    deposit!(ρ, mesh, positions; charge) -> nout

Deposits pseudo-particles of weight `charge` onto the mesh's collocation points
by trilinear interpolation ("cloud-in-cell"), and normalises by the dual volume
of each node to obtain a density.

`ρ` has the **full** size of the collocation grid (boundaries included), not
that of the interior problem. Returns the number of particles that fell outside
the domain, which are ignored.
"""
function deposit!(ρ::Array{T,3}, mesh::SplineMesh{3,T},
                  positions; charge::T, buffers = nothing) where {T}
    mx, my, mz = mesh.axes
    size(ρ) == (nbasis(mx), nbasis(my), nbasis(mz)) ||
        throw(DimensionMismatch("ρ must cover the whole collocation grid"))
    tx, ty, tz = mesh.locators
    nout = scatter!(ρ, mesh, positions, buffers) do dest, mx, my, mz, p
        lx = locate(tx, mx, p[1])
        ly = locate(ty, my, p[2])
        lz = locate(tz, mz, p[3])
        (lx === nothing || ly === nothing || lz === nothing) && return false
        (i, ax), (j, ay), (k, az) = lx, ly, lz
        bx, by, bz = 1 - ax, 1 - ay, 1 - az
        @inbounds begin
            dest[i, j, k]       += ax * ay * az
            dest[i, j, k+1]     += ax * ay * bz
            dest[i, j+1, k]     += ax * by * az
            dest[i, j+1, k+1]   += ax * by * bz
            dest[i+1, j, k]     += bx * ay * az
            dest[i+1, j, k+1]   += bx * ay * bz
            dest[i+1, j+1, k]   += bx * by * az
            dest[i+1, j+1, k+1] += bx * by * bz
        end
        true
    end

    # Normalisation into a density. The dual volume is the outer product of the
    # three dual lengths: the broadcast walks it without ever materialising it
    # in 3D, where the Fortran kept 58³ floats for it (`volm1`).
    #
    # ⚠️ These three names must NOT coincide with those of the `do` block above.
    # A variable assigned both inside a closure AND in the enclosing function is
    # one single variable, which Julia boxes: all eight threads would then write
    # into the same slot. The symptom was a non-deterministic deposition,
    # correct on one thread and wrong on eight.
    wx, wy, wz = map(dual_lengths, (mx, my, mz))
    ρ .*= charge ./ (wx .* wy' .* reshape(wz, 1, 1, :))
    nout
end

"""
    spline_coefficients!(c, ρ, mesh)

Converts values at the collocation points into spline coefficients by applying
`S⁻¹` in each direction (the Fortran's `tensrus2`).
"""
function spline_coefficients!(c::AbstractArray{T,N}, ρ::AbstractArray{T,N},
                              mesh::SplineMesh{N,T}) where {T,N}
    # Same kernel as the tensor solver: `N` rotations, `N` matrix-matrix
    # products, no slicing. The buffer comes from the mesh rather than from a
    # 1.4 MB allocation on every call.
    apply_all_rotating!(c, map(cm -> cm.Sinv, mesh.collocation), ρ, mesh.scratch[3])
end

"""Allocating version of [`spline_coefficients!`](@ref)."""
spline_coefficients(ρ::AbstractArray{T,N}, mesh::SplineMesh{N,T}) where {T,N} =
    spline_coefficients!(similar(ρ), ρ, mesh)

"""
    total_charge(ρ, mesh) -> T

Total charge `∫ρ dV`, obtained by contracting the spline coefficients with the
order-0 moments of each direction.

This is the deposition's conservation check: depositing `N` electrons must
return `N`, up to the accuracy of the interpolation.
"""
function total_charge(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T}
    # As for `multipole`: the dual moments carry `S⁻ᵀ`, so the density
    # contracts as it stands and its coefficients need never be formed.
    px, py, pz = map(m -> m[1], mesh.dual_moments)
    s = zero(T)
    @inbounds for k in eachindex(pz), j in eachindex(py), i in eachindex(px)
        s += ρ[i, j, k] * px[i] * py[j] * pz[k]
    end
    s
end
