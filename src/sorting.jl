"""
Ordering the particles by cell.

Charge deposition is a *scatter*: each particle writes into 8³ grid points, and
neighbouring particles write into the same ones. Ordering them by cell serves
two purposes at once — **locality** on the CPU, and on the GPU the possibility of
handling one cell per thread group, which divides the atomic additions by the
number of particles that cell contains.

This is the thesis's sort, for the reason of 1997. It used PSRS, whose point is
to minimise the **redistribution between processors**: on the T3E, sorting is a
communication problem. In shared memory there is no exchange to balance, and a
counting sort suffices — `O(N)`, insensitive to the starting order. See
`docs/gpu.md`.
"""

"""
    CellSort(axis, npart, nthreads = Threads.nthreads())

Buffers for a counting sort on the fine grid's cell index.

Everything is allocated once: at 800 000 particles and 91 125 cells, allocating
at every step would cost more than the sort itself.

⚠️ Assumes a **uniform** grid — the cell index is then computed rather than
searched for.
"""
struct CellSort{T<:AbstractFloat}
    x0::T
    h::T
    nknots::Int
    "Cell of each particle, in the current order."
    keys::Vector{Int32}
    "Per-chunk counters, then their sum."
    partial::Vector{Vector{Int32}}
    total::Vector{Int32}
    "Where each chunk writes, for each cell."
    offsets::Vector{Vector{Int32}}
    "Permutation: `perm[s]` is the current index of the `s`-th sorted particle."
    perm::Vector{Int32}
    "Cells holding at least one particle, and bounds `[bounds[g]+1, bounds[g+1]]`."
    occupied::Vector{Int32}
    bounds::Vector{Int32}
    chunks::Vector{UnitRange{Int}}
end

function CellSort(axis::SplineAxis{T}, npart::Integer,
                  nthreads::Integer = Threads.nthreads()) where {T}
    k = axis.knots
    h = (k[end] - k[1]) / (length(k) - 1)
    maximum(abs, diff(k) .- h) <= 1e-9 * abs(h) ||
        throw(ArgumentError("`CellSort` assumes a uniform grid"))
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

"""Linear cell index of a point — the nearest knot, as in deposition. Computed,
the grid being uniform."""
@inline function cell_of(p, x0, h, nk)
    i(u) = clamp(round(Int32, (u - x0) / h) + Int32(1), Int32(1), Int32(nk))
    i(p[1]) + Int32(nk) * (i(p[2]) - Int32(1) + Int32(nk) * (i(p[3]) - Int32(1)))
end

"""
    cellsort!(cs, positions) -> cs

Orders the particles by cell. Fills `perm`, `occupied` and `bounds`.

Four passes: cell of each particle, merge of the counters, offsets, placement.
The first two and the last are parallel; the third visits only the **occupied**
cells, ten times fewer than the rest — the cluster fills only a fraction of the
box.
"""
function cellsort!(cs::CellSort, positions)
    length(positions) == length(cs.perm) ||
        throw(DimensionMismatch("sort sized for $(length(cs.perm)) particles"))
    x0, h, nk = cs.x0, cs.h, cs.nknots

    # 1. Cell of each particle, and per-chunk counting.
    # ⚠️ Reset the counters: without this, two successive calls accumulate them,
    # the offsets become wrong and the placement writes out of bounds.
    Threads.@threads for t in eachindex(cs.chunks)
        cnt = cs.partial[t]
        fill!(cnt, Int32(0))
        @inbounds for i in cs.chunks[t]
            c = cell_of(positions[i], x0, h, nk)
            cs.keys[i] = c
            cnt[c] += Int32(1)
        end
    end

    # 2. Merge. ⚠️ Into `total`, never into `partial[1]`: the offsets need each
    # chunk's counters, the first one included.
    copyto!(cs.total, cs.partial[1])
    for t in 2:length(cs.partial)
        cs.total .+= cs.partial[t]
    end

    # 3. Occupied cells, then offsets — restricted to those.
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

    # 4. Placement. Each chunk writes into its own region of each cell, hence no
    # synchronisation.
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

"""Number of occupied cells — ten times fewer than cells, in practice."""
noccupied(cs::CellSort) = length(cs.occupied)
