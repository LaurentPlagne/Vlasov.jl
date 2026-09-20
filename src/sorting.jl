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
                  nthreads::Integer = Threads.nthreads();
                  buffers::Bool = true) where {T}
    k = axis.knots
    h = (k[end] - k[1]) / (length(k) - 1)
    maximum(abs, diff(k) .- h) <= 1e-9 * abs(h) ||
        throw(ArgumentError("`CellSort` assumes a uniform grid"))
    nk = length(k)
    ncell = nk^3
    # `buffers = false` keeps only what a *device* sort fills — `occupied` and
    # `bounds`, its cell list. What it leaves out is what the host sort would
    # need: two vectors of `npart` and two of `ncell` per thread, which at
    # 8×10⁷ particles on 111³ cells is 720 MB nobody reads on that route.
    # [`_ensure_buffers!`](@ref) gives them back if a host sort does turn up.
    nt = buffers ? nthreads : 0
    cells(n) = [Vector{Int32}(undef, ncell) for _ in 1:n]
    CellSort{T}(k[1], h, nk,
                Vector{Int32}(undef, buffers ? npart : 0),
                cells(nt),
                Vector{Int32}(undef, buffers ? ncell : 0),
                cells(nt),
                Vector{Int32}(undef, buffers ? npart : 0),
                Int32[], Int32[], chunks(npart, nthreads))
end

"""
    _ensure_buffers!(cs, npart) -> cs

Gives back the buffers [`CellSort`](@ref) was built without, the first time a
host sort actually asks for them.

⚠️ **Not `buffers = false` as a promise, only as a default.** An accelerator
built for a device-resident cloud can still be handed a host-held one — the
suite does exactly that, to compare the two force paths on the same tables —
and the host sort must then work rather than complain. Growing the vectors in
place is enough: `CellSort` is immutable, its fields are not.
"""
function _ensure_buffers!(cs::CellSort, npart::Integer)
    isempty(cs.perm) || return cs
    ncell = cs.nknots^3
    resize!(cs.keys, npart)
    resize!(cs.perm, npart)
    resize!(cs.total, ncell)
    for v in (cs.partial, cs.offsets), _ in (length(v)+1):length(cs.chunks)
        push!(v, Vector{Int32}(undef, ncell))
    end
    cs
end

"""Linear cell index of a point — the nearest knot, as in deposition. Computed,
the grid being uniform."""
@inline function cell_of(p, x0, h, nk)
    i(u) = clamp(round(Int32, (u - x0) / h) + Int32(1), Int32(1), Int32(nk))
    i(p[1]) + Int32(nk) * (i(p[2]) - Int32(1) + Int32(nk) * (i(p[3]) - Int32(1)))
end

"""
Cell of each particle, and per-chunk counting — the one pass of the sort that
depends on the positions.

⚠️ Reset the counters: without this, two successive calls accumulate them, the
offsets become wrong and the placement writes out of bounds.
"""
function _count_cells!(cs::CellSort, positions)
    x0, h, nk = cs.x0, cs.h, cs.nknots
    Threads.@threads for t in eachindex(cs.chunks)
        cnt = cs.partial[t]
        fill!(cnt, Int32(0))
        @inbounds for i in cs.chunks[t]
            c = cell_of(positions[i], x0, h, nk)
            cs.keys[i] = c
            cnt[c] += Int32(1)
        end
    end
end

"""
The same pass on a cloud held as `(k, δ)`: the cell key **is** `knode`, so it is
read rather than recomputed.

The saving is not the arithmetic but the traffic — 12 bytes of `Int32` per
particle instead of 24 of `Float64`, and no division. Measured at 8×10⁷
particles: **53.2 ms → 17.8**, for keys identical bit for bit.

`knode` is unclamped by [`PackedPositions`](@ref), so the clamp that `cell_of`
applies happens here instead.
"""
function _count_cells!(cs::CellSort, positions::PackedPositions)
    data = positions.data
    prev = positions.prev
    nk = Int32(cs.nknots)
    Threads.@threads for t in eachindex(cs.chunks)
        cnt = cs.partial[t]
        fill!(cnt, Int32(0))
        @inbounds for i in cs.chunks[t]
            k, _ = _half(data[i], prev)
            kx = clamp(k[1], Int32(1), nk)
            ky = clamp(k[2], Int32(1), nk)
            kz = clamp(k[3], Int32(1), nk)
            c = kx + nk * (ky - Int32(1) + nk * (kz - Int32(1)))
            cs.keys[i] = c
            cnt[c] += Int32(1)
        end
    end
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
    _ensure_buffers!(cs, length(positions))
    length(positions) == length(cs.perm) ||
        throw(DimensionMismatch("sort sized for $(length(cs.perm)) particles"))

    # 1. Cell of each particle, and per-chunk counting.
    _count_cells!(cs, positions)

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
