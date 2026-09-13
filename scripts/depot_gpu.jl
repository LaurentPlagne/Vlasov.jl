#!/usr/bin/env julia
"""
Charge deposit on GPU: the two routes, measured side by side.

    julia --project=gpu -t auto scripts/depot_gpu.jl

The deposit is a **scatter**: each particle writes into 8³ = 512 grid points,
and neighbouring particles write to the same places. It is the heaviest stage
of the step once the forces are on GPU, and the hardest to port.

Two routes, and they are not remotely equivalent:

  * **atomic** — one particle per thread, 512 atomic additions. Simple, needs no
    preparation, and **three times slower than the CPU**: 800 000 × 512 = 410
    million contended atomics.
  * **sorted** — particles are ranged by cell, then a group of 512 threads
    handles one cell and **each thread owns one stencil point**. It walks every
    particle of the cell accumulating in a register and performs only **one**
    atomic at the end. Reversing the loops divides the atomics by the number of
    particles per cell — 108 here.

This is the thesis's sort, and for the same reason as in 1997: data locality.
It benefits the CPU too, which gains ×1.34 on this deposit with nothing else
changed.

This file is a **measured prototype**, not the production path: it is not yet
wired into `update_forces!`.
"""

using Vlasov, Metal, Printf, LinearAlgebra

const ROOT = dirname(@__DIR__)
const NPART = 800_000

# ---------------------------------------------------------------------------
# Parallel counting sort
# ---------------------------------------------------------------------------

"""Linear index of the cell — the nearest knot, as the deposit does. The fine
grid being uniform, it is **computed**: no bisection."""
@inline function cell_of(p, x0, h, nk)
    kx = clamp(round(Int32, (p[1] - x0) / h) + Int32(1), Int32(1), Int32(nk))
    ky = clamp(round(Int32, (p[2] - x0) / h) + Int32(1), Int32(1), Int32(nk))
    kz = clamp(round(Int32, (p[3] - x0) / h) + Int32(1), Int32(1), Int32(nk))
    kx + Int32(nk) * (ky - Int32(1) + Int32(nk) * (kz - Int32(1)))
end

"""Count per chunk. ⚠️ Resets the counters: without that, a repeated
measurement accumulates them and the sort produces out-of-bounds indices."""
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

"""List of the **occupied** cells. Of 91 125 cells, 7 413 are: the cluster of
radius 40 fills only a fraction of the ±78 box."""
function occupied_cells(total::Vector{Int32})
    occ = Vector{Int32}(undef, count(>(Int32(0)), total))
    j = 0
    @inbounds for c in eachindex(total)
        total[c] > Int32(0) && (j += 1; occ[j] = Int32(c))
    end
    occ
end

"""Where each chunk writes, for each cell: the parallel counting sort.

⚠️ Sweep only the **occupied** cells. The version that walked all 91 125 cells
cost 1.86 ms; restricted, it costs 0.13, and building the list 0.06 — a net
gain of 1.67 ms out of a 6.7 ms preparation."""
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

"""Placement. Each thread writes into its own region of every cell: no
synchronisation at all. ⚠️ A `Dict` in place of `offsets` cost ten milliseconds
on its own — twenty times this version."""
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
# Metal kernels
# ---------------------------------------------------------------------------

"""**Atomic** version: one particle per thread, 512 atomic additions."""
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

"""Table columns, on GPU.

This is the biggest piece of the sort's preparation — 3.63 ms on CPU — and it
has no business being there: it is purely particle-wise, and the positions are
already uploaded for the forces. Only the **permutation** needs uploading, and
that is a vector of integers.

⚠️ **This kernel is not the one retained.** Computed in `Float32`, it tips
0.05 % of the columns onto their neighbour and takes the density discrepancy
from 1.5e-07 to 9.0e-05. The cause is structural: the `Float32` ULP at 78 a₀ is
7.6e-06, that is 0.22 % of a column's width (0.00355 a₀). One particle in five
hundred sits less than one ULP from a boundary, and the wrong Gaussian sample
is applied to it — a wrong discrete choice, not a rounding error that averages
out.

It is kept here so the measurement can be replayed. See `docs/gpu.md` for what
should be done instead: upload `(k, δ)` computed in `Float64` on the host,
rather than the absolute positions.
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

"""**Sorted** version: one group of 512 threads per occupied cell, each thread
owning one stencil point.

The table indices go through threadgroup memory, loaded in slabs of 64:
without that, each of the 512 threads would re-read every particle's data, and
multiply the memory traffic by as much."""
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

    # --- sort ---------------------------------------------------------------
    nch = Threads.nthreads()
    chunks = [round(Int, NPART * (t - 1) / nch) + 1 : round(Int, NPART * t / nch) for t in 1:nch]
    ncell = nk^3
    keys = Vector{Int32}(undef, NPART)
    partial = [Vector{Int32}(undef, ncell) for _ in 1:nch]
    offsets = [Vector{Int32}(undef, ncell) for _ in 1:nch]
    perm = Vector{Int32}(undef, NPART)
    cols = Matrix{Int32}(undef, 3, NPART)
    # ⚠️ A separate buffer: writing the totals into `partial[1]` would destroy
    # the first chunk's counters, which `slice_offsets!` needs. The price of
    # forgetting is an out-of-bounds write, hence a segmentation fault.
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

    """The columns the old way: on CPU, for comparison."""
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

    @printf("Na1000, %d particles, grid %d — %d occupied cells out of %d, %.0f per cell\n\n",
            NPART, p.nfine, length(occ), ncell, NPART / length(occ))
    @printf("%-34s %9s %14s\n", "version", "ms", "error vs ref.")
    for (label, t, ρ) in (("CPU, current order",
                           chrono(() -> deposit_smoothed!(ρref, fine, sm, pos;
                                                          charge = w, buffers = sim.scatter[1])), nothing),
                          ("CPU, sorted order",
                           chrono(() -> deposit_smoothed!(ρs, fine, sm, sorted_pos;
                                                          charge = w, buffers = buf)), ρs),
                          ("GPU atomic", chrono(run_atomic), normalised(run_atomic)),
                          ("GPU sorted", chrono(run_sorted), normalised(run_sorted)))
        e = ρ === nothing ? "" : @sprintf("%.1e", norm(ρ - ρref) / norm(ρref))
        @printf("%-34s %9.1f %14s\n", label, t, e)
    end
    @printf("\n%-34s %9.2f\n", "sort: permutation (CPU)", chrono(sort!; k = 4))
    @printf("%-34s %9.2f\n", "table columns (CPU)", chrono(cols_cpu!))
    @printf("%-34s %9.2f\n", "table columns (GPU)", chrono(run_cols))

    # Do the columns computed in Float32 designate the same ones?
    cols_cpu!(); ref_cols = copy(cols)
    run_cols(); gpu_cols = Array(gcols)
    diff = count(!=(0), gpu_cols .- ref_cols)
    ρ_gpu = normalised(run_sorted)
    @printf("\ndiffering columns: %d out of %d (%.3f %%)\n", diff, 3NPART, 100diff / (3NPART))
    @printf("density discrepancy: %.1e\n", norm(ρ_gpu - ρref) / norm(ρref))
end

main()
