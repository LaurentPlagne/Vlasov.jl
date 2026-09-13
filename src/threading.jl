"""
Chunking for parallel loops.

The loops that weigh on a time step all have the same shape: an independent
sweep over the pseudo-particles or the grid points, optionally followed by a
sum. They are split into **contiguous** chunks — not interleaved — so that each
thread works on a continuous region of memory.
"""

using LinearAlgebra: BLAS

"""
Global parallelism switch.

It serves two purposes: comparing a parallel and a sequential version
**alternately** on the same machine at the same moment — the only way to measure
anything when the load varies — and recovering deterministic behaviour while
debugging.

    Vlasov.PARALLEL[] = false
"""
const PARALLEL = Ref(true)

"""Number of chunks to use: 1 when parallelism is switched off."""
@inline nchunks_now() = PARALLEL[] ? Threads.nthreads() : 1

"""
    configure_blas!(; threads = 2)

Caps the number of OpenBLAS threads.

⚠️ **The OpenBLAS default is bad here.** It takes as many threads as there are
cores, on top of Julia's, and the two fight over the machine. Measured on
`poisson!`, 8 Julia threads on 10 cores:

| BLAS threads | median | spread (max/min) |
|---|---|---|
| 1 | 11.1 ms | 1.1× |
| 2 | 9.6 ms | 1.4× |
| 4 | 8.6 ms | 2.6× |
| 8 *(default)* | **20.5 ms** | **5.4×** |

The default is not merely twice as slow in the median: it makes timings
**unpredictable**, by a factor of five. Two threads give the best compromise
between speed and regularity.

⚠️ On Apple Silicon, prefer `AppleAccelerate` outright: its BLAS has no thread
pool to contend with Julia's, which is worth ×1.31 on the whole step — see
`docs/gpu.md`.

Never called automatically: changing a global setting behind the caller's back
would be discourteous. Invoke it before a batch of runs.
"""
function configure_blas!(; threads::Integer = 2)
    BLAS.set_num_threads(threads)
    threads
end

"""
    chunks(n, nchunks = nchunks_now()) -> Vector{UnitRange}

Splits `1:n` into contiguous chunks of as nearly equal size as possible.
Returns one chunk fewer than asked rather than empty chunks.
"""
function chunks(n::Integer, nchunks::Integer = nchunks_now())
    nchunks = max(1, min(nchunks, n))
    base, remainder = divrem(n, nchunks)
    stop = 0
    map(1:nchunks) do c
        start = stop + 1
        stop += base + (c <= remainder)
        start:stop
    end
end

"""
    tmapreduce(f, n) -> T

Sums `f(range)` over the chunks of `1:n`, in parallel. `f` receives a **chunk**
and not an index: it is up to `f` to loop, which lets it accumulate into a local
variable rather than into a shared array.

Falls back to a direct single-threaded call, so that the result is identical —
up to summation order — and the overhead disappears.
"""
function tmapreduce(f, n::Integer)
    parts = chunks(n)
    length(parts) == 1 && return f(parts[1])
    results = Vector{Any}(undef, length(parts))
    Threads.@threads for c in eachindex(parts)
        results[c] = f(parts[c])
    end
    reduce(_addall, results)
end

"""Element-wise addition, accepting numbers as well as tuples."""
@inline _addall(a::Number, b::Number) = a + b
@inline _addall(a::Tuple, b::Tuple) = map(_addall, a, b)

"""
    tforeach(f, n)

Applies `f` to each chunk of `1:n`, in parallel, collecting nothing.
For loops where each write goes to its own slot.
"""
function tforeach(f, n::Integer)
    parts = chunks(n)
    length(parts) == 1 && return f(parts[1])
    Threads.@threads for c in eachindex(parts)
        f(parts[c])
    end
    nothing
end
