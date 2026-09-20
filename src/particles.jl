"""
Pseudo-particles and time integration.

The method samples the phase-space density with pseudo-particles, each
representing `weight` electrons: its mass and charge are those of those `weight`
electrons taken together.

The integration is a position Verlet: the velocity is never stored, it is read
from the gap between two successive positions.
"""

"Electron mass, in atomic units (the Fortran's `mel`)."
const ELECTRON_MASS = 1.0

"Electron charge, in atomic units (the Fortran's `qel`)."
const ELECTRON_CHARGE = -1.0

"""
    ParticleCloud(positions, previous, forces, weight)

Cloud of pseudo-particles for a position Verlet scheme.

`previous` holds the positions at the previous step — that is what stands in for
the velocity. `weight` is the number of electrons each pseudo-particle
represents.

The Fortran stored all of this in `(3, npartmax)` arrays in column order, which
is exactly the memory layout of a `Vector{NTuple{3,T}}`: the port is a
reinterpretation, not a conversion.

That coincidence is what spares us a rewrite for the accelerators. The **element
type stays** `NTuple{3,T}` — it is already the `(3, N)` layout a device wants —
and only the **container** is a parameter, so that `A` may be a `Vector` or any
device vector. `ParticleCloud{T}` remains valid in a signature: the container is
left free there.
"""
struct ParticleCloud{T<:AbstractFloat,A<:AbstractVector{NTuple{3,T}},
                     F<:AbstractVector{NTuple{3,T}}}
    positions::A
    previous::A
    """Forces carry their own container type. They are a vector field, not
       positions, and a cloud whose positions are a [`PackedPositions`](@ref)
       still wants plain triples here."""
    forces::F
    weight::T
end

"""
    PackedPositions(knode, delta, x0, h)

Positions held as **cell index and offset** rather than as absolute coordinates:
`x = x0 + (knode − 1)·h + delta`, with `delta` bounded by `h/2`.

This is the form every device kernel already consumes — see
[`_pack_kd_kernel!`](@ref) — so a cloud held this way needs no packing step at
all. It reads as an ordinary `AbstractVector{NTuple{3,T}}`, rebuilding the
absolute triple on access, so every consumer of `cloud.positions` keeps working.

⚠️ **`knode` is not clamped to the grid.** A particle that has left the fine
mesh keeps a virtual cell index — negative, or past the last knot — so the pair
stays an exact description of where it is. Clamping belongs to the kernels that
index a stencil, not to the representation.

Why this form and not `Float32` coordinates: `δ` is small, so `Float32` resolves
it to 6e-8, where the same `Float32` on an absolute coordinate at 78 a₀ resolves
7.6e-6 — **0.54 % of a smoothing-table column**. The difference compounds in the
position Verlet, whose observables are all read from differences of positions:
measured over 101 steps on the real cloud, the drift is **40×** smaller in this
form (2.55e-4 a₀ against 1.03e-2).
"""
struct PackedPositions{T,K,D} <: AbstractVector{NTuple{3,T}}
    knode::K
    delta::D
    x0::T
    h::T
end

Base.size(p::PackedPositions) = (size(p.knode, 2),)
Base.IndexStyle(::Type{<:PackedPositions}) = IndexLinear()

@inline Base.getindex(p::PackedPositions{T}, i::Int) where {T} =
    ntuple(d -> @inbounds(p.x0 + (p.knode[d, i] - one(eltype(p.knode))) * p.h +
                          T(p.delta[d, i])), 3)

@inline function Base.setindex!(p::PackedPositions{T}, x, i::Int) where {T}
    @inbounds for d in 1:3
        k = round(Int32, (T(x[d]) - p.x0) / p.h) + Int32(1)
        p.knode[d, i] = k
        p.delta[d, i] = eltype(p.delta)(T(x[d]) - (p.x0 + (k - Int32(1)) * p.h))
    end
    x
end

Base.similar(p::PackedPositions) =
    PackedPositions(similar(p.knode), similar(p.delta), p.x0, p.h)

Base.copy(p::PackedPositions) =
    PackedPositions(copy(p.knode), copy(p.delta), p.x0, p.h)

Adapt.adapt_structure(to, p::PackedPositions) =
    PackedPositions(adapt(to, p.knode), adapt(to, p.delta), p.x0, p.h)

function ParticleCloud(positions::AbstractVector{NTuple{3,T}}, weight::T) where {T}
    # `similar` and not `fill`: a device cloud must yield device buffers.
    p = copy(positions)
    zero3 = ntuple(_ -> zero(T), 3)
    ParticleCloud(p, fill!(similar(p), zero3), fill!(similar(p), zero3), weight)
end

Base.length(c::ParticleCloud) = length(c.positions)

"""Moves the three buffers to another backend — `adapt(MtlArray, cloud)` and the
like. `weight` is a scalar and travels as it is."""
Adapt.adapt_structure(to, c::ParticleCloud) =
    ParticleCloud(adapt(to, c.positions), adapt(to, c.previous),
                  adapt(to, c.forces), c.weight)

"""Mass of a pseudo-particle: that of the `weight` electrons it carries."""
mass(c::ParticleCloud) = ELECTRON_MASS * c.weight

"""Charge of a pseudo-particle."""
charge(c::ParticleCloud) = ELECTRON_CHARGE * c.weight

"""Cross product of two triples."""
@inline cross3(a, b) = (a[2] * b[3] - a[3] * b[2],
                        a[3] * b[1] - a[1] * b[3],
                        a[1] * b[2] - a[2] * b[1])

"""
    step!(cloud, dt; rcmax = Inf) -> (; kinetic, escaped, angular)

Advances the cloud by one time step with position Verlet:

    q(t+dt) = 2q(t) − q(t−dt) + dt²·F/M

and returns the diagnostics the Fortran computed in the same loop — kinetic
energy, total angular momentum, and the share of the kinetic energy carried by
the particles that have **left**. They are obtained from the centred momentum
`p = M·(q(t+dt) − q(t−dt))/2dt`, which exists only here: computing it afterwards
would require keeping one more piece of state.

`rcmax` is the radius beyond which a particle counts as gone, measured on the
position **before** the step — like the Fortran's `move`, which tests `ract` on
`qp` and not on the new position. It was hard-coded to `100.d0` until 1997, and
becomes an input parameter in 1998. The default `Inf` counts nothing as gone,
leaving `escaped` at zero for callers who do not use it.
"""
function step!(cloud::ParticleCloud{T}, dt::T; rcmax::Real = T(Inf)) where {T}
    M = mass(cloud)
    acc = dt^2 / M
    pfac = M / 2dt
    r2max = T(rcmax)^2
    # ⚠️ **Threaded by chunks, and recombined in chunk order.** This loop is
    # pure streaming — 108 bytes per particle, read and written once — and left
    # serial it ran at 30 GB/s where the ten cores sustain 171: one core's share,
    # exactly. It stayed serial because it carries three reductions, which is a
    # reason to chunk them, not a reason to give up the other nine cores.
    #
    # The recombination walks the chunks in order, so the result depends on the
    # thread *count* but never on the order they happen to finish in.
    parts = chunks(length(cloud.positions))
    nc = length(parts)
    ekins = zeros(T, nc)
    eouts = zeros(T, nc)
    angs = fill(ntuple(_ -> zero(T), 3), nc)

    Threads.@threads for c in 1:nc
        ek = zero(T); eo = zero(T)
        an = ntuple(_ -> zero(T), 3)
        @inbounds for i in parts[c]
            q, qold, f = cloud.positions[i], cloud.previous[i], cloud.forces[i]
            qnew = 2 .* q .- qold .+ acc .* f
            p = pfac .* (qnew .- qold)
            e = (p[1]^2 + p[2]^2 + p[3]^2) / 2M
            ek += e
            # Comparing squares: one square root per particle for a mere
            # threshold is one square root too many.
            q[1]^2 + q[2]^2 + q[3]^2 > r2max && (eo += e)
            an = an .+ cross3(q, p)
            cloud.previous[i] = q
            cloud.positions[i] = qnew
        end
        ekins[c] = ek; eouts[c] = eo; angs[c] = an
    end

    ekin = zero(T); eout = zero(T)
    angular = ntuple(_ -> zero(T), 3)
    @inbounds for c in 1:nc
        ekin += ekins[c]; eout += eouts[c]
        angular = angular .+ angs[c]
    end
    (; kinetic = ekin, escaped = eout, angular)
end

"""
    packed_cloud(axis, positions, weight, E; buffers = nothing) -> ParticleCloud

A cloud whose positions are held as `(k, δ)` on `axis`, offsets in `E`.

The offsets are what the precision argument applies to: `k` is exact whatever
`E` is, and that is the whole point of the form. `previous` is left at the
origin — [`prime_leapfrog!`](@ref) is what fills it.

`buffers` hands in the two `(knode, delta)` pairs the cloud is to live in —
current positions then previous — which is how the accelerator's own staging
becomes the cloud's storage: the step then has nothing to pack and nothing to
copy, and the integrator finds both positions where its kernel runs.
"""
function packed_cloud(axis::SplineAxis{T}, positions, weight::T,
                      ::Type{E} = Float32; buffers = nothing) where {T,E}
    k = axis.knots
    h = (k[end] - k[1]) / (length(k) - 1)
    n = length(positions)
    geom(kn, dl) = PackedPositions{T,typeof(kn),typeof(dl)}(kn, dl, T(k[1]), T(h))
    fresh() = geom(Matrix{Int32}(undef, 3, n), Matrix{E}(undef, 3, n))
    P = buffers === nothing ? fresh() : geom(buffers[1], buffers[2])
    O = buffers === nothing ? fresh() : geom(buffers[3], buffers[4])
    size(P.knode) == (3, n) ||
        throw(DimensionMismatch("buffers sized for $(size(P.knode, 2)) particles"))
    copyto!(P, positions)
    fill!(O, ntuple(_ -> zero(T), 3))
    ParticleCloud(P, O, fill(ntuple(_ -> zero(T), 3), n), weight)
end

"""One component of the packed Verlet: the new cell, the new offset, and the
centred gap `q(t+dt) − q(t−dt)` that the diagnostics are read from."""
@inline function _verlet_packed(kq, dq, ok, od, a, f, h::E) where {E}
    # The integer part carries no rounding whatsoever.
    kn = 2kq - ok
    dn = 2dq - od + a * E(f)
    # `δ` is kept inside half a cell; `m` is 0, ±1 or ±2 in practice.
    m = round(Int32, dn / h)
    kn += m
    dn -= E(m) * h
    (kn, dn, E(kn - ok) * h + (dn - od))
end

"""
    step!(cloud::ParticleCloud{T,<:PackedPositions}, dt; rcmax = Inf)

The same position Verlet, on a cloud held as cell index and offset.

What the form buys is that `k(t+dt) = 2k(t) − k(t−dt)` is **exact**: every
rounding that remains falls on `δ`, which is bounded by `h/2`. That matters
because both observables of the step — the centred momentum, and the kinetic
energy through it — are *differences of positions*, and a difference is what an
absolute coordinate in reduced precision destroys.

Measured over 101 steps on the production cloud, against the same integrator on
absolute `Float32` coordinates: **2.55e-4 a₀ of drift against 1.03e-2**, a factor
40 — and the gap widens with the step count.

The arithmetic runs in `eltype(delta)`, so the precision of the step is the
precision of the storage. `rcmax` and the angular momentum still need the
absolute position, which is rebuilt for them; it is a multiply-add per component
on a loop that is bound by memory.
"""
function step!(cloud::ParticleCloud{T,<:PackedPositions}, dt::T;
               rcmax::Real = T(Inf)) where {T}
    q, o = cloud.positions, cloud.previous
    E = eltype(q.delta)
    M = mass(cloud)
    # ⚠️ A function barrier, and not a convenience. Computing `E = eltype(...)`
    # and then `E(dt^2/M)` inside the sweep leaves the arithmetic type-unstable:
    # measured at 8×10⁷ particles, 162 ms that way against **56** with the
    # scalars passed in already typed — three times the cost of the whole step's
    # integrator, for a spelling.
    _verlet_packed_sweep!(q.knode, q.delta, o.knode, o.delta, cloud.forces,
                          E(dt^2 / M), E(q.h), q.x0, q.h, M / 2dt,
                          T(rcmax)^2, M, chunks(length(q)))
end

function _verlet_packed_sweep!(kq, dq, ok, od, forces, a::E, h::E, x0::T, hT::T,
                               pfac::T, r2max::T, M::T, parts) where {T,E}
    nc = length(parts)
    ekins = zeros(T, nc)
    eouts = zeros(T, nc)
    angs = fill(ntuple(_ -> zero(T), 3), nc)

    Threads.@threads for c in 1:nc
        ek = zero(T); eo = zero(T)
        an = ntuple(_ -> zero(T), 3)
        @inbounds for i in parts[c]
            f = forces[i]
            k1, k2, k3 = kq[1, i], kq[2, i], kq[3, i]
            o1, o2, o3 = ok[1, i], ok[2, i], ok[3, i]
            d1, d2, d3 = dq[1, i], dq[2, i], dq[3, i]
            e1, e2, e3 = od[1, i], od[2, i], od[3, i]

            n1 = _verlet_packed(k1, d1, o1, e1, a, f[1], h)
            n2 = _verlet_packed(k2, d2, o2, e2, a, f[2], h)
            n3 = _verlet_packed(k3, d3, o3, e3, a, f[3], h)

            p = (T(n1[3]) * pfac, T(n2[3]) * pfac, T(n3[3]) * pfac)
            e = (p[1]^2 + p[2]^2 + p[3]^2) / 2M
            ek += e
            qa = (x0 + (k1 - one(k1)) * hT + T(d1),
                  x0 + (k2 - one(k2)) * hT + T(d2),
                  x0 + (k3 - one(k3)) * hT + T(d3))
            qa[1]^2 + qa[2]^2 + qa[3]^2 > r2max && (eo += e)
            an = an .+ cross3(qa, p)

            ok[1, i] = k1; ok[2, i] = k2; ok[3, i] = k3
            od[1, i] = d1; od[2, i] = d2; od[3, i] = d3
            kq[1, i] = n1[1]; kq[2, i] = n2[1]; kq[3, i] = n3[1]
            dq[1, i] = n1[2]; dq[2, i] = n2[2]; dq[3, i] = n3[2]
        end
        ekins[c] = ek; eouts[c] = eo; angs[c] = an
    end

    ekin = zero(T); eout = zero(T)
    angular = ntuple(_ -> zero(T), 3)
    @inbounds for c in 1:nc
        ekin += ekins[c]; eout += eouts[c]
        angular = angular .+ angs[c]
    end
    (; kinetic = ekin, escaped = eout, angular)
end

"""
    half_step_back(positions, momenta, M, dt)

Leapfrog priming, first stage: `q(−dt/2) = q(0) − (dt/2M)·p`.

This is the Fortran's `moveback1`, where the array of previous positions still
held the momenta produced by the initial sampling.
"""
half_step_back(positions, momenta, M, dt) =
    map((q, p) -> q .- (dt / 2M) .* p, positions, momenta)

"""
    full_step_back(positions, half, forces, M, dt)

Leapfrog priming, second stage: from `q(0)` and `q(−dt/2)` to `q(−dt)`, using
the forces evaluated at `q(−dt/2)`.

⚠️ **Reproduces a dubious coefficient of the original code.** `moveback2`
computes

    coef2 = 0.5·dltt*2·npart/(mel·nbelec)

that is, `dt/M`. But `coef2·F` is then a *velocity*, added to lengths: the
formula is not dimensionally consistent. A Taylor expansion gives `dt²/4M`, and
the neighbouring routine `move` does write `dltt**2`. Everything points to a
typo, `dltt*2` for `dltt**2` — but it is present identically in **all five**
versions of the thesis code, hence never fixed.

The effect is one-off: it only distorts the priming, like an error on the
initial velocity. We reproduce it as is, to stay comparable with the oracle;
`consistent = true` selects the dimensionally consistent variant `dt²/4M`, to
measure what the typo costs.
"""
function full_step_back(positions, half, forces, M, dt; consistent::Bool = false)
    coef = consistent ? dt^2 / 4M : dt / M
    map((q, h, f) -> .-q .+ 2 .* h .+ coef .* f, positions, half, forces)
end

# No standalone `kinetic_energy(cloud, dt)`: the centred momentum at t is read
# between q(t+dt) and q(t−dt), which coexist only inside a step. An after-the-
# fact function would see only q(t) and q(t−dt) and would return the momentum at
# t−dt/2 — a different quantity under the same name. `step!` returns it.
