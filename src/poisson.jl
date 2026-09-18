"""
Assembly of the right-hand side of Poisson's equation (the Fortran's `makerh2`).

The computational domain is finite, but a cluster's potential is not: on the
faces we impose the value the potential would take at large distance, given by
the **multipole expansion** of the density — monopole and quadrupole, expressed
in the barycentre frame where the dipole vanishes.

Those boundary values being non-zero, they are **lifted**: they are carried over
to the right-hand side through the outermost columns of the complete operator.
"""

"""
    Multipole(charge, center, quadrupole)

Multipole expansion of a charge distribution.

`center` is the barycentre — sitting there cancels the dipole term, leaving
monopole and quadrupole. `quadrupole` keeps only the 6 independent components of
the symmetric tensor, in the order `xx, yy, zz, xy, xz, yz`.
"""
struct Multipole{T}
    charge::T
    center::NTuple{3,T}
    quadrupole::NTuple{6,T}
end

"""
    contract(c, u, v, w) -> T

Contracts the spline coefficients with an outer product of 1D moments:
`Σ c[i,j,k]·u[i]·v[j]·w[k]`. Every multipole moment has this form, up to a
choice of moments.
"""
function contract(c::Array{T,3}, u, v, w) where {T}
    # ⚠️ Deliberately SEQUENTIAL. One contraction costs ~0.3 ms: splitting it
    # over eight threads brings it to 0.8 ms, the orchestration dominating the
    # computation. Parallelism is taken one level up, in `multipole`, where the
    # ten contractions are independent of one another.
    s = zero(T)
    @inbounds for k in eachindex(w), j in eachindex(v), i in eachindex(u)
        s += c[i, j, k] * u[i] * v[j] * w[k]
    end
    s
end

"""
Kernel: one work-item per `(j,k)` pair, writing the ten partial moments of its
own column. The sum over the columns is left to the reduction that follows.

The loop over `i` stays inside the work-item: it runs along the **contiguous**
direction, and it is what lets the three sums `s0, s1, s2` feed all ten moments
from a single read of `c`.
"""
@kernel function _all_moments_kernel!(partials, @Const(c),
                                      @Const(p0x), @Const(p0y), @Const(p0z),
                                      @Const(p1x), @Const(p1y), @Const(p1z),
                                      @Const(p2x), @Const(p2y), @Const(p2z),
                                      nx, ny)
    t = @index(Global, Linear)
    @inbounds begin
        j = (t - 1) % ny + 1
        k = (t - 1) ÷ ny + 1

        a  = p0y[j] * p0z[k]
        b  = p1y[j] * p0z[k]
        cc = p0y[j] * p1z[k]
        d  = p2y[j] * p0z[k]
        e  = p0y[j] * p2z[k]
        f  = p1y[j] * p1z[k]

        # The only three sums over `i` that the ten moments need.
        Tp = eltype(partials)
        s0 = zero(Tp); s1 = zero(Tp); s2 = zero(Tp)
        for i in 1:nx
            v = c[i, j, k]
            s0 += v * p0x[i]
            s1 += v * p1x[i]
            s2 += v * p2x[i]
        end

        partials[1, t]  = s0 * a     # q
        partials[2, t]  = s1 * a     # d100
        partials[3, t]  = s0 * b     # d010
        partials[4, t]  = s0 * cc    # d001
        partials[5, t]  = s2 * a     # m200
        partials[6, t]  = s0 * d     # m020
        partials[7, t]  = s0 * e     # m002
        partials[8, t]  = s1 * b     # mxy
        partials[9, t]  = s1 * cc    # mxz
        partials[10, t] = s0 * f     # myz
    end
end

"""
    all_moments(c, p0, p1, p2, partials) -> NTuple{10,T}

The ten contractions of the multipole expansion, in a **single pass** over the
coefficients.

Computing them separately reads `c` ten times. But the operation is limited by
memory bandwidth and not by arithmetic — measured: the ten contractions launched
in parallel over eight threads are 2.6 times SLOWER than the same thing
sequentially, because they contend for memory instead of sharing work.

A single pass factors everything: for each pair `(j,k)`, three partial sums over
`i` suffice to feed all ten moments.

The parallelism therefore goes over `(j,k)` and **not** over the ten moments —
which is the same conclusion as the paragraph above, now enforced by the shape
of the kernel. Measured on 10 threads against the former sequential loop: ×2.1
at 90³, ×1.23 at 134³, bit for bit identical since each column's ten values are
computed exactly as before.

`partials` is the caller's scratch — `mesh.moment_partials` — and not an
allocation: this runs at every step, on both grids.
"""
function all_moments(c::AbstractArray{T,3}, p0, p1, p2,
                     partials::AbstractMatrix{T}) where {T}
    p0x, p0y, p0z = p0
    p1x, p1y, p1z = p1
    p2x, p2y, p2z = p2
    nx, ny, nz = size(c)
    njk = ny * nz

    backend = get_backend(c)
    _all_moments_kernel!(backend)(partials, c, p0x, p0y, p0z, p1x, p1y, p1z,
                                  p2x, p2y, p2z, nx, ny; ndrange = njk)
    synchronize(backend)

    # One pass for the ten rows, then the ten scalars come back to the host —
    # they are consumed as scalars by `Multipole` and by the boundary kernel.
    totals = Array(vec(sum(view(partials, :, 1:njk); dims = 2)))
    ntuple(r -> totals[r], 10)
end

"""
    multipole(ρ, mesh) -> Multipole

Moments of the density `ρ` given at the collocation points.

The integration ought to go through the spline coefficients — `∫f = Σ cᵦ ∫φᵦ` —
but they need not be formed. With `c = S⁻¹ρ`:

    Σ c[i,j,k]·u[i]v[j]w[k] = Σ ρ[a,b,c]·ũ[a]ṽ[b]w̃[c]     where  ũ = S⁻ᵀu

Transforming the three moment **vectors** costs three matrix-vector products;
transforming the **3D array** cost three matrix-matrix products, four orders of
magnitude more. And since the moments depend only on the mesh, `SplineMesh`
already keeps them transformed.
"""
function multipole(ρ::AbstractArray{T,3}, mesh::SplineMesh{3,T}) where {T}
    # The dual moments already carry `S⁻ᵀ`: we contract the density directly,
    # without forming its spline coefficients.
    p0 = map(m -> m[1], mesh.dual_moments)
    p1 = map(m -> m[2], mesh.dual_moments)
    p2 = map(m -> m[3], mesh.dual_moments)

    q, d100, d010, d001, m200, m020, m002, mxy, mxz, myz =
        all_moments(ρ, p0, p1, p2, mesh.moment_partials)

    # Dipole, brought to the barycentre. A vanishing charge density has none.
    dip = (d100, d010, d001)
    center = iszero(q) ? ntuple(_ -> zero(T), 3) : dip ./ q

    # Quadrupole in traceless form: Qₗₗ = 2∫xₗ² − Σ_{m≠l} ∫xₘ², and
    # Qₗₘ = 3∫xₗxₘ off the diagonal.
    quad = (2m200 - m020 - m002,
            2m020 - m200 - m002,
            2m002 - m200 - m020,
            3mxy, 3mxz, 3myz)

    # Translation of the tensor to the barycentre (parallel axis theorem).
    bx, by, bz = center
    b2 = bx^2 + by^2 + bz^2
    shift = (q * (3bx^2 - b2), q * (3by^2 - b2), q * (3bz^2 - b2),
             3q * bx * by, 3q * bx * bz, 3q * by * bz)

    Multipole{T}(q, center, quad .- shift)
end

"""
    potential(mp, x, y, z) -> T

Potential of the multipole expansion at the given point: `q/r` plus the
quadrupole term in `1/r⁵`. The off-diagonal components count twice, the tensor
being symmetric.
"""
function potential(mp::Multipole{T}, x, y, z) where {T}
    px, py, pz = (x, y, z) .- mp.center
    r2 = px^2 + py^2 + pz^2
    invr = inv(sqrt(r2))
    qxx, qyy, qzz, qxy, qxz, qyz = mp.quadrupole
    quad = qxx * px^2 + qyy * py^2 + qzz * pz^2 +
           2 * (qxy * px * py + qxz * px * pz + qyz * py * pz)
    mp.charge * invr + quad * invr^5 / 2
end

"""
    foreach_face(f, nx, ny, nz)

Applies `f(i, j, k)` to the points on the **surface** of a grid, and to those
only.

Sweeping the whole volume and discarding the interior by a test visits 195 000
points to handle 19 500: nine tenths of the time spent deciding to do nothing.

The two full faces are handled apart from the side walls. This is not fussiness:
between them they account for a third of the points, and spreading them together
with the rest would unbalance the threads.
"""
function foreach_face(f, nx::Integer, ny::Integer, nz::Integer)
    tforeach(ny) do slice
        for j in slice, i in 1:nx
            f(i, j, 1)
            f(i, j, nz)
        end
    end
    tforeach(nz - 2) do slice
        for kk in slice
            k = kk + 1
            for i in 1:nx
                f(i, 1, k)
                f(i, ny, k)
            end
            for j in 2:(ny-1)
                f(1, j, k)
                f(nx, j, k)
            end
        end
    end
end

"""
    boundary_potential!(φ, mesh, mp) -> φ

Fills `φ` (full grid size) with the multipole potential on the **faces** of the
domain. The interior is left at zero: it is never read, only the faces serve the
lifting.
"""
function boundary_potential!(φ::Array{T,3}, mesh::SplineMesh{3,T},
                             mp::Multipole{T}) where {T}
    gx, gy, gz = map(ax -> ax.colloc, mesh.axes)
    nx, ny, nz = length(gx), length(gy), length(gz)
    fill!(φ, zero(T))
    foreach_face(nx, ny, nz) do i, j, k
        @inbounds φ[i, j, k] = potential(mp, gx[i], gy[j], gz[k])
    end
    φ
end

"""
    poisson_rhs!(rhs, ρ, mesh) -> rhs

Assembles the right-hand side of `∇²Φ = −4πρ` at the **interior** collocation
points, multipole boundary conditions included.

`ρ` covers the whole grid, `rhs` only the interior — which is what
[`solve!`](@ref) expects.
"""
function poisson_rhs!(rhs::Array{T,3}, ρ::Array{T,3}, mesh::SplineMesh{3,T},
                      φ::Array{T,3}) where {T}
    size(rhs) == size(mesh) ||
        throw(DimensionMismatch("rhs must have the size of the interior problem"))

    Dx, Dy, Dz = mesh.laplacians
    nsx, nsy, nsz = size(mesh)
    nx, ny, nz = size(φ)
    c = -4 * T(π)

    # ⚠️ **A single pass.** The natural form — one broadcast for the density then
    # six for the lifting of the faces — makes seven sweeps over 681 000 points,
    # and cost 4.6 ms where this one costs 0.29: sixteen times more, and more
    # than the tensor solver it feeds. The result is identical bit for bit.
    #
    # Each face contributes through the corresponding column of the complete
    # operator, seen from the interior rows. The `y` and `z` terms do not depend
    # on `i`: they come out of the inner loop.
    Threads.@threads for k in 1:nsz
        @inbounds for j in 1:nsy
            dyl = Dy[j+1, 1]; dyr = Dy[j+1, ny]
            dzl = Dz[k+1, 1]; dzr = Dz[k+1, nz]
            for i in 1:nsx
                rhs[i, j, k] = c * ρ[i+1, j+1, k+1] -
                    Dx[i+1, 1] * φ[1, j+1, k+1] - Dx[i+1, nx] * φ[nx, j+1, k+1] -
                    dyl * φ[i+1, 1, k+1] - dyr * φ[i+1, ny, k+1] -
                    dzl * φ[i+1, j+1, 1] - dzr * φ[i+1, j+1, nz]
            end
        end
    end
    rhs
end

"""
Variant that computes the boundary potential itself. To be avoided inside a time
loop: [`poisson!`](@ref) reuses it rather than redoing the multipole
contractions.
"""
poisson_rhs!(rhs::Array{T,3}, ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T} =
    poisson_rhs!(rhs, ρ, mesh,
                 boundary_potential!(mesh.scratch[1], mesh, multipole(ρ, mesh)))

"""Allocating version of [`poisson_rhs!`](@ref)."""
poisson_rhs(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T} =
    poisson_rhs!(Array{T,3}(undef, size(mesh)), ρ, mesh)

"""
    poisson!(φ, ρ, mesh) -> φ

Solves `∇²Φ = −4πρ` over the whole grid: the faces receive the multipole
potential, the interior the solution of the tensor system.

`φ` has the full size of the collocation grid, as does `ρ`. This is the complete
density → potential chain, and the Fortran's `makerh2` + `solve`.
"""
function poisson!(φ::Array{T,3}, ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T}
    boundary_potential!(φ, mesh, multipole(ρ, mesh))
    solve_interior!(φ, ρ, mesh)
end

"""
    solve_interior!(φ, ρ, mesh) -> φ

Solves the interior, taking as Dirichlet conditions the values **already
present** on the faces of `φ`.

This is the half shared by [`poisson!`](@ref), which lays those values down by
multipole expansion, and by the inter-grid junction, which reads them from the
coarser level's solution.
"""
function solve_interior!(φ::Array{T,3}, ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T}
    rhs = poisson_rhs!(mesh.scratch_inner, ρ, mesh, φ)
    solve!(rhs, rhs, mesh.solver)
    @views φ[2:end-1, 2:end-1, 2:end-1] .= rhs
    φ
end

"""
    boundary_from_coarse!(φ, mesh, coarse, csol_coarse) -> φ

Lays onto the faces of `φ` the values read from a coarser grid's solution (the
Fortran's `makerhsf`).

This is the whole inter-level junction: of the outside world, the fine grid sees
only what the coarse one tells it at its boundary.
"""
function boundary_from_coarse!(φ::Array{T,3}, mesh::SplineMesh{3,T},
                               coarse::SplineMesh{3,T}, csol_coarse::Array{T,3}) where {T}
    gx, gy, gz = map(ax -> ax.colloc, mesh.axes)
    nx, ny, nz = length(gx), length(gy), length(gz)
    fill!(φ, zero(T))
    foreach_face(nx, ny, nz) do i, j, k
        p = spline_potential(coarse.axes, csol_coarse, (gx[i], gy[j], gz[k]))
        p === nothing && throw(ArgumentError(
            "the boundary point ($(gx[i]), $(gy[j]), $(gz[k])) lies outside the " *
            "coarse grid: the levels are not nested"))
        @inbounds φ[i, j, k] = p
    end
    φ
end

"""Allocating version of [`poisson!`](@ref)."""
poisson(ρ::Array{T,3}, mesh::SplineMesh{3,T}) where {T} = poisson!(similar(ρ), ρ, mesh)

"""
    poisson!(φs, ρs, nested) -> φs

Solves Poisson on a hierarchy of nested grids, from coarsest to finest.

The coarsest level takes its conditions from the multipole expansion of its own
density; each finer level reads its own from the solution of the level above.
`φs` and `ρs` are ordered like the levels, finest to coarsest.
"""
function poisson!(φs::NTuple{L,Array{T,3}}, ρs::NTuple{L,Array{T,3}},
                  nested::NestedMeshes{L,3,T}) where {L,T}
    poisson!(φs[L], ρs[L], nested[L])
    for l in (L-1):-1:1
        coefs = spline_coefficients!(nested[l+1].scratch[2], φs[l+1], nested[l+1])
        boundary_from_coarse!(φs[l], nested[l], nested[l+1], coefs)
        solve_interior!(φs[l], ρs[l], nested[l])
    end
    φs
end
