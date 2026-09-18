"""
Device-side mirror of a [`SplineMesh`](@ref)'s **computational** content.

The mesh itself stays where it is. It describes a discretisation, and holds a
good deal that has no business on a GPU: banded collocation factorisations, the
locator tables of the coarse deposition, the eigenbasis as it came out of
`eigen`. What the time loop actually *reads* is a short list of tables, and that
is what is copied here, once, in the kernels' own precision.

Every function of the chain was split in two beforehand — one half taking the
tables, one taking the mesh that holds them — so nothing below re-implements
anything: `poisson!` on a device mirror and `poisson!` on a host mesh run the
same code over different arrays.

⚠️ `E` is the precision of the copy. On Metal that is `Float32`, and the whole
Poisson chain then runs in single precision: the tensor solver is a similarity
transform, so the conditioning is that of the 1D operators, but this is a real
change of regime and not a detail. On CUDA or ROCm, `E === Float64` and the
chain is bit-comparable with the host.
"""
struct DeviceMesh{E,V,M,A3,VI}
    "Mirrors `SplineMesh.dual_moments`: orders 0, 1, 2 per direction."
    duals::NTuple{3,NTuple{3,V}}
    partials::M
    colloc::NTuple{3,V}
    knots::NTuple{3,V}
    laplacians::NTuple{3,M}
    "Eigenbasis of each 1D operator, and its inverse."
    Ms::NTuple{3,M}
    Minvs::NTuple{3,M}
    "Inverse collocation matrices, for the value → coefficient change."
    Sinvs::NTuple{3,M}
    invλsum::A3
    work1::A3
    work2::A3
    scratch::NTuple{3,A3}
    scratch_inner::A3
    "Size of the interior problem."
    interior::NTuple{3,Int}
    """Lookup tables of [`locate`](@ref), for the cloud-in-cell deposition: the
       three axes share a geometry, so one `x₀`, one width and one cell table."""
    loc_x0::E
    loc_invwidth::E
    loc_cells::VI
    "Dual length of each node, per direction: what turns a count into a density."
    duals_len::NTuple{3,V}
end

function DeviceMesh(backend, ::Type{E}, mesh::SplineMesh{3,T}) where {E,T}
    function dev(x)
        a = KernelAbstractions.zeros(backend, E, size(x)...)
        copyto!(a, E.(x))
        a
    end
    s = mesh.solver
    function devi(x)
        a = KernelAbstractions.zeros(backend, Int32, size(x)...)
        copyto!(a, Int32.(x))
        a
    end
    tbl = mesh.locators[1]
    V = typeof(dev(mesh.axes[1].colloc))
    M = typeof(dev(mesh.laplacians[1]))
    A3 = typeof(dev(mesh.scratch[1]))
    VI = typeof(devi(tbl.cell))
    DeviceMesh{E,V,M,A3,VI}(
        map(d -> map(dev, d), mesh.dual_moments),
        dev(mesh.moment_partials),
        map(ax -> dev(ax.colloc), mesh.axes),
        map(ax -> dev(ax.knots), mesh.axes),
        map(dev, mesh.laplacians),
        map(o -> dev(o.M), s.ops),
        map(o -> dev(o.Minv), s.ops),
        map(cm -> dev(cm.Sinv), mesh.collocation),
        dev(s.invλsum), dev(s.work1), dev(s.work2),
        map(dev, mesh.scratch),
        dev(mesh.scratch_inner),
        size(mesh),
        E(tbl.x0), E(tbl.invwidth), devi(tbl.cell),
        map(ax -> dev(dual_lengths(ax)), mesh.axes))
end

"""
    deposit_cic!(ρ, dm, acc, npart, charge) -> ρ

Cloud-in-cell deposition of the packed particles onto this mesh — see
[`_deposit_cic_kernel!`](@ref). `acc` supplies `(knode, delta)`, so the cloud
itself need not be on the device.
"""
function deposit_cic!(ρ::AbstractArray{E,3}, dm::DeviceMesh{E}, acc, npart,
                      charge, x0f, hf) where {E}
    gx, gy, gz = dm.colloc
    wx, wy, wz = dm.duals_len
    fill!(ρ, zero(E))
    _deposit_cic_kernel!(acc.backend)(
        ρ, acc.knode.device, acc.delta.device, gx, gy, gz, dm.loc_cells,
        E(x0f), E(hf), dm.loc_x0, dm.loc_invwidth, Int32(npart); ndrange = npart)
    synchronize(acc.backend)
    ρ .*= E(charge) ./ (wx .* wy' .* reshape(wz, 1, 1, :))
    ρ
end

Base.size(dm::DeviceMesh) = dm.interior

# --- the chain, each line delegating to the half that takes tables ----------

total_charge(ρ::AbstractArray, dm::DeviceMesh) =
    _total_charge(ρ, map(m -> m[1], dm.duals), dm.partials)

multipole(ρ::AbstractArray, dm::DeviceMesh{E}) where {E} =
    _multipole(E, all_moments(ρ, map(m -> m[1], dm.duals), map(m -> m[2], dm.duals),
                              map(m -> m[3], dm.duals), dm.partials))

boundary_potential!(φ::AbstractArray, dm::DeviceMesh, mp::Multipole) =
    _boundary_potential!(φ, mp, dm.colloc)

poisson_rhs!(rhs::AbstractArray, ρ::AbstractArray, dm::DeviceMesh,
             φ::AbstractArray) =
    _poisson_rhs!(rhs, ρ, φ, dm.laplacians, dm.interior)

spline_coefficients!(c::AbstractArray, ρ::AbstractArray, dm::DeviceMesh) =
    apply_all_rotating!(c, dm.Sinvs, ρ, dm.scratch[3])

boundary_from_coarse!(φ::AbstractArray, dm::DeviceMesh, coarse::DeviceMesh,
                      csol_coarse::AbstractArray) =
    _boundary_from_coarse!(φ, csol_coarse, coarse.knots, dm.colloc)

function solve_interior!(φ::AbstractArray, ρ::AbstractArray, dm::DeviceMesh)
    rhs = poisson_rhs!(dm.scratch_inner, ρ, dm, φ)
    _solve!(rhs, rhs, dm.Minvs, dm.Ms, dm.invλsum, dm.work1, dm.work2)
    # ⚠️ A kernel, not `@views φ[2:end-1, …] .= rhs` — see
    # [`_fill_interior_kernel!`](@ref). The broadcast into a non-contiguous view
    # cost 25 ms per call against this kernel's 0.5, and the two of them were
    # 30 % of the whole nested solve.
    backend = get_backend(φ)
    _fill_interior_kernel!(backend)(φ, rhs; ndrange = size(rhs))
    synchronize(backend)
    φ
end

function poisson!(φ::AbstractArray, ρ::AbstractArray, dm::DeviceMesh)
    boundary_potential!(φ, dm, multipole(ρ, dm))
    solve_interior!(φ, ρ, dm)
end

"""
    poisson!(φs, ρs, dms::NTuple{L,DeviceMesh}) -> φs

The nested solve, on the device. Same order as the host version: the coarsest
level takes its conditions from its own multipole expansion, each finer one
reads them from the level above.
"""
function poisson!(φs::NTuple{L,A}, ρs::NTuple{L,A},
                  dms::NTuple{L,<:DeviceMesh}) where {L,A<:AbstractArray}
    poisson!(φs[L], ρs[L], dms[L])
    for l in (L-1):-1:1
        coefs = spline_coefficients!(dms[l+1].scratch[2], φs[l+1], dms[l+1])
        boundary_from_coarse!(φs[l], dms[l], dms[l+1], coefs)
        solve_interior!(φs[l], ρs[l], dms[l])
    end
    φs
end

function effective_potential!(csol::AbstractArray{E,3}, ρ::AbstractArray{E,3},
                              dm::DeviceMesh{E}, jel::Jellium) where {E}
    gx, gy, gz = dm.colloc
    extra = dm.scratch[1]
    backend = get_backend(extra)
    _effective_potential_kernel!(backend)(extra, ρ, gx, gy, gz,
                                          Jellium(E(jel.nions), E(jel.radius));
                                          ndrange = size(extra))
    synchronize(backend)
    csol .+= spline_coefficients!(dm.scratch[2], extra, dm)
end
