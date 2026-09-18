"""
Metal backend.

Almost nothing, and that is the point. The kernels that used to live here —
smoothed field, sorted deposition, packing, their tables and their buffers,
four hundred lines of them — are now in `src/kernels.jl` and `src/accelerator.jl`,
written once for every backend `KernelAbstractions` supports. Measured against
what they replaced, at 4×10⁶ particles on a 134³ grid: the field kernel is
×0.999 and **bit for bit identical**, the deposition ×0.961.

What is left is what is genuinely *Apple*, and it is one thing: where the
buffers live.
"""
module VlasovMetalExt

using Vlasov
using Metal

import Vlasov: SplineAxis, GaussianSmoothing, DualBuffer, dual_buffer

"""
Zero-copy buffers on Apple Silicon.

`Metal.SharedStorage` places the array in memory that both the CPU and the GPU
address, so `unsafe_wrap` hands back an `Array` over the very same bytes.
`Vlasov.upload!` and `Vlasov.download!` then perform no copy at all, which is
the point: on unified memory the transfers the generic path would otherwise
make are pure waste.

⚠️ `download!` still **synchronises**, precisely because it no longer copies.
A version that only copied would order nothing here, and the host would read
buffers the device had not finished writing — a bug that looks like wrong
physics rather than a crash. See `Vlasov.download!`.
"""
function Vlasov.dual_buffer(::Metal.MetalBackend, ::Type{T},
                            dims::Integer...) where {T}
    mtl = MtlArray{T,length(dims),Metal.SharedStorage}(undef, dims...)
    host = unsafe_wrap(Array, mtl)
    fill!(host, zero(T))
    DualBuffer(mtl, host, true)
end

"""
    ForceAccelerator(MtlArray, fine, smoothing, npart, n)

The entry point the scripts already name. It builds the backend-agnostic
[`Vlasov.DeviceAccelerator`](@ref) on `MetalBackend()`, in `Float32` — Apple
GPUs having no double precision — so the scripts change nothing and gain the
portable kernels.
"""
Vlasov.ForceAccelerator(::Type{MtlArray}, fine::NTuple{3,SplineAxis{T}},
                        sm::GaussianSmoothing{T}, npart::Integer,
                        n::Integer) where {T} =
    Vlasov.DeviceAccelerator(Metal.MetalBackend(), Float32, fine, sm, npart, n)

end # module
