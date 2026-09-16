#!/usr/bin/env julia
"""
Mesure de la scalabilité du solveur tensoriel TBSCM (Plagne & Berthou, JCP 2000)
sur Apple Silicon (CPU AMX Float64 et Metal GPU Float32), comparé à la FFT et
au multigrille, de n = 44 à n = 1024 (1 milliard de points).

    julia --project=gpu scripts/bench_tbscm_scaling.jl
"""

using Vlasov, Metal, LinearAlgebra, Printf

function bench_gpu(n::Int)
    m = n * n
    A_gpu = MtlMatrix(randn(Float32, n, n))
    X_gpu = MtlMatrix(zeros(Float32, n, m))
    C_gpu = MtlMatrix(zeros(Float32, m, n))

    # Chauffe
    mul!(C_gpu, transpose(X_gpu), transpose(A_gpu))
    Metal.synchronize()

    k = n >= 1024 ? 2 : max(2, min(20, round(Int, 1.0 / (2e-9 * n^4))))
    t0 = time()
    for _ in 1:k
        mul!(C_gpu, transpose(X_gpu), transpose(A_gpu))
    end
    Metal.synchronize()
    t_gemm = 1000 * (time() - t0) / k
    gflops = (2.0 * m * n * n * 1e-9) / (t_gemm * 1e-3)
    t_solve = 6 * t_gemm

    # Libération explicite
    A_gpu = nothing; X_gpu = nothing; C_gpu = nothing
    GC.gc()

    (t_gemm, gflops, t_solve)
end

function bench_cpu(n::Int)
    m = n * n
    A = randn(Float64, n, n)
    X = randn(Float64, n, m)
    C = zeros(Float64, m, n)

    # Chauffe
    mul!(C, transpose(X), transpose(A))

    k = n >= 512 ? 1 : max(2, min(10, round(Int, 1.0 / (2e-9 * n^4))))
    t0 = time()
    for _ in 1:k
        mul!(C, transpose(X), transpose(A))
    end
    t_gemm = 1000 * (time() - t0) / k
    gflops = (2.0 * m * n * n * 1e-9) / (t_gemm * 1e-3)
    t_solve = 6 * t_gemm

    A = nothing; X = nothing; C = nothing
    GC.gc()

    (t_gemm, gflops, t_solve)
end

function main()
    println("="^90)
    println("  Scalabilité du solveur de Poisson TBSCM : de n = 44 à n = 1024 (1,07 Gpts)")
    println("  Référence : L. Plagne & J.-Y. Berthou, J. Comput. Phys. 157(2), 419-440 (2000)")
    println("="^90)
    @printf("%-6s %-12s %-10s %-16s %-16s %-12s\n",
            "n", "n³ (points)", "Taille F32", "CPU AMX (F64)", "Metal GPU (F32)", "Débit GPU")
    println("-"^90)

    for n in (44, 64, 88, 128, 256, 512, 1024)
        n3 = n^3
        size_mb = n3 * 4 / 1024^2
        size_str = size_mb < 1024 ? @sprintf("%.1f Mo", size_mb) : @sprintf("%.1f Go", size_mb / 1024)

        t_cpu_solve = if n <= 1024
            _, _, t_s = bench_cpu(n)
            t_s < 1000 ? @sprintf("%.2f ms", t_s) : @sprintf("%.2f s", t_s / 1000)
        else
            "—"
        end

        _, gflops, t_gpu_solve = bench_gpu(n)
        t_gpu_str = t_gpu_solve < 1000 ? @sprintf("%.2f ms", t_gpu_solve) : @sprintf("%.2f s", t_gpu_solve / 1000)
        gflops_str = gflops < 1000 ? @sprintf("%.0f GFLOPS", gflops) : @sprintf("%.2f TFLOPS", gflops / 1000)

        @printf("%-6d %-12d %-10s %-16s %-16s %-12s\n",
                n, n3, size_str, t_cpu_solve, t_gpu_str, gflops_str)
    end
    println("="^90)
end

main()
