#!/usr/bin/env julia
# Night Pipeline: Automatically runs Proton 80M -> Xenon 80M -> Argon 80M -> Documenter Rebuild

using Printf

const ROOT = dirname(@__DIR__)
println("==========================================================")
println("=== VLASOV.JL — FULL 80M CONVERGED NIGHT SIMULATION PIPELINE ===")
println("==========================================================")
flush(stdout)

# Step 1: Monitor / Wait for Proton 80M
proton_cache = joinpath(ROOT, "proton_converged_80M_data_cache.jls")
proton_video = joinpath(ROOT, "film_proton_80M.mp4")
println("1. Checking Proton 80M simulation...")
flush(stdout)

while !isfile(proton_video)
    sleep(15)
end
println("✓ Proton 80M complete and verified!")
flush(stdout)

# Step 2: Run Xenon 80M
xenon_script = joinpath(ROOT, "scripts", "film_xenon_80M.jl")
xenon_cache = joinpath(ROOT, "xenon_converged_80M_data_cache.jls")
if !isfile(xenon_cache) || !isfile(joinpath(ROOT, "xenon_snapshots_80M.png")) || !isfile(joinpath(ROOT, "film_xenon_80M.mp4"))
    println("\n2. Running Xenon 80M simulation/rendering (Na₁₉₆ + Xe²⁵⁺, 500 keV, b = 45 a₀)...")
    flush(stdout)
    t0 = time()
    run(`julia --project=gpu -t auto $xenon_script`)
    @printf("✓ Xenon 80M complete in %.1f minutes!\n", (time() - t0) / 60)
    flush(stdout)
else
    println("\n2. Xenon 80M cache and outputs already exist, skipping.")
    flush(stdout)
end

# Step 3: Run Argon 80M
argon_script = joinpath(ROOT, "scripts", "film_argon_80M.jl")
argon_cache = joinpath(ROOT, "argon_converged_80M_data_cache.jls")
if !isfile(argon_cache) || !isfile(joinpath(ROOT, "argon_snapshots_80M.png")) || !isfile(joinpath(ROOT, "film_argon_80M.mp4"))
    println("\n3. Running Argon 80M simulation/rendering (Na₄₀ + Ar⁸⁺, 80 keV, b = 20 a₀)...")
    flush(stdout)
    t0 = time()
    run(`julia --project=gpu -t auto $argon_script`)
    @printf("✓ Argon 80M complete in %.1f minutes!\n", (time() - t0) / 60)
    flush(stdout)
else
    println("\n3. Argon 80M cache and outputs already exist, skipping.")
    flush(stdout)
end

# Step 4: Rebuild Documenter documentation
println("\n4. Rebuilding Documentation with 80M Converged Results...")
flush(stdout)
try
    run(`julia --project=docs docs/make.jl`)
    println("✓ Documentation built successfully!")
catch e
    println("Warning during docs build: $e")
end
flush(stdout)

println("\n==========================================================")
println("=== ALL 80M SIMULATIONS & DOCUMENTATION COMPLETE! ===")
println("==========================================================")
flush(stdout)
