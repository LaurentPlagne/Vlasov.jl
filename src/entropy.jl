"""
    Classical entropy and relaxation diagnostics for isolated clusters.

Subject of Chapter 4: in exact Vlasov dynamics, Liouville's theorem preserves
phase-space volume and hence the Gibbs-Boltzmann entropy S(t) = -k ∫ f ln(f/f₀).
Finite pseudo-particle sampling introduces artificial binary collision noise
that drives relaxation from the initial Thomas-Fermi distribution towards
the classical Boltzmann distribution f ∝ exp(-β h) maximizing entropy.
"""

"""
    density_of_states(gridrad, potrad, energy_grid) -> Vector{Float64}

Computes the density of states g(ε) for a spherically symmetric potential V(r):
    g(ε) = (√2 / π²) ∫_{V(r) < ε} 4π r² √(ε - V(r)) dr
"""
function density_of_states(gridrad::AbstractVector{T}, potrad::AbstractVector{T},
                           energy_grid::AbstractVector{T}) where {T<:AbstractFloat}
    ng = length(energy_grid)
    dos = zeros(T, ng)
    coef = sqrt(T(2)) / (T(π)^2)

    for j in 1:(length(gridrad) - 1)
        rinf = gridrad[j]
        rsup = gridrad[j + 1]
        poteinf = potrad[j]
        potesup = potrad[j + 1]
        dr_half = T(0.5) * (rsup - rinf)
        coefinf = 4 * T(π) * rinf^2
        coefsup = 4 * T(π) * rsup^2

        for i in 1:ng
            ε = energy_grid[i]
            finf = ε > poteinf ? coefinf * sqrt(ε - poteinf) : zero(T)
            fsup = ε > potesup ? coefsup * sqrt(ε - potesup) : zero(T)
            dos[i] += dr_half * (finf + fsup)
        end
    end
    dos .*= coef
    dos
end

"""
    occupation_number(cloud, pot_interp, energy_grid, dos) -> (nocc, energies)

Computes single-particle energies ε_i = v_i²/2 + V(r_i) and the occupation
numbers n(ε_k) across `energy_grid`.
"""
function occupation_number(cloud::ParticleCloud{T}, pot_interp,
                           energy_grid::AbstractVector{T},
                           dos::AbstractVector{T}; dt::Real = 1.0) where {T<:AbstractFloat}
    npart = length(cloud.positions)
    w = cloud.weight
    ng = length(energy_grid)
    dε = energy_grid[2] - energy_grid[1]
    counts = zeros(Int, ng)

    energies = Vector{T}(undef, npart)
    inv_dt2 = inv(T(dt)^2)
    @inbounds for i in 1:npart
        r = sqrt(sum(abs2, cloud.positions[i]))
        v2 = sum(abs2, cloud.positions[i] .- cloud.previous[i]) * inv_dt2
        vr = pot_interp(r)
        ε = v2 / 2 + vr
        energies[i] = ε

        # Find energy bin
        k = round(Int, (ε - energy_grid[1]) / dε) + 1
        if 1 <= k <= ng
            counts[k] += 1
        end
    end

    nocc = zeros(T, ng)
    for k in 1:ng
        if dos[k] > 1e-8
            nocc[k] = (w * counts[k]) / (dos[k] * dε)
        end
    end
    (nocc, energies)
end

"""
    entropy_from_occupation(energy_grid, dos, nocc) -> Float64

Evaluates the entropy S = -∫ g(ε) n(ε) ln(n(ε)) dε.
"""
function entropy_from_occupation(energy_grid::AbstractVector{T},
                                 dos::AbstractVector{T},
                                 nocc::AbstractVector{T}) where {T<:AbstractFloat}
    dε = energy_grid[2] - energy_grid[1]
    s = zero(T)
    for k in eachindex(nocc)
        n = nocc[k]
        if n > 1e-12
            s -= dos[k] * n * log(n) * dε
        end
    end
    s
end

"""
    boltzmann_entropy(nelectrons, total_energy, energy_grid, dos) -> Float64

Computes the maximum (Boltzmann) entropy for a gas with given number of electrons
and total energy:
    n_B(ε) = exp(-α - β ε)
subject to:
    ∫ g(ε) n_B(ε) dε = N_e
    ∫ ε g(ε) n_B(ε) dε = E_tot
"""
function boltzmann_entropy(nelectrons::T, total_energy::T,
                           energy_grid::AbstractVector{T},
                           dos::AbstractVector{T}) where {T<:AbstractFloat}
    dε = energy_grid[2] - energy_grid[1]

    mean_energy_target = total_energy / nelectrons

    f_beta(β) = begin
        w = [dos[k] * exp(-β * energy_grid[k]) for k in eachindex(energy_grid)]
        sum_w = sum(w)
        sum_w <= 0 && return 1e10
        sum_ew = sum(energy_grid[k] * w[k] for k in eachindex(energy_grid))
        sum_ew / sum_w - mean_energy_target
    end

    # Bisection search for β > 0
    β_lo, β_hi = zero(T), T(100.0)
    for _ in 1:60
        β_mid = (β_lo + β_hi) / 2
        if f_beta(β_mid) > 0
            β_lo = β_mid
        else
            β_hi = β_mid
        end
    end
    β = (β_lo + β_hi) / 2

    # Normalization α
    sum_w = sum(dos[k] * exp(-β * energy_grid[k]) * dε for k in eachindex(energy_grid))
    exp_minus_α = nelectrons / sum_w
    α = -log(max(exp_minus_α, eps(T)))

    # S_Boltz = - ∫ g n_B ln n_B = ∫ g n_B (α + β ε) = α N_e + β E_tot
    α * nelectrons + β * total_energy
end

"""
    phase_space_entropy(cloud; rmax = 30.0, pmax = 2.5, nbins = 50) -> Float64

Computes classical Boltzmann entropy S = - ∫ f ln(f/f₀) d³r d³p directly by
spherical phase-space binning (r = ‖r⃗‖, p = ‖p⃗‖):
    f(r_i, p_j) = (w · N_{ij}) / [ (4π/3 Δ(r³)) (4π/3 Δ(p³)) ]
"""
function phase_space_entropy(cloud::ParticleCloud{T};
                             rmax::Real = 30.0, pmax::Real = 2.5,
                             nbins::Int = 50, dt::Real = 1.0) where {T<:AbstractFloat}
    dr = T(rmax) / nbins
    dp = T(pmax) / nbins
    w = cloud.weight
    f0 = T(1) / (4 * T(π)^3) # 2 / (2π)³ in atomic units
    inv_dt = inv(T(dt))

    counts = zeros(Int, nbins, nbins)
    @inbounds for i in eachindex(cloud.positions)
        r = sqrt(sum(abs2, cloud.positions[i]))
        p = sqrt(sum(abs2, cloud.positions[i] .- cloud.previous[i])) * inv_dt
        ir = floor(Int, r / dr) + 1
        ip = floor(Int, p / dp) + 1
        if 1 <= ir <= nbins && 1 <= ip <= nbins
            counts[ir, ip] += 1
        end
    end

    s = zero(T)
    coef = (4 * T(π) / 3)^2
    for ip in 1:nbins
        p_inf = (ip - 1) * dp
        p_sup = ip * dp
        vol_p = (p_sup^3 - p_inf^3)
        for ir in 1:nbins
            c = counts[ir, ip]
            c == 0 && continue
            r_inf = (ir - 1) * dr
            r_sup = ir * dr
            vol_r = (r_sup^3 - r_inf^3)
            vol_d6 = coef * vol_r * vol_p
            f = (w * c) / vol_d6
            if f > 1e-30
                # Integral contribution: - f ln(f/f₀) d³r d³p
                s -= f * log(f / f0) * vol_d6
            end
        end
    end
    s
end
