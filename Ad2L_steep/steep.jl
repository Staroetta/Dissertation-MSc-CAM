using Distributed
addprocs(15)
@everywhere begin
    using LinearAlgebra, StatsBase, Serialization, Distributions, LaTeXStrings, SharedArrays, SpecialFunctions #, QuadGK
    include("../AdL.jl"); include("../Ad2L.jl"); include("../force.jl");
end

function save_variable(var, file_name)
    variable = copy(var)
    name = string(file_name, ".jls")
    open(name, "w") do file
        serialize(file, variable)
    end
end

@everywhere function compute_bias_Ad_BAOAB(γ, Nsteps, h, beta, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, noise1 = nothing, noise2 = nothing)
    dim = 1; q0 = randn(dim); p0 = randn(dim); xi0 = randn()
    idx = round(Int, 0.2 * Nsteps)  # take last 80% samples
    g_avg = 0.0
    q = copy(q0); p = copy(p0); t = 0.0; total_pot = 0.0
    xi = copy(γ); sigma_A = sqrt(2*γ*beta^(-1))
    force = q_force(q)
    tmp = mean(q_monitor(q)); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * q_monitor_prime(q)

    for i in 1:Nsteps
        p = B_step_hat(q, p, xi, h/2, g, g_prime, force, beta)
        q = A_step_hat(q, p, xi, h/2, g, g_prime, tol = 1e-11); 
        tmp = mean(q_monitor(q)); g = psi(tmp, m, M); 
        g_avg += g
        p = O_step_hat(q, p, xi, h, g, g_prime, sigma_A)
        q = A_step_hat(q, p, xi, h/2, g, g_prime, tol = 1e-11); 
        tmp = mean(q_monitor(q)); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * q_monitor_prime(q)  # BAOAB step with Ad Stepsize
        g_avg += g
        if noise1 === nothing
            force = q_force(q)
        elseif noise2 === nothing
            force = q_force(q) + rand(noise1, dim)
        else
            force = q_force(q) + rand(noise1, dim) + rand(noise2, dim)
        end
        p = B_step_hat(q, p, xi, h/2, g, g_prime, force, beta)

        if i > idx
            total_pot += mean(q_pot(q))
        end
    end
    g_avg = g_avg / (2*Nsteps)

    total_pot /= (Nsteps * 0.8)
    if isnan(total_pot) || isinf(total_pot)
        return Inf
    end
    return abs(total_pot - obs), g_avg
end

@everywhere function compute_bias_Ad2L(Nsteps, h, sigma_A, mu, beta, Nd, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, noise1 = nothing, noise2 = nothing)
    dim = 1; q0 = randn(dim); p0 = randn(dim); xi0 = randn()
    idx = round(Int, 0.2 * Nsteps)  # take last 80% samples
    g_avg = 0.0
    q = copy(q0); p = copy(p0); xi = copy(xi0); t = 0.0; total_pot = 0.0
    force = q_force(q)
    tmp = mean(q_monitor(q)); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * q_monitor_prime(q)

    for i in 1:Nsteps
        p = B_step_hat(q, p, xi, h/2, g, g_prime, force, beta)
        q = A_step_hat(q, p, xi, h/2, g, g_prime, tol = 1e-11); 
        tmp = mean(q_monitor(q)); g = psi(tmp, m, M); 
        g_avg += g
        xi = D_step_hat(q, p, xi, h/2, g, g_prime, mu, beta, Nd)
        p = O_step_hat(q, p, xi, h, g, g_prime, sigma_A)
        xi = D_step_hat(q, p, xi, h/2, g, g_prime, mu, beta, Nd)
        q = A_step_hat(q, p, xi, h/2, g, g_prime, tol = 1e-11); 
        tmp = mean(q_monitor(q)); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * q_monitor_prime(q)
        g_avg += g
        if noise1 === nothing
            force = q_force(q)
        elseif noise2 === nothing
            force = q_force(q) + rand(noise1, dim)
        else
            force = q_force(q) + rand(noise1, dim) + rand(noise2, dim)
        end
        p = B_step_hat(q, p, xi, h/2, g, g_prime, force, beta)

        if i > idx
            total_pot += mean(q_pot(q))
        end
    end
    g_avg = g_avg / (2*Nsteps)

    total_pot /= (Nsteps * 0.8)
    if isnan(total_pot) || isinf(total_pot)
        return Inf
    end
    return abs(total_pot - obs), g_avg
end

@everywhere function compute_bias_BAOAB(γ, Nsteps, h, beta, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, noise1 = nothing, noise2 = nothing)
    dim = 1; q0 = randn(dim); p0 = randn(dim); xi0 = randn()
    idx = round(Int, 0.2 * Nsteps)  # take last 80% samples
    g_avg = 0.0
    q = copy(q0); p = copy(p0); t = 0.0; total_pot = 0.0
    xi = copy(γ); sigma_A = sqrt(2*γ*beta^(-1))
    force = q_force(q)

    for i in 1:Nsteps
        p = B_step(q, p, xi, h/2, force)
        q = A_step(q, p, xi, h/2)
        p = O_step(q, p, xi, h, sigma_A)
        q = A_step(q, p, xi, h/2)
        if noise1 === nothing
            force = q_force(q)
        elseif noise2 === nothing
            force = q_force(q) + rand(noise1, dim)
        else
            force = q_force(q) + rand(noise1, dim) + rand(noise2, dim)
        end
        p = B_step(q, p, xi, h/2, force)

        if i > idx
            total_pot += mean(q_pot(q))
        end
    end

    total_pot /= (Nsteps * 0.8)
    if isnan(total_pot) || isinf(total_pot)
        return Inf
    end
    return abs(total_pot - obs), g_avg
end

@everywhere function compute_bias_AdL(Nsteps, h, sigma_A, mu, beta, Nd, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, noise1 = nothing, noise2 = nothing)
    dim = 1; q0 = randn(dim); p0 = randn(dim); xi0 = randn()
    idx = round(Int, 0.2 * Nsteps)  # take last 80% samples
    g_avg = 0.0
    q = copy(q0); p = copy(p0); xi = copy(xi0); t = 0.0; total_pot = 0.0
    force = q_force(q)

    for i in 1:Nsteps
        p = B_step(q, p, xi, h/2, force)
        q = A_step(q, p, xi, h/2)
        xi = D_step(q, p, xi, h/2, mu, beta, Nd)
        p = O_step(q, p, xi, h, sigma_A)
        xi = D_step(q, p, xi, h/2, mu, beta, Nd)
        q = A_step(q, p, xi, h/2)
        if noise1 === nothing
            force = q_force(q)
        elseif noise2 === nothing
            force = q_force(q) + rand(noise1, dim)
        else
            force = q_force(q) + rand(noise1, dim) + rand(noise2, dim)
        end
        p = B_step(q, p, xi, h/2, force)

        if i > idx
            total_pot += mean(q_pot(q))
        end
    end

    total_pot /= (Nsteps * 0.8)
    if isnan(total_pot) || isinf(total_pot)
        return Inf
    end
    return abs(total_pot - obs), g_avg
end

@everywhere begin
    # setting potential and force
    a = 5; b = 0.1; x0 = 0.5; c = 0.1;
    q_pot = pot_V  # pot_ho or pot_V
    q_force = force_V  # force_ho or force_V
    q_monitor = omega  # x -> norm(q_force(x)) ; x -> mean(omega(x))
    q_monitor_prime = omega_prime  # x -> x / q_monitor(x) ; omega_prime

    epsilon = 1.0; r = 0.8; λ = 1    # λ_lst = [1, 3, 50]  Weibull: using k=λ to keep the scale
    Gaussian_noise = (1-r)*Normal()
    Weibull_mean = λ * gamma(1 + 1/λ)
    Weibull_noise = r * (Weibull(λ, λ) .- Weibull_mean)

    # h_lst = [0.1*1.1^n for n in 6:15]
    h_lst = [0.202, 0.216, 0.231, 0.247, 0.264, 0.282, 0.301, 0.322, 0.344, 0.368]
    repeats = 60; num_cores = nprocs()
    t = 100000; m = 0.1; M = 1.1
    sigma_A = 1.0; mu = 1.0; beta = 1.0; Nd = 1.0; γ = 1.0
    obs = -0.8054779535405598
end

bias_matrix1 = SharedArray{Float64}(length(h_lst), repeats); g_avg_matrix1 = SharedArray{Float64}(length(h_lst), repeats)
bias_matrix2 = SharedArray{Float64}(length(h_lst), repeats); g_avg_matrix2 = SharedArray{Float64}(length(h_lst), repeats)
bias_matrix3 = SharedArray{Float64}(length(h_lst), repeats); g_avg_matrix3 = SharedArray{Float64}(length(h_lst), repeats)
bias_matrix4 = SharedArray{Float64}(length(h_lst), repeats); g_avg_matrix4 = SharedArray{Float64}(length(h_lst), repeats)

# let
#     f(x) = q_pot(x) * exp(-q_pot(x))
#     g(x) = exp(-q_pot(x))
#     result1, error1 = quadgk(x -> f(x), -Inf, Inf)
#     result2, error2 = quadgk(x -> g(x), -Inf, Inf)
#     obs = result1 / result2
#     @everywhere obs = $obs
# end

@sync for i in 1:length(h_lst)
    h = h_lst[i]; Nsteps = round(Int, t/h)
    @sync @distributed for core in 1:num_cores
        for k in core:num_cores:repeats
            # bias, g_avg = compute_bias_Ad_BAOAB(γ, Nsteps, h, beta, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, epsilon*Weibull_noise, epsilon*Gaussian_noise)
            # bias, g_avg = compute_bias_BAOAB(γ, Nsteps, h, beta, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, epsilon*Weibull_noise, epsilon*Gaussian_noise)
            # bias, g_avg = compute_bias_Ad2L(Nsteps, h, sigma_A, mu, beta, Nd, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, epsilon*Weibull_noise, epsilon*Gaussian_noise)
            # bias, g_avg = compute_bias_AdL(Nsteps, h, sigma_A, mu, beta, Nd, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, epsilon*Weibull_noise, epsilon*Gaussian_noise)
            # bias_matrix[i,k] = mean(bias); g_avg_matrix[i,k] = g_avg
            bias1, g_avg1 = compute_bias_Ad_BAOAB(γ, Nsteps, h, beta, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, epsilon*Weibull_noise, epsilon*Gaussian_noise)
            bias2, g_avg2 = compute_bias_BAOAB(γ, Nsteps, h, beta, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, epsilon*Weibull_noise, epsilon*Gaussian_noise)
            bias3, g_avg3 = compute_bias_Ad2L(Nsteps, h, sigma_A, mu, beta, Nd, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, epsilon*Weibull_noise, epsilon*Gaussian_noise)
            bias4, g_avg4 = compute_bias_AdL(Nsteps, h, sigma_A, mu, beta, Nd, m, M, q_force, q_pot, q_monitor, q_monitor_prime, obs, epsilon*Weibull_noise, epsilon*Gaussian_noise)
            bias_matrix1[i,k] = mean(bias1); g_avg_matrix1[i,k] = g_avg1
            bias_matrix2[i,k] = mean(bias2); g_avg_matrix2[i,k] = g_avg2
            bias_matrix3[i,k] = mean(bias3); g_avg_matrix3[i,k] = g_avg3
            bias_matrix4[i,k] = mean(bias4); g_avg_matrix4[i,k] = g_avg4
        end
    end
end

save_variable(bias_matrix1, "bias_ASL_λ="*string(λ))
save_variable(g_avg_matrix1, "g_avg_ASL_λ="*string(λ))
# println("bias_ASL_λ="*string(λ))
# println(bias_matrix)

save_variable(bias_matrix2, "bias_BAOAB_λ="*string(λ))
# println("bias_BAOAB_λ="*string(λ))
# println(bias_matrix)

save_variable(bias_matrix3, "bias_Ad2L_λ="*string(λ))
save_variable(g_avg_matrix3, "g_avg_Ad2L_λ="*string(λ))
# println("bias_Ad2L_λ="*string(λ))
# println(bias_matrix)

save_variable(bias_matrix4, "bias_AdL_λ="*string(λ))
# println("bias_AdL_λ="*string(λ))
# println(bias_matrix)

println("fin")