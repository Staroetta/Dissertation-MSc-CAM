using Distributed
addprocs(15)
@everywhere begin
    using LinearAlgebra, StatsBase, Serialization, Distributions, LaTeXStrings, SharedArrays, SpecialFunctions, Integrals
    include("../AdL.jl"); include("../Ad2L.jl"); include("../force.jl");
end

function save_variable(var, file_name)
    variable = copy(var)
    name = string(file_name, ".jls")
    open(name, "w") do file
        serialize(file, variable)
    end
end

@everywhere function bias_MAE(q_traj; k=1)
    if isnan(q_traj[1,end]) || isinf(q_traj[1,end])
        return Inf
    end
    tar_dist = Normal(0, 1)

    x1 = range(minimum(q_traj[1,:]), maximum(q_traj[1,:]), length=100)
    edge_q = collect(x1)
    hist_q = fit(Histogram, q_traj[1,:], edge_q)
    bin_centers_q = (edge_q[1:end-1] .+ edge_q[2:end]) ./ 2
    bin_widths_q = diff(edge_q)
    total_counts_q = sum(hist_q.weights)
    hist_density_q = hist_q.weights ./ (total_counts_q .* bin_widths_q)

    q_dist = pdf(tar_dist, bin_centers_q)
    errors_q = q_dist - hist_density_q
    MAE_q = mean(abs.(errors_q))
    # q_dist = pdf.(tar_dist, bin_centers_q)
    # errors_q = q_dist .* bin_widths_q - hist_density_q
    # MAE_q = mean(abs.(errors_q))
    return MAE_q
end

@everywhere begin
    # setting potential and force
    q_pot = pot_mho
    q_force = force_mho
    q_monitor = monitor_mho
    q_monitor_prime = monitor_prime_mho

    h_lst = [0.202, 0.216, 0.231, 0.247, 0.264, 0.282, 0.301, 0.322, 0.344, 0.368]
    repeats = 60; num_cores = nprocs()
    t = 100000; m = 0.1; M = 1.1
    sigma_A = 1.0; mu = 1.0; beta = 1.0; Nd = 1.0; γ = 1.0
    obs = 0.5
end

bias_matrix1 = SharedArray{Float64}(length(h_lst), repeats); g_avg_matrix1 = SharedArray{Float64}(length(h_lst), repeats)
bias_matrix2 = SharedArray{Float64}(length(h_lst), repeats); g_avg_matrix2 = SharedArray{Float64}(length(h_lst), repeats)
bias_matrix3 = SharedArray{Float64}(length(h_lst), repeats); g_avg_matrix3 = SharedArray{Float64}(length(h_lst), repeats)
bias_matrix4 = SharedArray{Float64}(length(h_lst), repeats); g_avg_matrix4 = SharedArray{Float64}(length(h_lst), repeats)

@sync for i in 1:length(h_lst)
    h = h_lst[i]; Nsteps = round(Int, t/h); idx = round(Int, 0.2 * Nsteps) + 1
    @sync @distributed for core in 1:num_cores
        for k in core:num_cores:repeats
            q0 = randn(1); p0 = randn(1); xi0 = randn()
            q_traj1, g_avg1 = run_Ad_BAOAB(q0, p0, γ, Nsteps, h, beta, m, M, q_force, q_monitor, q_monitor_prime)
            q_traj2 = run_BAOAB(q0, p0, γ, Nsteps, h, beta, q_force)
            q_traj3, g_avg3 = run_Ad_BADODAB(q0, p0, xi0, Nsteps, h, sigma_A, mu, beta, Nd, m, M, q_force, q_monitor, q_monitor_prime)
            q_traj4 = run_BADODAB(q0, p0, xi0, Nsteps, h, sigma_A, mu, beta, Nd, q_force)
            bias_matrix1[i,k] = bias_MAE(q_traj1[:,idx:end]); g_avg_matrix1[i,k] = g_avg1
            bias_matrix2[i,k] = bias_MAE(q_traj2[:,idx:end])
            bias_matrix3[i,k] = bias_MAE(q_traj3[:,idx:end]); g_avg_matrix3[i,k] = g_avg3
            bias_matrix4[i,k] = bias_MAE(q_traj4[:,idx:end])
        end
    end
end

save_variable(bias_matrix1, "bias_ASL_mho_MAE")
save_variable(g_avg_matrix1, "g_avg_ASL_mho_MAE")
# println("bias_ASL_mho_MAE")
# println(bias_matrix1)

save_variable(bias_matrix2, "bias_BAOAB_mho_MAE")
# println("bias_BAOAB_mho_MAE")
# println(bias_matrix)

save_variable(bias_matrix3, "bias_Ad2L_mho_MAE")
save_variable(g_avg_matrix3, "g_avg_Ad2L_mho_MAE")
# println("bias_Ad2L_mho_MAE")
# println(bias_matrix)

save_variable(bias_matrix4, "bias_AdL_mho_MAE")
# println("bias_AdL_mho_MAE")
# println(bias_matrix)

println("fin")