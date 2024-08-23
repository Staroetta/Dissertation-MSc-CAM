using Distributed
addprocs(15)
@everywhere begin
    using LinearAlgebra, Random, StatsBase, Serialization, Distributions, LaTeXStrings, SharedArrays
    include("../AdL.jl"); include("../Ad2L.jl"); include("../force.jl");
end

function save_variable(var, file_name)
    variable = copy(var)
    name = string(file_name, ".jls")
    open(name, "w") do file
        serialize(file, variable)
    end
end

@everywhere function run_predict_ASL(Nsteps, h, γ, beta, Samples, m, M; traj_len=50)
    dim = size(Samples, 2) - 1
    q0 = zeros(dim); p0 = randn(dim)/10; q = copy(q0); p = copy(p0)
    xi = copy(γ); sigma_A = sqrt(2*γ/beta)
    q_mean = zeros(dim); q_mean_traj = zeros(dim, traj_len); g_avg_traj = zeros(traj_len); g_avg = 0.0
    bi_idx = 0  # no burn-in
    rec = (Nsteps-bi_idx) ÷ traj_len; p_num = 0

    n = 50; N = size(Samples, 1)  # initial case
    idx = randperm(N)[1:n]; data = Samples[idx,:]; force = -grad_BLR(data, q, N, n)
    # tmp, tmp_prime = get_monitor_prime(q, data); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * tmp_prime
    tmp = sqrt(N)/n; g = psi(tmp, m, M); g_prime = zeros(dim)

    for i in 1:Nsteps
        p = B_step_hat(q, p, xi, h, g, g_prime, force, beta)
        q = A_step(q, p, xi, g*h)  # A_step_hat(q, p, xi, h, g, g_prime, data)
        # tmp = get_monitor(q, data); g = psi(tmp, m, M)
        g_avg += g
        p = O_step_hat(q, p, xi, h, g, g_prime, sigma_A)
        q = A_step(q, p, xi, g*h)  # A_step_hat(q, p, xi, h, g, g_prime, data)
        n = cyclic_n(i - bi_idx)
        idx = randperm(N)[1:n]; data = Samples[idx,:]; force = -grad_BLR(data, q, N, n)
        # tmp, tmp_prime = get_monitor_prime(q, data); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * tmp_prime
        tmp = sqrt(N)/n; g = psi(tmp, m, M)
        g_avg += g
        p = B_step_hat(q, p, xi, h, g, g_prime, force, beta)

        if i > bi_idx
            q_mean += q
            if i % rec == 0
                p_num += 1
                q_mean_traj[:,p_num] = q_mean ./ (i - bi_idx)
                g_avg_traj[p_num] = g_avg / (i*2)
                if p_num == traj_len
                    break
                end
            end
        end
    end
    return q_mean_traj, g_avg_traj
end

@everywhere function run_predict_BAOAB(Nsteps, h, γ, beta, Samples; traj_len=50)
    N = size(Samples, 1); dim = size(Samples, 2) - 1
    q0 = zeros(dim); p0 = randn(dim)/10; q = copy(q0); p = copy(p0)
    xi = copy(γ); sigma_A = sqrt(2*γ/beta)
    q_mean = zeros(dim); q_mean_traj = zeros(dim, traj_len)
    bi_idx = 0  # no burn-in
    rec = (Nsteps-bi_idx) ÷ traj_len; p_num = 0
    n = 50; idx = randperm(N)[1:n]; data = Samples[idx,:]; force = -grad_BLR(data, q, N, n); 

    for i in 1:Nsteps
        p = B_step(q, p, xi, h/2, force)
        q = A_step(q, p, xi, h/2)
        p = O_step(q, p, xi, h, sigma_A)
        q = A_step(q, p, xi, h/2)
        n = cyclic_n(i - bi_idx)
        idx = randperm(N)[1:n]; data = Samples[idx,:]; force = -grad_BLR(data, q, N, n)
        p = B_step(q, p, xi, h/2, force)
        if i > bi_idx
            q_mean += q
            if i % rec == 0
                p_num += 1
                q_mean_traj[:,p_num] = q_mean ./ (i - bi_idx)
                if p_num == traj_len
                    break
                end
            end
        end
    end
    return q_mean_traj
end

@everywhere function run_predict_Ad2L(Nsteps, h, sigma_A, mu, beta, Nd, Samples, m, M; traj_len=50)
    N = size(Samples, 1); dim = size(Samples, 2) - 1
    q0 = zeros(dim); p0 = randn(dim)/10; xi0 = randn() 
    q = copy(q0); p = copy(p0); xi = copy(xi0)
    q_mean = zeros(dim); q_mean_traj = zeros(dim, traj_len); g_avg_traj = zeros(traj_len); g_avg = 0.0
    bi_idx = 0  # no burn-in
    rec = (Nsteps-bi_idx) ÷ traj_len; p_num = 0
    
    n = 50; N = size(Samples, 1)  # initial case
    idx = randperm(N)[1:n]; data = Samples[idx,:]; force = -grad_BLR(data, q, N, n)
    # tmp, tmp_prime = get_monitor_prime(q, data); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * tmp_prime
    tmp = sqrt(N)/n; g = psi(tmp, m, M); g_prime = zeros(dim)

    for i in 1:Nsteps
        p = B_step_hat(q, p, xi, h, g, g_prime, force, beta)
        q = A_step(q, p, xi, g*h)  # A_step_hat(q, p, xi, h, g, g_prime, data)
        # tmp = get_monitor(q, data); g = psi(tmp, m, M)
        g_avg += g
        xi = D_step_hat(q, p, xi, h, g, g_prime, mu, beta, Nd)
        p = O_step_hat(q, p, xi, h, g, g_prime, sigma_A)
        xi = D_step_hat(q, p, xi, h, g, g_prime, mu, beta, Nd)
        q = A_step(q, p, xi, g*h)  # A_step_hat(q, p, xi, h, g, g_prime, data)
        n = cyclic_n(i - bi_idx)
        idx = randperm(N)[1:n]; data = Samples[idx,:]; force = -grad_BLR(data, q, N, n)
        # tmp, tmp_prime = get_monitor_prime(q, data); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * tmp_prime
        tmp = sqrt(N)/n; g = psi(tmp, m, M)
        g_avg += g
        p = B_step_hat(q, p, xi, h, g, g_prime, force, beta)

        if i > bi_idx
            q_mean += q
            if i % rec == 0
                p_num += 1
                q_mean_traj[:,p_num] = q_mean ./ (i - bi_idx)
                g_avg_traj[p_num] = g_avg / (i*2)
                if p_num == traj_len
                    break
                end
            end
        end
    end
    return q_mean_traj, g_avg_traj
end

@everywhere function run_predict_AdL(Nsteps, h, sigma_A, mu, beta, Nd, Samples; traj_len=50)
    N = size(Samples, 1); dim = size(Samples, 2) - 1
    q0 = zeros(dim); p0 = randn(dim)/10; xi0 = randn() 
    q = copy(q0); p = copy(p0); xi = copy(xi0)
    q_mean = zeros(dim); q_mean_traj = zeros(dim, traj_len)
    bi_idx = 0  # no burn-in
    rec = (Nsteps-bi_idx) ÷ traj_len; p_num = 0
    n = 50; idx = randperm(N)[1:n]; data = Samples[idx,:]; force = -grad_BLR(data, q, N, n)

    for i in 1:Nsteps
        p = B_step(q, p, xi, h/2, force)
        q = A_step(q, p, xi, h/2)
        xi = D_step(q, p, xi, h/2, mu, beta, Nd)
        p = O_step(q, p, xi, h, sigma_A)
        xi = D_step(q, p, xi, h/2, mu, beta, Nd)
        q = A_step(q, p, xi, h/2)
        n = cyclic_n(i - bi_idx)
        idx = randperm(N)[1:n]; data = Samples[idx,:]; force = -grad_BLR(data, q, N, n)
        p = B_step(q, p, xi, h/2, force)
        if i > bi_idx
            q_mean += q
            if i % rec == 0
                p_num += 1
                q_mean_traj[:,p_num] = q_mean ./ (i - bi_idx)
                if p_num == traj_len
                    break
                end
            end
        end
    end
    return q_mean_traj
end

function preprocess_data(data_x, data_y, rpm)
    input_dim = size(data_x, 1) * size(data_x, 2)
    ind = findall(x -> x == 7 || x == 9, data_y)

    y_data = data_y[ind]
    y_data = ifelse.(y_data .== 7, 1, -1)
    x_data = transpose(reshape(data_x, input_dim, size(data_x, 3))[:, ind])
    x_data = x_data * rpm
    return x_data, y_data
end

using MLDatasets

dim = 100
train_x, train_y = MNIST(split=:train)[:]
test_x, test_y = MNIST(split=:test)[:]
input_dim = size(train_x, 1) * size(train_x, 2)
rpm = random_projection_matrix(input_dim, dim)

train = hcat(preprocess_data(train_x, train_y, rpm)...)
x_test, y_test = preprocess_data(test_x, test_y, rpm)

@everywhere begin
    global train = $train
    global x_test = $x_test
    global y_test = $y_test
    # cyclic_n = i -> 50 + mod(i, 1000)  # mod(i, 200) < 100 ? 50 : 500
    function cyclic_n(i, max_value=500, base_value=50)
        mod_i = mod(i - 1, 2 * max_value) + 1
        if mod_i <= max_value
            n = mod_i
        else
            n = 2 * max_value - mod_i + 1
        end
        return n + base_value
    end

    repeats = 60; num_cores = nprocs()
    t = 5; h = 0.0001; Nsteps = round(Int, t/h); m = 0.5; M = 1.8
    sigma_A = 9.0; mu = 10; γ = 100.0; beta = 1.0; Nd = dim
    traj_len = 100
end

log_likelihood_matrix1 = SharedArray{Float64}(traj_len, repeats); g_avg_matrix1 = SharedArray{Float64}(traj_len, repeats)
log_likelihood_matrix2 = SharedArray{Float64}(traj_len, repeats);
log_likelihood_matrix3 = SharedArray{Float64}(traj_len, repeats); g_avg_matrix3 = SharedArray{Float64}(traj_len, repeats)
log_likelihood_matrix4 = SharedArray{Float64}(traj_len, repeats);

println("Number of processes: ", nprocs())
println("Worker process IDs: ", workers())

@sync @distributed for core in 1:num_cores
    for k in core:num_cores:repeats
        q_mean_traj1, g_avg_traj1 = run_predict_ASL(Nsteps, h, γ, beta, train, m, M; traj_len=traj_len)
        q_mean_traj2 = run_predict_BAOAB(Nsteps, h, γ, beta, train, traj_len = traj_len)
        q_mean_traj3, g_avg_traj3 = run_predict_Ad2L(Nsteps, h, sigma_A, mu, beta, Nd, train, m, M; traj_len=traj_len)
        q_mean_traj4 = run_predict_AdL(Nsteps, h, sigma_A, mu, beta, Nd, train, traj_len = traj_len)
        log_likelihood_matrix1[:,k] = [log_likelihood_BLR(q_mean_traj1[:,j], x_test, y_test) for j in 1:traj_len]
        log_likelihood_matrix2[:,k] = [log_likelihood_BLR(q_mean_traj2[:,j], x_test, y_test) for j in 1:traj_len]
        log_likelihood_matrix3[:,k] = [log_likelihood_BLR(q_mean_traj3[:,j], x_test, y_test) for j in 1:traj_len]
        log_likelihood_matrix4[:,k] = [log_likelihood_BLR(q_mean_traj4[:,j], x_test, y_test) for j in 1:traj_len]
        g_avg_matrix1[:,k] = g_avg_traj1; g_avg_matrix3[:,k] = g_avg_traj3
    end
end

save_variable(log_likelihood_matrix1, "log_likelihood_ASL_h="*string(h)); save_variable(g_avg_matrix1, "g_avg_ASL_BLR_h="*string(h))
save_variable(log_likelihood_matrix2, "log_likelihood_BAOAB_h="*string(h));
save_variable(log_likelihood_matrix3, "log_likelihood_Ad2L_h="*string(h)); save_variable(g_avg_matrix1, "g_avg_Ad2L_BLR_h="*string(h))
save_variable(log_likelihood_matrix4, "log_likelihood_AdL_h="*string(h));

println("fin")