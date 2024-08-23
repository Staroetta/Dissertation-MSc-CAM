using LinearAlgebra

function A_step_hat(q, p, xi, h, g, g_prime; tol = 1e-12, nmax = 100, count = false)
    q_new = q + g * h * p  # q_(n+1)^(0)
    for i in 1:nmax
        q_old = copy(q_new)
        mid = (q + q_new) / 2; g = psi(first(q_monitor(mid)), m, M)  
        q_new = q + g * h * p
        if norm(q_new - q_old) < tol
            # println("midpoint iteration converge in $i iterations")
            if count == true
                return q_new, i
            else
                return q_new
            end
        end
    end
    if count == true
        return q_new, nmax
    else
        return q_new
    end
end

function A_step_hat(q, p, xi, h, g, g_prime, data; tol = 1e-12, nmax = 100)
    q_new = q + g * h * p  # q_(n+1)^(0)
    for i in 1:nmax
        q_old = copy(q_new)
        mid = (q + q_new) / 2; g = psi(q_monitor(mid, data), m, M)  
        q_new = q + g * h * p
        if norm(q_new - q_old) < tol
            # println("midpoint iteration converge in $i iterations")
            return q_new
        end
    end
    return q_new
end

function B_step_hat(q, p, xi, h, g, g_prime, force, beta)
    p = p + h * (g * force .+ beta^(-1) * g_prime)
    return p
end

function O_step_hat(q, p, xi, h, g, g_prime, sigma_A)
    R = randn(size(p))
    if abs(xi) < 1e-12
        p = p - g * h * xi * p + sqrt(g * h) * sigma_A * R
    else
        alpha = exp(-g * h * xi)
        p = alpha * p + sigma_A * sqrt((1 - alpha^2) / (2 * xi)) .* R
    end
    return p
end

function D_step_hat(q, p, xi, h, g, g_prime, mu, beta, Nd)
    xi = xi + (sum(p .* p) - Nd * beta^(-1)) * h * g / mu
    return xi
end

function psi(x, m, M; alpha = 2, r = 1)
    u = r * x^(2*alpha)
    v = sqrt(1 + m^2 * u)
    return v / (v / M + sqrt(u))
end

function psi_prime(x, m, M; alpha = 2, r = 1)
    u = r * x^(2*alpha)
    v = sqrt(1 + m^2 * u)
    du_dx = 2*alpha * r * x^(2*alpha - 1)
    dv_dx = (m^2 * du_dx) / (2*v)
    tmp = (v/M + sqrt(u))^2
    dpsi_du = -v / (2 * sqrt(u) * tmp)
    dpsi_dv = sqrt(u) / tmp
    return dpsi_du * du_dx + dpsi_dv * dv_dx
end

function run_Ad_BAOAB(q0, p0, γ, Nsteps, h, beta, m, M, q_force, q_monitor, q_monitor_prime, noise1 = nothing, noise2 = nothing)
    # initialization
    q_traj = zeros(eltype(q0), (size(q0)..., Nsteps));
    # p_traj = zeros(eltype(q0), (size(q0)..., Nsteps)); 
    g_avg = 0.0
    q = copy(q0); p = copy(p0); t = 0.0
    xi = copy(γ); sigma_A = sqrt(2*γ*beta^(-1))
    force = q_force(q)
    tmp = mean(q_monitor(q)); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * q_monitor_prime(q)

    for i in 1:Nsteps
        p = B_step_hat(q, p, xi, h/2, g, g_prime, force, beta)
        q = A_step_hat(q, p, xi, h/2, g, g_prime); 
        tmp = mean(q_monitor(q)); g = psi(tmp, m, M);  # BAOAB step with Ad Stepsize
        g_avg += g
        p = O_step_hat(q, p, xi, h, g, g_prime, sigma_A)
        q = A_step_hat(q, p, xi, h/2, g, g_prime); 
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

        q_traj[:,i] = q
        # p_traj[:,i] = p
    end
    g_avg = g_avg / (2*Nsteps)
    # return q_traj, p_traj, g_avg
    return q_traj, g_avg
end;

function run_Ad_BADODAB(q0, p0, xi0, Nsteps, h, sigma_A, mu, beta, Nd, m, M, q_force, q_monitor, q_monitor_prime, noise1 = nothing, noise2 = nothing)
    # initialization
    q_traj = zeros(eltype(q0), (size(q0)..., Nsteps));
    # p_traj = zeros(eltype(q0), (size(q0)..., Nsteps)); 
    g_avg = 0.0
    q = copy(q0); p = copy(p0); xi = copy(xi0); t = 0.0
    force = q_force(q)
    tmp = mean(q_monitor(q)); g = psi(tmp, m, M); g_prime = psi_prime(tmp, m, M) * q_monitor_prime(q)

    for i in 1:Nsteps
        p = B_step_hat(q, p, xi, h/2, g, g_prime, force, beta)
        q = A_step_hat(q, p, xi, h/2, g, g_prime); 
        tmp = mean(q_monitor(q)); g = psi(tmp, m, M);
        g_avg += g
        xi = D_step_hat(q, p, xi, h/2, g, g_prime, mu, beta, Nd)
        p = O_step_hat(q, p, xi, h, g, g_prime, sigma_A)
        xi = D_step_hat(q, p, xi, h/2, g, g_prime, mu, beta, Nd)
        q = A_step_hat(q, p, xi, h/2, g, g_prime); 
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

        q_traj[:,i] = q
        # p_traj[:,i] = p
    end
    g_avg = g_avg / (2*Nsteps)
    # return q_traj, p_traj, g_avg
    return q_traj, g_avg
end;