using LinearAlgebra

function A_step(q, p, xi, h)
    q = q + h * p
    return q
end

function B_step(q, p, xi, h, force)
    p = p + h * force
    return p
end

function O_step(q, p, xi, h, sigma_A)
    R = randn(size(p))
    if abs(xi) < 1e-12
        p = p - h * xi * p + sqrt(h) * sigma_A * R
    else
        alpha = exp(-h * xi)
        p = alpha * p + sigma_A * sqrt((1 - alpha^2) / (2 * xi)) .* R
    end
    return p
end

function run_BAOAB(q0, p0, γ, Nsteps, h, beta, q_force, noise1 = nothing, noise2 = nothing)
    # initialization
    q_traj = zeros(eltype(q0), (size(q0)..., Nsteps));
    q = copy(q0); p = copy(p0); t = 0.0
    force = q_force(q)
    xi = copy(γ); sigma_A = sqrt(2*γ*beta^(-1))

    for i in 1:Nsteps
        # BAOAB step
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

        t += h
        q_traj[:,i] = q;  # p_traj[:,i] = p; xi_traj[i] = xi; t_traj[i] = t
    end

    # return q_traj, p_traj, xi_traj, t_traj
    return q_traj
end;

function D_step(q, p, xi, h, mu, beta, Nd)
    xi = xi + (sum(p .* p) - Nd * beta^(-1)) * h / mu
    return xi
end

function run_BADODAB(q0, p0, xi0, Nsteps, h, sigma_A, mu, beta, Nd, q_force, noise1 = nothing, noise2 = nothing)
    # initialization
    q_traj = zeros(eltype(q0), (size(q0)..., Nsteps));
    # p_traj = zeros(dim,Nsteps); xi_traj = zeros(Nsteps); t_traj = zeros(Nsteps)
    q = copy(q0); p = copy(p0); xi = copy(xi0); t = 0.0
    force = q_force(q)

    for i in 1:Nsteps
        # BADODAB step
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

        t += h
        q_traj[:,i] = q; 
        # p_traj[:,i] = p; xi_traj[i] = xi; t_traj[i] = t
    end

    # return q_traj, p_traj, xi_traj, t_traj
    return q_traj
end;

function P_step(q, p, xi, h, sigma_A, force)
    R = randn(length(p))
    p = p + h * force - h * xi * p + sqrt(h) * sigma_A * R
    return p
end

function run_PAD(q0, p0, xi0, Nsteps, h, sigma_A, mu, beta, Nd, q_force)
    # initialization
    dim = 1
    q_traj = zeros(dim,Nsteps); p_traj = zeros(dim,Nsteps); xi_traj = zeros(Nsteps); t_traj = zeros(Nsteps)
    q = copy(q0); p = copy(p0); xi = copy(xi0); t = 0.0

    for i in 1:Nsteps
        force = q_force(q)
        p = P_step(q, p, xi, h, sigma_A, force)
        q = A_step(q, p, xi, h)
        xi = D_step(q, p, xi, h, mu, beta, Nd)
        t += h
        q_traj[:,i] = q; p_traj[:,i] = p; xi_traj[i] = xi; t_traj[i] = t
    end

    return q_traj, p_traj, xi_traj, t_traj
end;

