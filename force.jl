function pot_ho(q; k=1)
    U = q.^2 * k/2
    return mean(U)
end

function force_ho(q)
    # U(q) = q.^2 * k/2, k = 1
    F = -q 
    return F
end

function Tconf_ho(q)
    T = q .^ 2
    return mean(T)
end

function my_pot(q)
    U = (q.^2).*(sin.(q).^2 .+ 0.1)
    return mean(U)
end

function my_force(q)
    # U(q) = (q.^2).*(sin.(q).^2 .+ 0.1)
    F = -2 .* q .* (sin.(q).^2 .+ 0.1) - 2 .* q.^2 .* sin.(q) .* cos.(q)
    return F
end

function bias_pot(q_traj, q_pot::Function, obs::Float64)::Float64
    q_mean = mean(q_pot(q_traj))
    if isnan(q_mean) || isinf(q_mean)
        return Inf
    end
    return abs(q_mean - obs)
end

function bias_Tconf(q_traj, q_Tconf::Function, obs::Float64)::Float64
    return abs(obs - q_Tconf(q_traj))
end

function rosenbrock(x, a=1, b=100)
    return (a - x[1])^2 + b * (x[2] - x[1]^2)^2
end

function rosenbrock_force(x, a=1, b=100)
    df_dx1 = -2 * (a - x[1]) - 4 * b * x[1] * (x[2] - x[1]^2)
    df_dx2 = 2 * b * (x[2] - x[1]^2)
    return [-df_dx1, -df_dx2]
end

function U_ms(q::Matrix{Float64}, k::Float64=25.0, rc::Float64=1.0)
    N = size(q, 2)
    pot = 0.0
    grad = zeros(Float64, 3, N)
    lap = 0.0
    grad_sq = 0.0

    for i in 1:N
        for j in 1:N
            if i != j
                dr = q[:, i] - q[:, j]
                dr = dr .- round.(dr ./ 5.) .* 5.
                r = norm(dr)
                if r < rc
                    tmp = k * (r - rc)
                    pot += 0.25 * tmp * (r - rc)
                    grad[:, i] += tmp * dr / r
                    # lap += k * r^2 + tmp
                    lap += k + 2*tmp/r
                    # grad_sq += norm(tmp * dr) ^ 2
                end
            end
        end
        grad_sq += dot(grad[:, i], grad[:, i])
    end
    
    return pot, -grad, lap, grad_sq
end

function random_projection_matrix(rows, cols)
    matrix = zeros(Int, rows, cols)
    for i in 1:rows
        for j in 1:cols
            rand_num = rand()
            if rand_num <= 1/6
                matrix[i, j] = 1
            elseif rand_num <= 2/6
                matrix[i, j] = -1
            end
        end
    end
    matrix = matrix .* sqrt(3/cols)
    return matrix 
end

function grad_BLR(data, w, N, n)
    # 2nd model - Large Scale Bayesian Logistic Regression
    dim = size(data, 2) - 1
    x = data[:, 1:dim]
    y = data[:, end]

    ws = x * w  # weighted sum of the data
    a = exp.(- y .* ws)
    tmp = ((-y .* a ./ (1 .+ a))' * x)'
    return w + tmp .* (N/n)
end

function sigmoid(z)
    return 1.0 ./ (1.0 .+ exp.(-z))
end

function predict_data(features, weights)
    # Calculate the linear combination of features and weights
    z = first(features' * weights)
    # Apply the logistic function to get the probability
    prob = sigmoid(z)
    # Apply threshold (0.5) for binary classification
    pred = ifelse.(prob .>= 0.5, 1, -1)
    return pred
end

function predict_BLR(q_mean, x_test, y_test)
    tmp = sum(q_mean)
    if isnan(tmp) || isinf(tmp)
        return 0
    end
    predicted_labels = zeros(size(x_test, 1))
    for i in 1:size(x_test, 1)
        predicted_labels[i] = predict_data(x_test[i,:], q_mean)
    end
    accuracy = sum(predicted_labels .== y_test) / length(y_test)
    return accuracy
end

function log_likelihood_BLR(q_mean, x_test, y_test)
    tmp = y_test .* (x_test * q_mean)
    log_likelihood = -sum(log.(1 .+ exp.(-tmp)))
    return log_likelihood
end

function pot_V(x; a = 5, b = 0.1, x0 = 0.5, c = 0.1)
    res = (a^1.5 * b^0.5 * x0 * atan.(sqrt(a/b) * (x .- x0)) +
           (a * b * (a * x0 * (x .- x0) .- b)) ./ (a * (x .- x0).^2 .+ b) +
           c * (x .- x0).^2 + 2 * c * (x .- x0) * x0) * 0.5
    return res
end

function force_V(x; a = 5, b = 0.1, x0 = 0.5, c = 0.1)
    wx = b ./ (b/a .+ (x .- x0).^2)
    res = (wx.^2 .+ c) .* x
    return - res
end

function omega(x; a = 5, b = 0.1, x0 = 0.5, c = 0.1)
    wx = b ./ (b/a .+ (x .- x0).^2)
    return wx
end

function omega_prime(x; a = 5, b = 0.1, x0 = 0.5, c = 0.1)
    return -2 * b * (x .- x0) ./ ((b/a .+ (x .- x0).^2).^2)
end

function pot_mho(q; k=1)
    U = q.^2 * k/2
    return mean(U)
end

function force_mho(q, s = 0.3; k=1)
    # U(q) = q.^2 * k/2, k = 1
    F = -q + (1.0 ./ (q.^2 .+ s)) * randn()
    return F
end

function monitor_mho(q, s = 1.0; k=1, eps=7.0)
    x = first(q)
    tmp = (eps / (x^2 + s))^2
    return tmp
end

function monitor_prime_mho(q, s = 1.0; k=1, eps=7.0)
    tmp = -4 * (eps^2) * q ./ (q.^2 .+ s).^3
    return tmp
end

# function monitor_mho(q, epsilon = 0.3; k=1)
#     x = first(q)
#     tmp = (k*x)^2 + (1 / (x^2 + epsilon)^2)
#     return sqrt(tmp)
# end

# function monitor_prime_mho(q, epsilon = 0.3; k=1)
#     tmp = q .* (k^2 .- 2 ./ (q.^2 .+ epsilon).^3)
#     return tmp / monitor_mho(q, epsilon; k=k)
# end