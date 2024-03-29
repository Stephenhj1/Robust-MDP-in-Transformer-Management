import numpy as np
import matplotlib.pyplot as plt

def element_product(f_j):
    """Compute f[j] * log(f[j]) for a given f[j], handling f[j] = 0."""
    if f_j > 0:
        return f_j * np.log(f_j)
    else:
        return 0.0

def compute_beta_max(f):
    """Compute beta_max for a given vector f."""
    assert np.sum(f) == 1 and np.all(f >= 0), "f must be a probability vector."
    beta_max = sum(element_product(f_j) for f_j in f)
    return beta_max

# def compute_beta_max(f):
#     # Computes the maximum likelihood value.
#     # f_modified = np.where(f == 0, 1e-10, f)  # Replace 0s with a small value to avoid log(0)
#     beta_max = np.sum( f * np.log(f) )
#     return beta_max

def compute_bounds(f, v, beta_max, beta):
    """Compute the lower bound mu_minus and upper bound mu_plus."""
    v_bar = np.dot(f, v)  # Compute the average of v under f
    mu_minus = np.max(v)
    
    mu_plus = ( np.max(v)-np.exp(beta-beta_max)* v_bar )/( 1-np.exp(beta-beta_max) )
    return mu_minus, mu_plus

def derivative_support_function(f, v, mu, beta):
    """Compute the partial derivative of function h which is the decision key of the Bisection Algorithm."""
    lambda_mu = 1 / np.sum( f / (mu-v) )
    partial_derivative_h = np.sum(f * np.log ( (lambda_mu * f) / (mu - v) ) ) - beta
    return partial_derivative_h

def bisection_algorithm(f, v, beta_max, beta, delta):
    """  Use Bisection Algorithm to find the optimal mu satisfying certain conditions.  """
    
    mu_minus, mu_plus = compute_bounds(f, v, beta_max, beta)
    
    while mu_plus - mu_minus > delta * (1 + mu_minus + mu_minus):
        mu = (mu_plus + mu_minus) / 2
        if derivative_support_function(f, v, mu, beta) > 0:
            mu_plus = mu
        else:
            mu_minus = mu
    return mu

def Robust_Dynamic_Programming(X, A, F_a, C, c_N, N):
    # Initialize value function for the final time step with terminal costs
    V = {i: c_N[i] for i in X}  # c_N is now directly a dictionary mapping states to terminal costs
    policy = {t: {i: None for i in X} for t in range(N)}  # Policy initialization

    # Loop over each time step in reverse
    for t in range(N-1, -1, -1):
        V_prev = V.copy()
        for i in X:
            # Calculate the cost for all actions and choose the min
            action_costs = {}
            for a in A:
                f = F_a[i-1, :]  # Extract frequency vector for action a and state s

                # action_cost = sum(P[i_prime][i][a] * (C[i][a] + V_prev[i_prime]) for i_prime in X)
                action_costs[a] = action_cost
            
            # Set the optimal value and policy
            optimal_action = min(action_costs, key=action_costs.get)
            V[i] = action_costs[optimal_action]
            policy[t][i] = optimal_action

    return policy, V

# Example usage (you need to define S, A, P, C, c_N, and N according to your specific problem)
# S = ['state1', 'state2', ...]
# A = ['action1', 'action2', ...]
# P = {'state1': {'state1': {'action1': prob, ...}, ...}, ...}
# C = {'state1': {'action1': cost, ...}, ...}
# c_N = {'state1': terminal_cost, ...}
# N = number_of_time_steps

# policy, V = backward_induction_cost(S, A, P, C, c_N, N)


# Main function
if __name__ == "__main__":
    # f = np.array([0.2, 0.1, 0.2, 0.4, 0.1])
    # v = np.array([1, 2, 7, 4, 3])
    # delta = 1e-5
    # beta_max = compute_beta_max(f)
    # print(f"beta_max is: {beta_max}")
    # beta = beta_max - 0.3
    # mu_minus, mu_plus = compute_bounds(f, v, beta_max, beta)
    # print(f"mu_minus is: {mu_minus},", f"mu_plus is: {mu_plus}")

    # mu = (mu_minus + mu_plus) / 2
    # derivative_sigma = derivative_support_function(f, v, mu, beta)
    # print(f"Derivative of the support function is: {derivative_sigma}")

    # mu = bisection_algorithm(f, v, beta_max, beta, delta)
    # print(f"Optimal mu is: {mu}")

    # Assign a value to the variable "sigma"
    # sigma = mu - ( 1+beta ) * lambda_mu + lambda_mu * np.sum(f * np.log ( (lambda_mu * f) / (mu - v) ) )
    # print(f"Optimal sigma is: {sigma}")