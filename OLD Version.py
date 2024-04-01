import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def element_product(f_j):
    """Compute f[j] * log(f[j]) for a given f[j], handling f[j] = 0."""
    if f_j > 0:
        return f_j * np.log(f_j)
    else:
        return 0.0

def compute_beta_max(f):
    """Compute beta_max for a given vector f."""
    # assert np.sum(f) == 1 and np.all(f >= 0), "f must be a probability vector."
    beta_max = sum(element_product(f_j) for f_j in f)
    return beta_max

def compute_bounds(f, v, beta_max, beta):
    """Compute the lower bound mu_minus and upper bound mu_plus."""
    v_bar = np.dot(f, v)  # Compute the average of v under f
    mu_minus = np.max(v)
    
    mu_plus = ( np.max(v)-np.exp(beta-beta_max)* v_bar )/( 1-np.exp(beta-beta_max) )
    return mu_minus, mu_plus

# def element_partial_derivative(f_j):
#     """Compute f[j] * np.log ( (lambda_mu * f[j]) / (mu - v) ) for a given f[j], handling f[j] = 0."""
#     if f_j > 0:
#         return f_j * np.log ( (lambda_mu * f_j) / (mu - v) )
#     else:
#         return 0.0

def derivative_support_function(f, v, mu, beta):
    """Compute the partial derivative of function h which is the decision key of the Bisection Algorithm."""
    valid_indices = (mu - V != 0) & (f != 0)
    result = np.sum(f[valid_indices] / (mu - V[valid_indices]))
    if result == 0:
        lambda_mu = 0
    else:
        lambda_mu = 1 / result
    # lambda_mu = 1 / np.sum( f / (mu-v) )
    # If f_j>0, add the partial derivative of the element to the sum, if not, add 0.0
    # Mask = f > 0
    if lambda_mu == 0:
        partial_derivative_h = -beta
    else:
        partial_derivative_h = np.sum( f[valid_indices] * np.log ( (lambda_mu * f[valid_indices]) / (mu - v[valid_indices]) ) ) - beta
    # partial_derivative_h = np.sum( f * np.log ( (lambda_mu * f) / (mu - v) ) ) - beta
    return partial_derivative_h

def bisection_algorithm(f, v, beta_max, beta, delta):
    """  Use Bisection Algorithm to find the optimal mu satisfying certain conditions.  """
    mu_minus, mu_plus = compute_bounds(f, v, beta_max, beta)
    mu = (mu_minus + mu_plus) / 2
    
    while mu_plus - mu_minus > delta * (1 + mu_minus + mu_minus):
        mu = (mu_plus + mu_minus) / 2
        if derivative_support_function(f, v, mu, beta) > 0:
            mu_plus = mu
        else:
            mu_minus = mu
    return mu

def main_or_intermediate(state, F_a1):
    """Check if a state is a main or intermediate state."""
    # Check if the i-th row of F_a1 is a zero vector
    if np.all(F_a1[state,:] == 0):
        return "Main State"
    else:
        return "Intermediate State"

def Robust_Dynamic_Programming(X, A, F, C, c_N, N):
    # Initialize value function for the final time step with terminal costs
    # Assign c_N to V
    V = c_N.copy()
    # V = {i: c_N[i] for i in X}  # c_N is now directly a dictionary mapping states to terminal costs
    policy = {t: {i: None for i in X} for t in range(N)}  # Policy initialization

    # Loop over each time step in reverse
    for t in range(N-1, -1, -1):
        V_prev = V.copy()
        for i in X:
            # Judge if the state is a main or intermediate state
            if (t % 2 == 0 and main_or_intermediate(i, F_a1) == "Main State") or (t % 2 != 0 and main_or_intermediate(i, F_a1) == "Intermediate State"):
                
                # Calculate the cost for all actions and choose the min
                action_costs = {}
                for a in A:
                    f = F[a][i, :]  # Extract frequency vector for action a and state s
                    # Skip if f is a zero vector
                    if np.all(f == 0):
                        continue

                    epsilon = 1e-5
                    beta_max = compute_beta_max(f)
                    beta = beta_max - 0.3
                    mu = bisection_algorithm(f, V_prev, beta_max, beta, epsilon)

                    valid_indices = (mu - V != 0) & (f != 0)
                    result = np.sum(f[valid_indices] / (mu - V[valid_indices]))
                    if result == 0:
                        lambda_mu = 0
                    else:
                        lambda_mu = 1 / result
                    
                    if lambda_mu == 0:
                        sigma = mu
                    else:
                        sigma = mu - ( 1+beta ) * lambda_mu + lambda_mu * np.sum(f * np.log ( (lambda_mu * f) / (mu - V) ) )
                    
                    action_cost = C[a][i] + sigma
                    action_costs[a] = action_cost
            
                # Set the optimal value and policy
                if action_costs:
                    optimal_action = min(action_costs, key=action_costs.get)
                    V[i] = action_costs[optimal_action]
                    policy[t][i] = optimal_action

    return policy, V

# Main function
if __name__ == "__main__":
    
    # Number of States
    num_states = 11

    # States and Actions
    X = np.arange(num_states)  # State indices from 0 to 368
    A = ['a0', 'a1', 'a2', 'a3', 'a4']  # Actions a0, a1, a2, a3, a4

    # Matrix of empirical frequencies transitions F^a
    F_a0 = pd.read_excel('F_a0.xlsx', header=None).values
    F_a1 = pd.read_excel('F_a1.xlsx', header=None).values
    F_a2 = pd.read_excel('F_a2.xlsx', header=None).values
    F_a3 = pd.read_excel('F_a3.xlsx', header=None).values
    F_a4 = pd.read_excel('F_a4.xlsx', header=None).values
    F = {'a0': F_a0, 'a1': F_a1, 'a2': F_a2, 'a3': F_a3, 'a4': F_a4}

    # Cost function C
    C_a0 = np.full(num_states, 0)  # Cost for action 'a0'
    C_a1 = np.full(num_states, 200)  # Cost for action 'a1'
    C_a2 = np.full(num_states, 1200)  # Cost for action 'a2'
    C_a3 = np.full(num_states, 14400)
    C_a4 = np.full(num_states, 144000)
    C = {'a0': C_a0, 'a1': C_a1, 'a2': C_a2, 'a3': C_a3, 'a4': C_a4}

    # Terminal Costs
    c_N = np.zeros(num_states)

    # Horizon
    N = 31

    

    # Run the Robust Dynamic Programming algorithm
    policy, V = Robust_Dynamic_Programming(X, A, F, C, c_N, N)
    
    print(policy)
    print(V)

    # delta = 1e-5
    
    # a0 = 'a0'
    # f = F[a0][0, :]
    # V = c_N.copy()
    # beta_max = compute_beta_max(f)
    # print(f"beta_max is: {beta_max}")
    # beta = beta_max - 0.3
    # mu_minus, mu_plus = compute_bounds(f, V, beta_max, beta)
    # print(f"mu_minus is: {mu_minus},", f"mu_plus is: {mu_plus}")

    # # derivative_sigma = derivative_support_function(f, V, mu, beta)
    # # print(f"Derivative of the support function is: {derivative_sigma}")

    # mu = bisection_algorithm(f, V, beta_max, beta, delta)
    # print(f"Optimal mu is: {mu}")

    # # Assign a value to the variable "sigma"
    # valid_indices = (mu - V != 0) & (f != 0)
    # result = np.sum(f[valid_indices] / (mu - V[valid_indices]))
    # if result == 0:
    #     lambda_mu = 0
    # else:
    #     lambda_mu = 1/result
    # print(f"lambda_mu is: {lambda_mu}")

    # if lambda_mu == 0:
    #     sigma = mu
    # else:
    #     sigma = mu - ( 1+beta ) * lambda_mu + lambda_mu * np.sum(f * np.log ( (lambda_mu * f) / (mu - V) ) )
    # print(f"Optimal sigma is: {sigma}")