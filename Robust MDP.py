import numpy as np

def bisection_algorithm(f_prime, lower_bound, upper_bound, delta):
    """
    Bisection algorithm to find the minimum of the support function.
    Args:
        f_prime (function): Derivative of the function for which the minimum is sought.
        lower_bound (float): Lower bound of the search interval.
        upper_bound (float): Upper bound of the search interval.
        delta (float): Convergence criterion.
    
    Returns:
        float: Value that minimizes the function.
    """
    while upper_bound - lower_bound > delta * (1 + abs(lower_bound) + abs(upper_bound)):
        mid = (upper_bound + lower_bound) / 2
        if f_prime(mid) > 0:
            upper_bound = mid
        else:
            lower_bound = mid
    return (upper_bound + lower_bound) / 2

def robust_dynamic_programming(states, actions, terminal_cost, transition_cost, T, epsilon):
    """
    Computes an ε-suboptimal policy using the robust finite-horizon dynamic programming algorithm.
    Args:
        states (list): List of states.
        actions (list): List of actions.
        terminal_cost (function): Function that returns the cost at the terminal stage for a state.
        transition_cost (function): Function that returns the transition cost from one state to another under an action.
        T (int): Time horizon.
        epsilon (float): Convergence parameter.
    
    Returns:
        dict: A dictionary mapping each state at each time to an action.
    """
    value_function = {state: terminal_cost(state) for state in states}
    policy = {}

    for t in reversed(range(T)):
        new_value_function = {}
        for state in states:
            value_action_pairs = []
            for action in actions:
                # Simplified; real implementation would use the bisection algorithm here
                # to compute a specific value that minimizes the transition cost
                cost = transition_cost(state, action, value_function)
                value_action_pairs.append((cost, action))
            min_cost, best_action = min(value_action_pairs)
            new_value_function[state] = min_cost
            policy[(state, t)] = best_action
        value_function = new_value_function

    return policy

# Main function