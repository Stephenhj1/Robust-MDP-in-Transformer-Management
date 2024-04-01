import numpy as np
import pandas as pd

n = 369  # Total number of states
n_main = n_intermediate = 184  # Number of main and intermediate states
n_failure = 1  # Number of failure states

# Conditions within states
cond_1 = 70
cond_2 = 64
cond_3 = 50

# Initialize the transition matrix with zeros
F = np.zeros((n, n))

# Helper function to fill in the transition probabilities for a block of states
def fill_block(start_row, end_row, start_col, end_col, conditions):
    for i in range(start_row, end_row):
        if i - start_row < conditions[0]:  # Condition 1
            valid_cols = np.r_[start_col + conditions[0]:start_col + sum(conditions[:2]), n-1:n]  # To Condition 2, 3, and Failure
        elif i - start_row < sum(conditions[:2]):  # Condition 2
            valid_cols = np.r_[start_col + conditions[0] + conditions[1]:start_col + sum(conditions), n-1:n]  # To Condition 3 and Failure
        else:  # Condition 3
            valid_cols = np.r_[n-1:n]  # To Failure only
        
        F[i, valid_cols] = np.random.rand(len(valid_cols))
        F[i, valid_cols] /= F[i, valid_cols].sum()  # Normalize to sum to 1

# Fill in for main to intermediate and failure
fill_block(0, n_main, n_main, n-1, [cond_1, cond_2, cond_3])

# Fill in for intermediate to main and failure
fill_block(n_main, n-1, 0, n_main, [cond_1, cond_2, cond_3])

# Failure state transitions to itself
F[n-1, n-1] = 1.0

# Check if the row sums are all close to 1
print("Row sums close to 1:", np.allclose(F.sum(axis=1), np.ones(n)))
# print(F)
print(F.sum(axis=1))