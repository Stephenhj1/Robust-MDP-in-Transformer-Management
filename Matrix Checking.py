import numpy as np
import pandas as pd

# def is_valid_matrix(matrix):
#     """
#     Check if the given matrix is a stochastic matrix. A stochastic matrix has every row summing to 1,
#     and every element is greater than or equal to 0.
    
#     Parameters:
#     - matrix (numpy.ndarray): The matrix to check.
    
#     Returns:
#     - bool: True if the matrix is stochastic, False otherwise.
#     """
#     # Check if all elements are greater than or equal to 0
#     non_negative = np.all(matrix >= 0)
    
#     # Check if the sum of each row is 1
#     row_sums = np.sum(matrix, axis=1)
#     rows_sum_to_one = np.all(np.isclose(row_sums, 1))
    
#     return non_negative and rows_sum_to_one

def check_matrix_rows(matrix):
    # Check if each row is either a zero vector or sums to 1
    # and if every element is greater than or equal to 0
    for row in matrix:
        # Check if the row sums to 0 or 1
        if not np.isclose(row.sum(), 1) and not np.isclose(row.sum(), 0):
            return False
        # Check if any element in the row is negative
        if np.any(row < 0):
            return False
    return True

# Read excel file
matrix = pd.read_excel('F_a4.xlsx', header=None).values

result = check_matrix_rows(matrix)
print(f"The given matrix is a valid matrix: {result}")
