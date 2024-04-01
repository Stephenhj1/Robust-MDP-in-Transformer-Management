import numpy as np

# Example vector v and matrix f
v = np.array([1, 2, 3])  # Example vector
f = np.array([[0, 0, 0],  # Zero vector
              [1, 2, 3],
              [0, 0, 0],  # Zero vector
              [4, 5, 6]])

# Iterate over the rows of f
for i, row in enumerate(f):
    # Check if the row vector is a zero vector
    if not np.all(row == 0):
        # If it's not a zero vector, compute the dot product with v
        a = np.dot(v, row)
        print(f"Dot product of v and f[{i}] is: {a}")