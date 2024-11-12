import numpy as np

# Define dimensions and rank
n, d, r = 2000, 2000, 200
T = 100  # Number of iterations
lambda_ridge = 0.1  # Regularization parameter for ridge regression

# Initialize random matrices A and B
np.random.seed(0)
A = np.random.randn(n, r)
B = np.random.randn(d, r)

# Generate a synthetic matrix X that we aim to approximate
X = A @ B.T  # Ground truth matrix

# Alternating minimization to find optimal A and B
for t in range(T):
    # Update A while fixing B using vectorized ridge regression update
    BBT = B.T @ B + lambda_ridge * np.eye(r)
    A = X @ B @ np.linalg.inv(BBT)
    
    # Update B while fixing A using vectorized ridge regression update
    AAT = A.T @ A + lambda_ridge * np.eye(r)
    B = X.T @ A @ np.linalg.inv(AAT)

# Compute the final approximation of Z*
Z_star = A @ B.T

# Display result
print("Approximate solution Z* found.")
