import numpy as np
import time
	
# Generate a random matrix X with n << p
n, p = 100, 1000
np.random.seed(0)
X = np.random.randn(n, p)
	
# Method 1: Compute eigenvectors of X^T X directly
start_time = time.time()
XtX = np.dot(X.T, X)
_, V1 = np.linalg.eigh(XtX)
time_direct = time.time() - start_time
	
# Method 2: Compute eigenvectors of X X^T and transform
start_time = time.time()
XXt = np.dot(X, X.T)
D, U = np.linalg.eigh(XXt)
V2 = np.dot(X.T, U) * (1 / np.sqrt(D))  # Normalize eigenvectors
time_indirect = time.time() - start_time
	
# Display the computational times
print("Time using direct method (X^T X):", time_direct, "seconds")
print("Time using indirect method (X X^T):", time_indirect, "seconds")