import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

n, d, r = 2000, 2000, 200  
T = 100 
lambda_ridge = 0.001  
tolerance = 1e-6  
np.random.seed(0)  

A = np.random.randn(n, r) * 0.01 
B = np.random.randn(d, r) * 0.01
# Ground truth matrix X
X = A @ B.T  

def update_A(A, B, X, lambda_ridge):
    for i in range(n):
        ridge_reg = Ridge(alpha=lambda_ridge, fit_intercept=False)
        ridge_reg.fit(B, X[i, :])  
        A[i, :] = ridge_reg.coef_  
    return A

def update_B(A, B, X, lambda_ridge):
    for j in range(d):
        ridge_reg = Ridge(alpha=lambda_ridge, fit_intercept=False)
        ridge_reg.fit(A, X[:, j])  
        B[j, :] = ridge_reg.coef_  
    return B

frob_errors = []
prev_error = None  

for t in range(T):
    A = update_A(A, B, X, lambda_ridge)
    B = update_B(A, B, X, lambda_ridge)
    Z_current = A @ B.T
    frobenius_error = np.linalg.norm(X - Z_current, 'fro') / np.linalg.norm(X, 'fro')
    frob_errors.append(frobenius_error)
    print(f"Iteration {t + 1}, Frobenius error: {frobenius_error:.6f}")
    
Z_star = A @ B.T
print("Approximate solution Z* has been found.")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(frob_errors) + 1), frob_errors, marker='o', color='b')
plt.xlabel('Iteration')
plt.ylabel('Relative Frobenius Error')
plt.title('Convergence of Frobenius Error Between X and Z*')
plt.grid(True)
plt.show()
