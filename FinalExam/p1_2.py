import numpy as np
import matplotlib.pyplot as plt

def soft_threshold(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def coordinate_descent_lasso(X, y, lambda_, max_iter=1000, tol=1e-6):
    n, p = X.shape
    beta = np.random.randn(p)
    X_squared = np.sum(X ** 2, axis=0)
    objectives = []
    
    obj_init = 0.5 * np.sum((y - np.dot(X, beta)) ** 2) + lambda_ * np.sum(np.abs(beta))
    objectives.append(obj_init)
    
    for iter in range(max_iter):
        beta_old = beta.copy()
        
        for j in range(p):
            r = y - np.dot(X, beta)
            r = r + X[:, j] * beta[j]
            rho = np.dot(X[:, j], r)
            beta[j] = soft_threshold(rho, lambda_) / (X_squared[j] + 1e-10)
        
        obj = 0.5 * np.sum((y - np.dot(X, beta)) ** 2) + lambda_ * np.sum(np.abs(beta))
        objectives.append(obj)
        
        if np.max(np.abs(beta - beta_old)) < tol:
            print(f"Converged after {iter+1} iterations")
            break

    return beta, objectives

np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
true_beta = np.random.randn(p)
y = np.dot(X, true_beta) + np.random.randn(n) * 0.1

Xty = np.dot(X.T, y)
Xty_inf_norm = np.max(np.abs(Xty))

lambda_ = 1.1 * Xty_inf_norm

print(f"||X^T * y||_∞ = {Xty_inf_norm:.6f}")
print(f"lambda = {lambda_:.6f}")

beta_opt, objectives = coordinate_descent_lasso(X, y, lambda_)

print("\nOptimal beta:")
print(beta_opt)
print(f"\nMax absolute value in beta: {np.max(np.abs(beta_opt)):.6e}")

plt.figure(figsize=(10, 5))
plt.plot(range(len(objectives)), objectives, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Convergence of Coordinate Descent')
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.show()