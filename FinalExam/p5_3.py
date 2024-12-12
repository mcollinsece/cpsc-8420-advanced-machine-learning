import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m = 100
n = 100

X = np.random.randn(m, n)
w_true = np.random.randn(n)
prob = 1 / (1 + np.exp(-X @ w_true))
y = (prob > 0.5).astype(float)

def sigmoid(x):
    # Clip values for numerical stability
    x = np.clip(x, -700, 700)
    return 1 / (1 + np.exp(-x))

def gradient_descent(X, y, w_init, lr=0.1, max_iter=1000):
    w = w_init.copy()
    losses = []
    
    for i in range(max_iter):
        z = X @ w
        p = sigmoid(z)
        loss = -np.mean(y * np.log(p + 1e-10) + (1-y) * np.log(1 - p + 1e-10))
        grad = X.T @ (p - y) / m
        
        w = w - lr * grad
        losses.append(loss)
    
    return w, losses

def newton_method(X, y, w_init, max_iter=1000):
    w = w_init.copy()
    losses = []
    
    # Small regularization to prevent singular Hessian
    lambda_reg = 1e-5
    I = np.eye(n)
    
    for i in range(max_iter):
        z = X @ w
        p = sigmoid(z)
        loss = -np.mean(y * np.log(p + 1e-10) + (1-y) * np.log(1 - p + 1e-10))
        grad = X.T @ (p - y) / m
        
        # Compute Hessian with regularization
        S = np.diag((p * (1-p)).flatten())
        H = X.T @ S @ X / m + lambda_reg * I
        
        # Newton update with unit stepsize
        w = w - np.linalg.solve(H, grad)
        losses.append(loss)
    
    return w, losses

w_init = np.zeros(n)

w_gd, losses_gd = gradient_descent(X, y, w_init)
w_newton, losses_newton = newton_method(X, y, w_init)

plt.figure(figsize=(10, 5))
plt.plot(losses_gd, label='Gradient Descent')
plt.plot(losses_newton, label='Newton Method')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.grid(True)
plt.show() 