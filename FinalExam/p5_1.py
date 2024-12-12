import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m = 100
n = 100

X = np.random.randn(m, n) 
w_true = np.random.randn(n)
prob = 1 / (1 + np.exp(-X @ w_true))
y_01 = (prob > 0.5).astype(float)
y_pm = 2 * y_01 - 1

def gradient_descent_pm(X, y, w_init, lr=0.01, max_iter=1000):
    w = w_init.copy()
    losses = []
    
    for i in range(max_iter):
        z = y * (X @ w)
        pred = 1 / (1 + np.exp(z))
        loss = np.mean(np.log(1 + np.exp(-z)))
        grad = -X.T @ (y * pred) / m
            
        w = w - lr * grad
        losses.append(loss)
    
    return w, losses

def gradient_descent_01(X, y, w_init, lr=0.01, max_iter=1000):
    w = w_init.copy()
    losses = []
    
    for i in range(max_iter):
        z = X @ w
        pred = 1 / (1 + np.exp(-z))
        loss = -np.mean(y * np.log(pred + 1e-10) + (1-y) * np.log(1 - pred + 1e-10))
        grad = X.T @ (pred - y) / m
            
        w = w - lr * grad
        losses.append(loss)
    
    return w, losses

w_init = np.zeros(n)

w_pm, losses_pm = gradient_descent_pm(X, y_pm, w_init)
w_01, losses_01 = gradient_descent_01(X, y_01, w_init)

plt.figure(figsize=(10, 5))
plt.plot(losses_pm, label='{-1,+1} formulation')
plt.plot(losses_01, '--', label='{0,1} formulation')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration for Both Formulations')
plt.legend()
plt.grid(True)

weight_diff = np.linalg.norm(w_pm - w_01)
print(f"L2 norm of weight difference: {weight_diff:.6f}")
print(f"Weights for ±1: {w_pm}")
print(f"Weights for 0/1: {w_01}")

plt.show()