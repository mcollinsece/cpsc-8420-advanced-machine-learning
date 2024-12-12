import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
w_true = np.array([2.0, -1.0])
p_true = 1 / (1 + np.exp(-X @ w_true))
y = (p_true > 0.5).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent_mse(X, y, w_init, lr=0.1, max_iter=1000):
    w = w_init.copy()
    losses = []
    probs = []
    
    for i in range(max_iter):
        p = sigmoid(X @ w)
        loss = 0.5 * np.mean((y - p)**2)
        grad = -X.T @ ((y - p) * p * (1-p)) / len(y)
        
        w = w - lr * grad
        losses.append(loss)
        probs.append(p[0])
    
    return w, losses, probs

def gradient_descent_ce(X, y, w_init, lr=0.1, max_iter=1000):
    w = w_init.copy()
    losses = []
    probs = []
    
    for i in range(max_iter):
        p = sigmoid(X @ w)
        loss = -np.mean(y * np.log(p + 1e-10) + (1-y) * np.log(1 - p + 1e-10))
        grad = X.T @ (p - y) / len(y)
        
        w = w - lr * grad
        losses.append(loss)
        probs.append(p[0])
    
    return w, losses, probs

w_init = np.array([-10.0, -10.0])

w_mse, losses_mse, probs_mse = gradient_descent_mse(X, y, w_init)
w_ce, losses_ce, probs_ce = gradient_descent_ce(X, y, w_init)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(losses_mse, label='MSE')
plt.plot(losses_ce, label='Cross-Entropy')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs Iteration')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.plot(probs_mse, label='MSE')
plt.plot(probs_ce, label='Cross-Entropy')
plt.xlabel('Iteration')
plt.ylabel('Probability (p)')
plt.title('Probability Evolution for First Sample')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final probability (MSE): {probs_mse[-1]:.6f}")
print(f"Final probability (CE): {probs_ce[-1]:.6f}")