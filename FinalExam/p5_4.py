import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    x = np.clip(x, -700, 700)
    return 1 / (1 + np.exp(-x))

def generate_data(m, n):
    X = np.random.randn(m, n)
    w_true = np.random.randn(n)
    prob = 1 / (1 + np.exp(-X @ w_true))
    y = (prob > 0.5).astype(float)
    return X, y

def batch_gradient_descent(X, y, w_init, lr=0.1, max_iter=1000):
    start_time = time.time()
    w = w_init.copy()
    times = [0]
    losses = [compute_loss(X, y, w)]
    
    for i in range(max_iter):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y) / len(y)
        w = w - lr * grad
        
        if i % 10 == 0:
            times.append(time.time() - start_time)
            losses.append(compute_loss(X, y, w))
    
    return w, times, losses

def stochastic_gradient_descent(X, y, w_init, max_iter=5000):
    start_time = time.time()
    w = w_init.copy()
    times = [0]
    losses = [compute_loss(X, y, w)]
    batch_size = 32
    
    for t in range(max_iter):
        indices = np.random.choice(len(y), batch_size)
        x_batch = X[indices]
        y_batch = y[indices]
        
        lr = 1.0 / np.sqrt(t + 1)
        p = sigmoid(x_batch @ w)
        grad = x_batch.T @ (p - y_batch) / batch_size
        w = w - lr * grad
        
        if t % 10 == 0:
            times.append(time.time() - start_time)
            losses.append(compute_loss(X, y, w))
    
    return w, times, losses

def compute_loss(X, y, w):
    p = sigmoid(X @ w)
    return -np.mean(y * np.log(p + 1e-10) + (1-y) * np.log(1 - p + 1e-10))

n = 100
sample_sizes = [100, 1000, 10000, 100000]

plt.figure(figsize=(15, 10))
for i, m in enumerate(sample_sizes, 1):
    np.random.seed(42)
    X, y = generate_data(m, n)
    w_init = np.zeros(n)
    
    _, times_gd, losses_gd = batch_gradient_descent(X, y, w_init)
    _, times_sgd, losses_sgd = stochastic_gradient_descent(X, y, w_init)
    
    plt.subplot(2, 2, i)
    plt.plot(times_gd, losses_gd, label='GD')
    plt.plot(times_sgd, losses_sgd, '--', label='SGD')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Loss')
    plt.title(f'm = {m}')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()