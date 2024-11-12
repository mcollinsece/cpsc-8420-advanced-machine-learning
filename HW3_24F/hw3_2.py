import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl

def cost_function(X, y, theta, reg_param):
    return ((X.dot(theta) - y).T.dot(X.dot(theta) - y) + reg_param * theta.T.dot(theta))[0][0] / 2

def gradient(X, y, theta, reg_param):
    return X.T.dot(X.dot(theta) - y) + reg_param * theta

def gradient_descent(cost_fn, grad_fn, X, y, init_theta, reg_param, step_size, tolerance):
    theta = init_theta
    iteration = 0
    delta = np.inf
    cost_history = [cost_fn(X, y, init_theta, reg_param)]
    while delta > tolerance:
        theta = theta - step_size * grad_fn(X, y, theta, reg_param)
        iteration += 1
        cost_history.append(cost_fn(X, y, theta, reg_param))
        delta = abs((cost_history[-1] - cost_history[-2]) / cost_history[-1])
    return iteration, theta, cost_history[1:]

X = np.array([[1, 2, 4], [1, 3, 5], [1, 7, 7], [1, 8, 9]])
y = np.array([[1], [2], [3], [4]])

tolerance = 1e-4
theta_init = np.zeros((3, 1))  
regularization_params = [0, 0.1, 1, 10, 100, 200]  
iterations_list = []  
learning_rates = []  
cost_values = []  
convergence_ratios = []  

for reg_param in regularization_params:
    step_size = 1 / (np.max(sl.svd(X.T.dot(X))[1]) + reg_param)
    iteration, _, cost_hist = gradient_descent(cost_function, gradient, X, y, theta_init, reg_param, step_size, tolerance)
    iterations_list.append(iteration)
    learning_rates.append(step_size)
    cost_values.append(cost_hist)
    _, singular_values, _ = sl.svd(X.T.dot(X) + reg_param * np.identity(3))
    convergence_ratios.append(1 - np.min(singular_values) / np.max(singular_values))

cost_values = [cost_hist / cost_hist[0] for cost_hist in cost_values]

plt.figure(figsize=(14, 8))
for reg_param, iteration, cost_hist, lr, cr in zip(regularization_params, iterations_list, cost_values, learning_rates, convergence_ratios):
    plt.plot(np.arange(1, iteration + 1), cost_hist, 
             label=fr'Step Size = $\frac{{1}}{{{reg_param} + \sigma_{{\max}}(X^T X)}}$ = {lr:.5e}; CR = {cr:.4f}')
plt.xscale('log')
plt.legend(frameon=False, fontsize=12)
plt.xlabel('Number of Iterations', fontsize=14)
plt.ylabel('Normalized Cost Function Values', fontsize=14)
plt.title("Convergence of Gradient Descent with Various Regularization Parameters")
plt.grid(True)
plt.show()
