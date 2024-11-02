# CPSC 8420-843, Fall 2024
# Homework 1, Problem 1 (Lasso)
# Matthew Collins - Charleston Campus
# References: 
#   Chan, Stanley H. Introduction to Probability for Data Science. Michigan Publishing, 2023, pp. 449-457
#   https://xavierbourretsicotte.github.io/lasso_implementation.html
#   https://towardsdatascience.com/regularized-linear-regression-models-dcf5aa662ab9
#   https://towardsdatascience.com/from-linear-regression-to-ridge-regression-the-lasso-and-the-elastic-net-4eaecaf5f7e6

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def lasso_coordinate_descent(A, y, lambd, num_iters=1000, tol=1e-4):
    m, n = A.shape
    x = np.zeros(n)
    
    for _ in range(num_iters):
        x_old = x.copy()
        
        for j in range(n):
            residual = y - np.dot(A, x) + A[:, j] * x[j]
            rho = np.dot(A[:, j].T, residual)
            
            if rho < -lambd / 2:
                x[j] = (rho + lambd / 2) / np.dot(A[:, j], A[:, j])
            elif rho > lambd / 2:
                x[j] = (rho - lambd / 2) / (A[:, j] @ A[:, j])
            else:
                x[j] = 0
        
        if np.linalg.norm(x - x_old, ord=2) < tol:
            break

    return x

np.random.seed(0)
A = np.random.randn(20, 10)
y = np.random.randn(20)
lambdas = np.logspace(np.log10(0.01), np.log10(1000), 50)
x_values = []
x_sklearn_values = []

print(A.shape)

for lambd in lambdas:
    x_star = lasso_coordinate_descent(A, y, 4*A.shape[1]*lambd)
    x_values.append(x_star)

    lasso = Lasso(alpha=lambd, max_iter=10000, tol=1e-4)
    lasso.fit(A,y)
    x_sklearn_values.append(lasso.coef_)

x_values = np.array(x_values)
x_sklearn_values = np.array(x_sklearn_values)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for i in range(A.shape[1]):
    plt.plot(lambdas, x_values[:, i], label=f'x{i+1}')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Values of x')
plt.title('Coordinate Descent: Changes in x as Lambda Varies')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(A.shape[1]):
    plt.plot(lambdas, x_sklearn_values[:, i], label=f'x{i+1}', linestyle='--')
plt.xscale('log')
plt.xlabel('Lambda')
plt.ylabel('Values of x')
plt.title('Sklearn Lasso: Changes in x as Lambda Varies')
plt.legend()

#plt.show()
plt.savefig(".\\sln_figures\\fig1.png")