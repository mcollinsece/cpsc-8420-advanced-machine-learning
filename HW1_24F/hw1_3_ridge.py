# CPSC 8420-843, Fall 2024
# Homework 1, Problem 3 (Monotonic Ridge Regression)
# Matthew Collins - Charleston Campus
# References: 
#   Chan, Stanley H. Introduction to Probability for Data Science. Michigan Publishing, 2023, pp. 440-448i
#   https://medium.com/@msoczi/ridge-regression-step-by-step-introduction-with-example-0d22dddb7d54

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
m, n = 100, 50
A = np.random.randn(m, n)
y = np.random.randn(m, 1)

lambdas = np.logspace(-3, 3, 100)
norms = []

for lam in lambdas:
    beta_ridge = np.linalg.inv(A.T @ A + lam * np.eye(n)) @ A.T @ y
    norm = np.linalg.norm(beta_ridge, 2)
    norms.append(norm)

plt.plot(lambdas, norms, marker='o', label='Norm of ridge solution')
plt.xscale('log')
plt.xlabel('lambda')
plt.ylabel('norms')
plt.title('Norm of Ridge Regression Solution vs Lambda')
plt.legend()
#plt.show()
plt.savefig(".\\sln_figures\\fig3.png")
