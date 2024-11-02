# CPSC 8420-843, Fall 2024
# Homework 1, Problem 2 (Least Squares Extension)
# Matthew Collins - Charleston Campus
# References: 
#   Chan, Stanley H. Introduction to Probability for Data Science. Michigan Publishing, 2023, pp. 396-422
#   https://math.stackexchange.com/questions/4808422/least-squares-solution-to-underdetermined-lyapunov-equation
#   https://www.aimspress.com/article/doi/10.3934/mmc.2021009?viewType=HTML 

import numpy as np

def solve_least_squares(A, C, Y):
    n = A.shape[0]  
    I_n = np.eye(n)
    A_kron = np.kron(I_n, A)
    C_kron = np.kron(C.T, I_n)
    A_least_squares = A_kron + C_kron
    b = Y.flatten()
    
    U, s, Vt = np.linalg.svd(A_least_squares, full_matrices=False)
    s_inv = np.diag(1 / s)
    A_pseudo_inv = Vt.T @ s_inv @ U.T  # Pseudo-inverse using SVD

    X_vec = A_pseudo_inv @ b
    X = X_vec.reshape(n, n)
    
    return X


def calculate_residual(A, C, X, Y):
    residual = A @ X + X @ C - Y
    return np.linalg.norm(residual, 'fro')  # Frobenius norm of the residual

def calculate_norm_difference(X_solution, X_star):
    return np.linalg.norm(X_solution - X_star, 'fro')


n = 4 
np.random.seed(0)
A = np.random.rand(n, n)*0.1
C = np.random.rand(n, n)*0.1
X_star = np.random.rand(n,n)

Y = A @ X_star + X_star @ C
X_solution = solve_least_squares(A, C, Y)


residual = calculate_residual(A, C, X_solution, Y)
norm_difference = calculate_norm_difference(X_solution, X_star)

print("X solved using least squares:\n", X_solution)
print("X (ground truth):\n", X_star)
print("Residual: ", residual)
print("Norm difference: ", norm_difference)
