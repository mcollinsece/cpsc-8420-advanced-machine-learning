import numpy as np
from sklearn.datasets import load_iris
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
solvers.options['show_progress'] = False

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target

def compute_kernel(x1, x2, gamma=1.0):
    gaussian = np.exp(-gamma * np.sum((x1 - x2)**2))
    linear = np.dot(x1, x2)
    return gaussian + linear

def train_svm(X, y, C=1.0):
    n_samples = len(X)
    
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = compute_kernel(X[i], X[j])
    
    y = y.astype(float)
    
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones(n_samples))
    G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = matrix(np.hstack((np.zeros(n_samples), C*np.ones(n_samples))))
    A = matrix(y.reshape(1, -1).astype(float))
    b = matrix(np.zeros(1))
    
    solution = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(solution['x']).flatten()
    
    sv = alphas > 1e-5
    return np.sum(sv)

print("\nNumber of support vectors for each binary classifier:")
for i in range(3):
    for j in range(i+1, 3):
        mask = np.logical_or(y == i, y == j)
        X_sub = X[mask]
        y_sub = y[mask]
        y_sub = np.where(y_sub == i, 1, -1)
        
        n_sv = train_svm(X_sub, y_sub)
        print(f"Classes {i} vs {j}: {n_sv} support vectors")