import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

n_values = [50, 100, 200]
p_values = [500, 1000, 2000]

results = []

for n in n_values:
    for p in p_values:
        np.random.seed(0)
        X = np.random.randn(n, p)

        # Direct Eigenvalue Decomposition of X^T X
        start_time = time.time()
        XtX = np.dot(X.T, X)
        _, V1 = np.linalg.eigh(XtX)
        time_direct = time.time() - start_time

        # Indirect (Efficient Computation) Eigenvalue Decomposition using X X^T
        start_time = time.time()
        XXt = np.dot(X, X.T)
        D, U = np.linalg.eigh(XXt)
        V2 = np.dot(X.T, U) / np.sqrt(D)  # Normalize eigenvectors
        time_indirect = time.time() - start_time

        epsilon = 1e-10
        speedup = time_direct / (time_indirect + epsilon)

        results.append({
            'n': n,
            'p': p,
            'time_direct': time_direct,
            'time_indirect': time_indirect,
            'speedup': speedup
        })

results_df = pd.DataFrame(results)
print(results_df)

for n in n_values:
    subset = results_df[results_df['n'] == n]
    plt.plot(subset['p'], subset['speedup'], marker='o', label=f'n={n}')

plt.xlabel('p (Feature Size)')
plt.ylabel('Speedup (Time Direct / Time Indirect)')
plt.title('Speedup of Indirect Method vs. Direct Method')
plt.legend()
plt.grid(True)
plt.show()
