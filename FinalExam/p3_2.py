import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("USArrests.csv")
states = data.iloc[:,0]
features = ["Murder", "Assault", "UrbanPop", "Rape"]

scaler = StandardScaler()
X = data[features]
X_scaled = scaler.fit_transform(X)

mask = np.ones_like(X_scaled)
for i in range(len(states)):
    mask[i, np.random.randint(0, 4)] = 0

def nuclear_prox(Z, tau):
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    s = np.maximum(s - tau, 0)
    return U @ np.diag(s) @ Vt

def objective(X, Z, mask, lambda_=1.0):
    return 0.5 * np.sum((mask * (X - Z))**2) + np.sum(np.linalg.svd(Z, compute_uv=False))

Z = np.zeros_like(X_scaled)
step_size = 1.0
max_iter = 100
obj_values = []

for iter in range(max_iter):
    grad = mask * (Z - X_scaled)
    Z_new = Z - step_size * grad
    Z_new = nuclear_prox(Z_new, step_size)
    
    obj = objective(X_scaled, Z_new, mask)
    obj_values.append(obj)
    
    if iter > 0 and abs(obj_values[-1] - obj_values[-2]) < 1e-6:
        break
    
    Z = Z_new

plt.figure(figsize=(10, 5))
plt.plot(obj_values, 'b-')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Objective Function vs Iteration')
plt.grid(True)
plt.show()