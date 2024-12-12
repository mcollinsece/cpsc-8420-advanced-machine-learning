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

X_centered = X_scaled - np.mean(X_scaled, axis=0)
n_samples = X_scaled.shape[0]

cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

pc1 = eigenvecs[:, 0]
pc2 = eigenvecs[:, 1]

proj_pc1 = np.dot(X_centered, pc1)
proj_pc2 = np.dot(X_centered, pc2)

plt.figure(figsize=(10, 8))
plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5)

for i in range(len(states)):
    plt.text(proj_pc1[i], proj_pc2[i], states[i])

for i in range(len(features)):
    plt.arrow(0, 0, 
              pc1[i] * 3, pc2[i] * 3, 
              head_width=0.05, 
              head_length=0.05, 
              color='red')
    plt.annotate(features[i],
                xy=(pc1[i] * 3.5, pc2[i] * 3.5),
                xytext=(20, -20),
                textcoords='offset pixels', 
                color='red')

plt.grid(True)
plt.title('PCA Biplot of US Arrests Data')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()