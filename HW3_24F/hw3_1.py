import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def recon(channel, nPC):
    # Perform SVD
    u, s, v = np.linalg.svd(channel, full_matrices=False)
    # Reconstruction using nPC columns
    picrecon = (u[:, :nPC] @ np.diag(s[:nPC]) @ v[:nPC, :])
    return picrecon

def reconRGB(nPC):
    R_recon = recon(R, nPC)
    G_recon = recon(G, nPC)
    B_recon = recon(B, nPC)
    recon_img = np.stack((R_recon, G_recon, B_recon), axis=-1) + mean
    return np.clip(recon_img, 0, 1)

def varPic(picArr):
    return np.var(picArr.reshape(-1, 3), axis=0)


image_path = "Lenna.png"
original = io.imread(image_path) / 255.0 
pxl = original.reshape(-1, 3)
mean = np.mean(pxl, axis=0)
pxlCtr = pxl - mean
R = pxlCtr[:, 0].reshape(512, 512)
G = pxlCtr[:, 1].reshape(512, 512)
B = pxlCtr[:, 2].reshape(512, 512)

VarOrig = varPic(original)
percentVar_R = []
percentVar_G = []
percentVar_B = []
nPC_values = [2, 5, 20, 50, 80, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, nPC in zip(axes.flat, nPC_values):
    recon_img = reconRGB(nPC)
    ax.imshow(recon_img)
    ax.set_title(f"{nPC} Principal Components")
    ax.axis('off')
    variance_retained = varPic(recon_img) / VarOrig
    percentVar_R.append(variance_retained[0])
    percentVar_G.append(variance_retained[1])
    percentVar_B.append(variance_retained[2])

plt.suptitle("PCA Reconstructions with Different Numbers of Principal Components")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(nPC_values, percentVar_R, color='red', marker='o', label='Red Channel')
plt.plot(nPC_values, percentVar_G, color='green', marker='o', label='Green Channel')
plt.plot(nPC_values, percentVar_B, color='blue', marker='o', label='Blue Channel')
plt.xlabel("Number of Principal Components")
plt.ylabel("Variance Retained (%)")
plt.title("Variance Retained Across Principal Components by Channel")
plt.legend()
plt.grid(True)
plt.show()