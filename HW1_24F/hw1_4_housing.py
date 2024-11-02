import numpy as np
from sklearn.model_selection import train_test_split

data = np.loadtxt('housing.data')

x = data[:, :13]
y = data[:, 13]
n, d = x.shape
np.random.seed(0)

perm = np.random.permutation(n)
x = x[perm]
y = y[perm]
training_samples = 300

Xtrain = x[:training_samples]
ytrain = y[:training_samples]
Xtest = x[training_samples:]
ytest = y[training_samples:]


