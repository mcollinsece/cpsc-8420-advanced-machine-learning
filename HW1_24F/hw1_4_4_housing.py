# CPSC 8420-843, Fall 2024
# Homework 1, Problem 4.3
# Matthew Collins - Charleston Campus
# References: 
#   Chan, Stanley H. Introduction to Probability for Data Science. Michigan Publishing, 2023, Chapter 7
#   https://medium.com/@dahami/understanding-ordinary-least-squares-ols-and-its-applications-in-statistics-machine-learning-ad2c13681501
#   https://mashkarharis.medium.com/linear-regression-in-python-scikit-learn-526b57a11a09
#   https://contactsunny.medium.com/linear-regression-in-python-using-scikit-learn-f0f7b125a204
#   https://medium.com/@bernardolago/mastering-ridge-regression-a-key-to-taming-data-complexity-98b67d343087
#   https://alok05.medium.com/ridge-and-lasso-regression-practical-implementation-in-python-c4a813a99bce

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as sl
from sklearn.metrics import mean_squared_error

data = np.loadtxt('housing.data')
seed = 0
np.random.seed(seed)
randperm = np.random.randint(0, len(data), len(data))
data = data[randperm, :]

def ridge_least_squares(A, y, lam):
    iden_part = lam * np.identity(len(A.T))
    iden_part[0, 0] = 1  # Do not regularize the bias term
    result = np.linalg.pinv((A.T).dot(A) + iden_part).dot(A.T)
    beta = result.dot(y)
    return beta

def mse(y, beta, X):
    y_hat = X.dot(beta)
    return ((y - y_hat).T).dot(y - y_hat)[0][0] / len(y)

Ntrain = 300

mse_test = []
mse_train = []
mse_SL_test = []
mse_SL_train = []

lambdas = np.logspace(-10, 10, 10)

degrees = [2, 3, 4, 5, 6]
features = data[:, :-1]

for deg in degrees:
    features = np.concatenate((features, data[:, :-1] ** deg), axis=1)

Xtrain = stats.zscore(features[:Ntrain, :], axis=0)
ytrain = data[:Ntrain, -1][..., None]

Xtest = stats.zscore(features[Ntrain:, :], axis=0)
ytest = data[Ntrain:, -1][..., None]

Xtrain_mt = np.concatenate((np.ones((len(Xtrain), 1)), Xtrain), axis=1)
Xtest_mt = np.concatenate((np.ones((len(Xtest), 1)), Xtest), axis=1)

for l in lambdas:
    beta = ridge_least_squares(Xtrain_mt, ytrain, l)
    skl_model = sl.Ridge(l).fit(Xtrain, ytrain)
    
    mse_train.append(mse(ytrain, beta, Xtrain_mt))
    mse_test.append(mse(ytest, beta, Xtest_mt))
    
    mse_SL_train.append(mean_squared_error(ytrain, skl_model.predict(Xtrain)))
    mse_SL_test.append(mean_squared_error(ytest, skl_model.predict(Xtest)))

plt.plot(lambdas, mse_train, color='green', label='train', marker='D')
plt.plot(lambdas, mse_test, color='cyan', label='test', marker='D')
plt.plot(lambdas, mse_SL_train, color='b', label='sklearn train')
plt.plot(lambdas, mse_SL_test, color='r', label='sklearn test')
plt.xscale('log')
plt.legend()
plt.ylabel('MSE')
plt.xlabel('lambda')

#plt.show()
plt.savefig(".\\sln_figures\\fig4_4.png")
