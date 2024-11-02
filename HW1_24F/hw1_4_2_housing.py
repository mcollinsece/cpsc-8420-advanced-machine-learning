# CPSC 8420-843, Fall 2024
# Homework 1, Problem 4.2
# Matthew Collins - Charleston Campus
# References: 
#   Chan, Stanley H. Introduction to Probability for Data Science. Michigan Publishing, 2023, Chapter 7
#   https://medium.com/@dahami/understanding-ordinary-least-squares-ols-and-its-applications-in-statistics-machine-learning-ad2c13681501
#   https://mashkarharis.medium.com/linear-regression-in-python-scikit-learn-526b57a11a09
#   https://contactsunny.medium.com/linear-regression-in-python-using-scikit-learn-f0f7b125a204

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import sklearn.linear_model as sl
from sklearn.metrics import mean_squared_error


data = np.loadtxt('housing.data')
np.random.seed(0)
randperm = np.random.randint(0,len(data),len(data))
data = data[randperm,:]

def ordinary_least_squares(A,y):
    result = np.linalg.pinv((A.T).dot(A)).dot(A.T)
    beta = result.dot(y)
    return beta

def mse(y,beta,x):
    y_hat = x.dot(beta)
    return ((y-y_hat).T).dot(y-y_hat)[0][0]/len(y)

n = [25,50,75,100,150,200,300]
mse_test = []
mse_train = []
mse_SL_train = []
mse_SL_test = []

for Ntrain in n:
    Xtrain = stats.zscore(data[:Ntrain, :-1], axis=0)
    ytrain = data[:Ntrain, -1][..., None]
    
    Xtest = stats.zscore(data[Ntrain:, :-1], axis=0)
    ytest = data[Ntrain:, -1][..., None]
    
    Xtrain_mt = np.concatenate((np.ones((len(Xtrain), 1)), Xtrain), axis=1)
    Xtest_mt = np.concatenate((np.ones((len(Xtest), 1)), Xtest), axis=1)
    
    skl_model = sl.LinearRegression().fit(Xtrain, ytrain)
    beta = ordinary_least_squares(Xtrain_mt, ytrain)
    
    mse_train.append(mse(ytrain, beta, Xtrain_mt))
    mse_test.append(mse(ytest, beta, Xtest_mt))
    mse_SL_train.append(mean_squared_error(ytrain, skl_model.predict(Xtrain)))
    mse_SL_test.append(mean_squared_error(ytest, skl_model.predict(Xtest)))


plt.plot(n,mse_train,color='red',label='train',marker='D')
plt.plot(n,mse_test,color='black',label='test',marker='D')
plt.plot(n,mse_SL_train,color='blue',label='sklearn train', linestyle=':')
plt.plot(n,mse_SL_test,color='yellow',label='sklearn test', linestyle=':')
plt.legend()
plt.ylabel('mse')
plt.xlabel('Size of training set')
#plt.show()
plt.savefig(".\\sln_figures\\fig4_2.png")