import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(inx):
    y = 1 / (1 + np.exp(-inx))
    return y

def classify(inX,weights):
    p = sigmoid(sum(inX * weights))
    if p < 0.5:
        return 0
    else:
        return 1

def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat,axis = 0)
    inVar = np.std(inMat,axis = 0)
    inMat = (inMat - inMeans)/inVar
    return inMat

def logisticAcc(dataSet, method, alpha=0.01, maxCycles=500):
    weights = method(dataSet, alpha=alpha, maxCycles=maxCycles)
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = regularize(xMat)
    p = sigmoid(xMat * weights).A.flatten()
    for i, j in enumerate(p):
        if j < 0.5:
            p[i] = 0
        else:
            p[i] = 1
    train_error = (np.fabs(yMat.A.flatten() - p)).sum()
    trainAcc = 1 - train_error / yMat.shape[0]
    return trainAcc

def BGD_LR(dataSet,alpha=0.001,maxCycles=500):
    xMat = np.mat(dataSet.iloc[:,:-1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    xMat = regularize(xMat)
    m,n = xMat.shape
    weights = np.zeros((n,1))
    for i in range(maxCycles):
        grad = xMat.T*(xMat * weights-yMat)/m
        weights = weights -alpha*grad
    return weights

def SGD_LR(dataSet,alpha=0.001,maxCycles=500):
    dataSet = dataSet.sample(maxCycles, replace=True)
    dataSet.index = range(dataSet.shape[0])
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    xMat = regularize(xMat)
    m, n = xMat.shape
    weights = np.zeros((n,1))
    for i in range(m):
        grad = xMat[i].T * (xMat[i] * weights - yMat[i])
        weights = weights - alpha * grad
    return weights

def get_acc(train,test,alpha=0.001, maxCycles=5000):
    weights = SGD_LR(train,alpha=alpha,maxCycles=maxCycles)
    xMat = np.mat(test.iloc[:, :-1].values)
    xMat = regularize(xMat)
    result = []
    for inX in xMat:
        label = classify(inX, weights)
        result.append(label)
    retest = test.copy()
    retest['predict'] = result
    acc = (retest.iloc[:, -1] == retest.iloc[:, -2]).mean()
    print(f'使用SGD方法模型的准确率为：{acc}')
    return retest

def get_acc2(train,test,alpha=0.001, maxCycles=5000):
    weights = BGD_LR(train,alpha=alpha,maxCycles=maxCycles)
    xMat = np.mat(test.iloc[:, :-1].values)
    xMat = regularize(xMat)
    result = []
    for inX in xMat:
        label = classify(inX, weights)
        result.append(label)
    retest = test.copy()
    retest['predict'] = result
    acc = (retest.iloc[:, -1] == retest.iloc[:, -2]).mean()
    print(f'使用BGD方法模型的准确率为：{acc}')
    return retest
if __name__ == '__main__':

    train = pd.read_table('horseColicTraining.txt', header=None)
    test = pd.read_table('horseColicTest.txt', header=None)
    # data = {'x': [3, 4, 1], 'y': [3, 3, 1], 'L': [1, 1, 0]}
    # dataSet = pd.DataFrame(data)
    a = get_acc(train, test, alpha=0.01, maxCycles=5000)
    b = get_acc2(train, test, alpha=0.01, maxCycles=5000)
