import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def loadDataSet(file):
    dataSet= pd.read_table(file,header = None)
    xMat=np.mat(dataSet.iloc[:,:-1].values)
    yMat=np.mat(dataSet.iloc[:,-1].values).T
    return xMat, yMat

def showDataSet(xMat, yMat):
    data_p = []  # 正样本
    data_n = []  # 负样本
    m = xMat.shape[0]  # 样本总数
    for i in range(m):
        if yMat[i] > 0:
            data_p.append(xMat[i])
    else:
        data_n.append(xMat[i])
    data_p_ = np.array(data_p)  # 转换为numpy矩阵
    data_n_ = np.array(data_n)  # 转换为numpy矩阵
    plt.scatter(data_p_.T[0], data_p_.T[1])  # 正样本散点图
    plt.scatter(data_n_.T[0], data_n_.T[1])  # 负样本散点图
    plt.show()

def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(xMat,yMat,C,toler,maxIter):
    b = 0  # 初始化b参数
    m, n = xMat.shape  # m为数据集的总行数，n为特征的数量
    alpha = np.mat(np.zeros((m, 1)))  # 初始化alpha参数，设为0
    iters = 0  # 初始化迭代次数
    while (iters < maxIter):
        alpha_ = 0  # 初始化alpha优化次数
        for i in range(m):
        # 步骤1：计算误差Ei
            fXi = np.multiply(alpha, yMat).T * (xMat * xMat[i, :].T) + b
            Ei = fXi - yMat[i]
        # 优化alpha，设定容错率
        if ((yMat[i] * Ei < -toler) and (alpha[i] < C)) or ((yMat[i] * Ei > toler) and (alpha[i] > 0)):
            # 随机选择一个与alpha_i成对优化的alpha_j
            j = selectJrand(i, m)
            # 步骤1：计算误差Ej
            fXj = np.multiply(alpha, yMat).T * (xMat * xMat[j, :].T) + b
            Ej = fXj - yMat[j]
            # 保存更新前的alpha_i和alpha_j
            alphaIold = alpha[i].copy()
            alphaJold = alpha[j].copy()
            # 步骤2：计算上下界H和L
            if (yMat[i] != yMat[j]):
                L = max(0, alpha[j] - alpha[i])
                H = min(C, C + alpha[j] - alpha[i])
            else:
                L = max(0, alpha[j] + alpha[i] - C)
                H = min(C, C + alpha[j] + alpha[i])
            if L == H:
                # print('L==H')
                continue
            # 步骤3：计算学习率eta(eta是alpha_j的最优修改量)
            eta = 2 * xMat[i, :] * xMat[j, :].T - xMat[i, :] * xMat[i, :].T - xMat[j, :] * xMat[j, :].T
            if eta >= 0:
                # print('eta>=0')
                continue
            # 步骤4：更新alpha_j
            alpha[j] -= yMat[j] * (Ei - Ej) / eta
            # 步骤5：修剪alpha_j
            alpha[j] = clipAlpha(alpha[j], H, L)
            if abs(alpha[j] - alphaJold) < 0.00001:
                # print('alpha_j 变化太小')
                continue
            # 步骤6：更新alpha_i
            alpha[i] += yMat[j] * yMat[i] * (alphaJold - alpha[j])
            # 步骤7：更新b_1和b_2
            b1 = b - Ei - yMat[i] * (alpha[i] - alphaIold) * xMat[i, :] * xMat[i, :].T - yMat[j] * (alpha[j] - alphaJold) * xMat[i, :] * xMat[j, :].T
            b2 = b - Ej - yMat[i] * (alpha[i] - alphaIold) * xMat[i, :] * xMat[j, :].T - yMat[j] * (alpha[j] - alphaJold) * xMat[j, :] * xMat[j, :].T
            # 步骤8：根据b_1和b_2更新b
            if (0 < alpha[i]) and (C > alpha[i]):
                b = b1
            elif (0 < alpha[j]) and (C > alpha[j]):
                b = b2
            else:
                b = (b1 + b2) / 2
            alpha_ += 1
        if alpha_ == 0:
            iters += 1
        else:
            iters = 0
    return b, alpha
def get_sv(xMat,yMat,alpha):
    m = xMat.shape[0]
    sv_x = []
    sv_y = []
    for i in range(m):
        if alpha[i] > 0:
            sv_x.append(xMat[i])
            sv_y.append(yMat[i])
    sv_x1 = np.array(sv_x).T
    sv_y1 = np.array(sv_y).T
    return sv_x1, sv_y1
def showPlot(xMat, yMat,alpha):
    data_p = []  # 正样本
    data_n = []  # 负样本
    m = xMat.shape[0]  # 样本总数
    for i in range(m):
        if yMat[i] > 0:
            data_p.append(xMat[i])
        else:
            data_n.append(xMat[i])
    data_p_ = np.array(data_p)  # 转换为numpy矩阵
    data_n_ = np.array(data_n)  # 转换为numpy矩阵
    # 样本散点图
    plt.scatter(data_p_.T[0], data_p_.T[1])  # 正样本散点图
    plt.scatter(data_n_.T[0], data_n_.T[1])  # 负样本散点图
    # 绘制支持向量
    sv_x, sv_y = get_sv(xMat, yMat, alpha)
    plt.scatter(sv_x[0], sv_x[1], s=150, c='none', alpha=0.7, linewidth=1.5,edgecolor='red')
if __name__ == '__main__':
    x,y = loadDataSet('iris.txt')
    showDataSet(x,y)