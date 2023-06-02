import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['simhei'] # 让title可以中文

def get_Mat(path):
    dataSet = pd.read_table(path, header=None)
    xMat = np.mat(dataSet.iloc[:, :-1].values)
    yMat = np.mat(dataSet.iloc[:, -1].values).T
    return xMat, yMat

def showPlot(xMat,yMat):
    x = np.array(xMat[:, 0])
    y = np.array(xMat[:, 1])
    label = np.array(yMat)
    plt.scatter(x, y, c=label)
    plt.title('单层决策树测试数据')
    plt.show()

def Classify0(xMat,i,Q,S):
    re = np.ones((xMat.shape[0], 1))  # 初始化re为1
    if S == 'lt':
        re[xMat[:, i] <= Q] = -1  # 如果小于阈值,则赋值为-1
    else:
        re[xMat[:, i] > Q] = -1  # 如果大于阈值,则赋值为-1
    return re

def get_Stump(xMat,yMat,D):
    m, n = xMat.shape  # m为样本个数，n为特征数
    Steps = 10  # 初始化一个步数
    bestStump = {}  # 用字典形式来储存树桩信息
    bestClas = np.mat(np.zeros((m, 1)))  # 初始化分类结果为1
    minE = np.inf  # 最小误差初始化为正无穷大
    for i in range(n):  # 遍历所有特征
        Min = xMat[:, i].min()  # 找到特征中最小值
        Max = xMat[:, i].max()  # 找到特征中最大值
        stepSize = (Max - Min) / Steps  # 计算步长
        for j in range(-1, int(Steps) + 1):
            for S in ['lt', 'gt']:
                Q = (Min + j * stepSize)  # 计算阈值
                re = Classify0(xMat, i, Q, S)  # 计算分类结果
                err = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                err[re == yMat] = 0  # 分类正确的,赋值为0
                eca = D.T * err  # 计算误差
                if eca < minE:  # 找到误差最小的分类方式
                    minE = eca
                    bestClas = re.copy()
                    bestStump['特征列'] = i
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump, minE, bestClas

def Ada_train(xMat, yMat, maxC = 40):
    weakClass = []
    m = xMat.shape[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化权重
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(maxC):
        Stump, error, bestClas = get_Stump(xMat, yMat, D)  # 构建单层决策树
        # print(f"D:{D.T}")
        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))  # 计算弱分类器权重alpha
        Stump['alpha'] = np.round(alpha, 2)  # 存储弱学习算法权重,保留两位小数
        weakClass.append(Stump)  # 存储单层决策树
        # print("bestClas: ", bestClas.T)
        expon = np.multiply(-1 * alpha * yMat, bestClas)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        aggClass += alpha * bestClas  # 更新累计类别估计值
        # print(f"aggClass: {aggClass.T}" )
        aggErr = np.multiply(np.sign(aggClass) != yMat, np.ones((m, 1)))  # 计算误差
        errRate = aggErr.sum() / m
        # print(f"分类错误率: {errRate}")
        if errRate == 0: break
    return weakClass, aggClass

def AdaClassify(data,weakClass):
    dataMat = np.mat(data)
    m = dataMat.shape[0]
    aggClass = np.mat(np.zeros((m, 1)))
    for i in range(len(weakClass)):  # 遍历所有分类器，进行分类
        classEst = Classify0(dataMat, weakClass[i]['特征列'], weakClass[i]['阈值'], weakClass[i]['标志'])
        aggClass += weakClass[i]['alpha'] * classEst
    return np.sign(aggClass)

def calAcc(maxC = 40):
    train_xMat, train_yMat = get_Mat('horseColicTraining2.txt')
    m = train_xMat.shape[0]
    weakClass, aggClass = Ada_train(train_xMat, train_yMat, maxC)
    yhat = AdaClassify(train_xMat, weakClass)
    train_re = 0
    for i in range(m):
        if yhat[i] == train_yMat[i]:
            train_re += 1
    train_acc = train_re / m
    print(f'训练集准确率为{train_acc}')


    test_re = 0
    test_xMat, test_yMat = get_Mat('horseColicTest2.txt')
    n = test_xMat.shape[0]
    yhat = AdaClassify(test_xMat, weakClass)
    for i in range(n):
        if yhat[i] == test_yMat[i]:
            test_re += 1
    test_acc = test_re / n
    print(f'测试集准确率为{test_acc}')
    return train_acc, test_acc

if __name__ == '__main__':
    # x,y = get_Mat('simpdata.txt')
    # m = x.shape[0]
    # D = np.mat(np.ones((m, 1)) / m)  # 初始化样本权重（每个样本权重相等）
    # weakClass, aggClass = Ada_train(x, y, maxC=40)
    # print(weakClass,aggClass)
    Cycles = [50]
    train_acc = []
    test_acc = []
    for maxC in Cycles:
        a, b = calAcc(maxC)
        train_acc.append(round(a * 100, 2))
        test_acc.append(round(b * 100, 2))
    df = pd.DataFrame({'分类器数目': Cycles, '训练集准确率': train_acc,'测试集准确率': test_acc})
    print(df)