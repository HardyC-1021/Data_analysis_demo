# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
def classify0(inX,dataSet,k):
    movie_data = pd.DataFrame(dataSet)
    dist = (((movie_data.iloc[:, 1:3] - inX) ** 2).sum(1)) ** 0.5
    dist_l = pd.DataFrame({'dist':dist, 'labels':movie_data.iloc[:,3]})
    dr = dist_l.sort_values(by = 'dist')[:k]
    re = dr.loc[:, 'labels'].value_counts()
    result = []
    result.append(re.index[0])
    return result

def minmax(dataSet): # 原理不懂
    minDf = dataSet.min()
    print(minDf)
    maxDf = dataSet.max()
    normSet = (dataSet - minDf)/(maxDf - minDf)
    return normSet

def randSplit(dataSet,rate=0.9):
    n = dataSet.shape[0]
    m = int(n*rate)
    train = dataSet.iloc[:m,:]
    test = dataSet.iloc[m:,:]
    test.index = range(test.shape[0])
    return train,test

def flowerClass(train,test,k):
    n = train.shape[1] - 1
    m = test.shape[0]
    result = []
    for i in range(m):
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) ** 2).sum(1))**0.5)
        dist_l = pd.DataFrame({'dist': dist, 'labels': (train.iloc[:, n])})
        dr = dist_l.sort_values(by = 'dist')[: k]
        re = dr.loc[:, 'labels'].value_counts()
        result.append(re.index[0])
    result = pd.Series(result)
    test.insert(len(test.columns),'predict',result)
    test['predict'] = result
    print(test.head())
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()
    print(f'模型预测准确率为{acc}')
    return test

if __name__ == '__main__':
    datingTest = pd.read_table('iris.txt', sep=',',header=None)
    print(datingTest.head())
    Colors = []
    for i in range(datingTest.shape[0]):
        m = datingTest.iloc[i, -1]
        if m == 'Iris-setosa':
            Colors.append('black')
        if m == 'Iris-versicolor':
            Colors.append('orange')
        if m == 'Iris-virginica':
            Colors.append('red')

    datingT = pd.concat([minmax(datingTest.iloc[:, :4]), datingTest.iloc[:, 4]], axis=1)
    train, test = randSplit(datingT)
    t = flowerClass(train,test,4)

    # 绘制两两特征之间的散点图
    plt.rcParams['font.sans-serif'] = ['Simhei']  # 图中字体设置为黑体
    pl = plt.figure(figsize=(12, 8))
    fig1 = pl.add_subplot(221)
    plt.scatter(datingTest.iloc[:, 1], datingTest.iloc[:, 2], marker='.', c=Colors)
    plt.xlabel('鲜花的特征2')
    plt.ylabel('鲜花的特征3')
    fig2 = pl.add_subplot(222)
    plt.scatter(datingTest.iloc[:, 0], datingTest.iloc[:, 1], marker='.', c=Colors)
    plt.xlabel('鲜花的特征1')
    plt.ylabel('鲜花的特征2')
    fig3 = pl.add_subplot(223)
    plt.scatter(datingTest.iloc[:, 0], datingTest.iloc[:, 2], marker='.', c=Colors)
    plt.xlabel('鲜花的特征1')
    plt.ylabel('鲜花的特征3')
    fig4 = pl.add_subplot(224)
    plt.scatter(datingTest.iloc[:, 0], datingTest.iloc[:, 3], marker='.', c=Colors)
    plt.xlabel('鲜花的特征1')
    plt.ylabel('鲜花的特征4')
    plt.show()