import numpy as np
import pandas as pd
import random
from functools import reduce

def randSplit(dataSet, rate):
    l = list(dataSet.index)  # 提取出索引
    random.shuffle(l)  # 随机打乱索引
    dataSet.index = l  # 将打乱后的索引重新赋值给原数据集
    n = dataSet.shape[0]  # 总行数
    m = int(n * rate)  # 训练集的数量
    train = dataSet.loc[range(m), :]  # 提取前m个记录作为训练集
    test = dataSet.loc[range(m, n), :]  # 剩下的作为测试集
    dataSet.index = range(dataSet.shape[0])  # 更新原数据集的索引
    test.index = range(test.shape[0])  # 更新测试集的索引
    return train, test

def gnb_classify(train,test):   #鲜花分类任务

    labels = train.iloc[:, -1].value_counts().index  # 提取训练集的标签种类
    mean = []  # 存放每个类别的均值
    std = []  # 存放每个类别的方差
    result = []  # 存放测试集的预测结果
    for i in labels:
        item = train.loc[train.iloc[:, -1] == i, :]  # 分别提取出每一种类别
        m = item.iloc[:, :-1].mean()  # 当前类别的平均值
        s = np.sum((item.iloc[:, :-1] - m) ** 2) / (item.shape[0])  # 当前类别的方差
        mean.append(m)  # 将当前类别的平均值追加至列表
        std.append(s)
    means = pd.DataFrame(mean, index=labels)  # 变成DF格式，索引为类标签
    stds = pd.DataFrame(std, index=labels)  # 变成DF格式，索引为类标签

    for j in range(test.shape[0]):
        iset = test.iloc[j, :-1].tolist()  # 当前测试实例
        iprob = np.exp(-1 * (iset - means) ** 2 / (stds * 2)) / (np.sqrt(2 * np.pi * stds))  # 正态分布公式
        prob = 1  # 初始化当前实例总概率
        for k in range(test.shape[1] - 1):  # 遍历每个特征
            prob *= iprob[k]  # 特征概率之积即为当前实例概率
            cla = prob.index[np.argmax(prob.values)]  # 返回最大概率的类别
        result.append(cla)
    test['predict'] = result
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()  # 计算预测准确率
    print("模型预测准确率为{:.2f}".format(acc))
    return test

############################# 文本分类
def loadDataSet():
    dataSet = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]  # 切分好的词条
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表侮辱性词汇，0代表非侮辱性词汇
    return dataSet, classVec

def createVocabList(dataSet):
    vocabSet = set()  # 创建一个空的集合
    for doc in dataSet:  # 遍历dataSet中的每一条言论
        vocabSet = vocabSet | set(doc)  # 取并集
    vocabList = list(vocabSet)
    return vocabList

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:  # 遍历每个词条
        if word in vocabList:  # 如果词条存在于词汇表中，则变为1
            returnVec[vocabList.index(word)] = 1
        else:
            print(f" {word} is not in my Vocabulary!")
    return returnVec  # 返回文档向量

def get_trainMat(dataSet):
    trainMat = []  # 初始化向量列表
    vocabList = createVocabList(dataSet)  # 生成词汇表
    for inputSet in dataSet:  # 遍历样本词条中的每一条样本
        returnVec = setOfWords2Vec(vocabList, inputSet)  # 将当前词条向量化
        trainMat.append(returnVec)  # 追加到向量列表中
    return trainMat

def trainNB(trainMat,classVec):
    n = len(trainMat)  # 计算训练的文档数目
    m = len(trainMat[0])  # 计算每篇文档的词条数
    pAb = sum(classVec) / n  # 文档属于侮辱类的概率
    p0Num = np.ones(m)  # 词条出现数初始化为1
    p1Num = np.ones(m)  # 词条出现数初始化为1
    p0Denom = 2  # 分母初始化为2
    p1Denom = 2  # 分母初始化为2
    for i in range(n):  # 遍历每一个文档
        if classVec[i] == 1:  # 统计属于侮辱类的条件概率所需的数据
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1V = p1Num / p1Denom
    p0V = p0Num / p0Denom
    return p0V, p1V, pAb

def classifyNB(vec2Classify, p0V, p1V, pAb):
    p1 = sum(vec2Classify * p1V) + np.log(pAb)  # 对应元素相乘
    p0 = sum(vec2Classify * p0V) + np.log(1 - pAb)  # 对应元素相乘
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB(testVec):
    dataSet, classVec = loadDataSet()  # 创建实验样本
    vocabList = createVocabList(dataSet)  # 创建词汇表
    trainMat = get_trainMat(dataSet)  # 将实验样本向量化
    p0V, p1V, pAb = trainNB(trainMat, classVec)  # 训练朴素贝叶斯分类器
    thisone = setOfWords2Vec(vocabList, testVec)  # 测试样本向量化
    if classifyNB(thisone, p0V, p1V, pAb):
        print(testVec, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(testVec, '属于非侮辱类')


if __name__ == '__main__':

    dataSet = pd.read_csv('iris.txt', header=None)
    for i in range(10):
        train, test = randSplit(dataSet, 0.8)
        gnb_classify(train, test)
# ######################
#     testVec1 = ['love', 'my', 'dalmation']
#     testingNB(testVec1)
#     testVec2 = ['stupid', 'garbage']
#     testingNB(testVec2)
