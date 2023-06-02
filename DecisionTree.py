import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import graphviz

def calEnt(dataSet):
    n = dataSet.shape[0]
    iset = dataSet.iloc[:, -1].value_counts()
    p = iset / n
    ent = (-p * np.log2(p)).sum()
    return ent

def createDataSet():
    row_data = {'no surfacing': [1, 1, 1, 0, 0],
                'flippers': [1, 1, 0, 1, 1],
                'fish': ['yes', 'yes', 'no', 'no', 'no']}
    dataSet = pd.DataFrame(row_data)

    return dataSet
def DataSet():
    row_data = {'年龄': [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],'有工作': [0,0,1,1,0,0,0,1,0,0,0,0,1,1,0],
                '有房子': [0,0,0,1,0,0,0,1,1,1,1,1,0,0,0],'信贷情况': [0,1,1,0,0,0,1,1,2,2,2,1,1,2,0],
                'fish': ['no','no','yes', 'yes', 'no', 'no', 'no','yes','yes','yes','yes','yes','yes','yes','no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet
def bestSplit(dataSet):
    baseEnt = calEnt(dataSet)
    bestGain = 0
    axis = -1
    for i in range(dataSet.shape[1]-1):
        Dv = dataSet.iloc[:,i].value_counts().index
        ents = 0
        for j in Dv:
            childSet = dataSet[dataSet.iloc[:,i]==j]
            ent = calEnt(childSet)
            ents += (childSet.shape[0]/dataSet.shape[0])*ent
        infoGain = baseEnt - ents
        if (infoGain > bestGain):
            bestGain = infoGain
            axis = i
    return axis

def mySplit(dataSet,axis,value):
    col = dataSet.columns[axis]
    redataSet = dataSet.loc[dataSet[col] == value, :].drop(col, axis=1)
    return redataSet

def createTree(dataSet):
    featlist = list(dataSet.columns)  # 提取出数据集所有的列
    classlist = dataSet.iloc[:, -1].value_counts()  # 获取最后一列类标签
    if classlist[0] == dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0]  # 如果是，返回类标签
    axis = bestSplit(dataSet)  # 确定出当前最佳切分列的索引
    bestfeat = featlist[axis]  # 获取该索引对应的特征
    myTree = {bestfeat: {}}  # 采用字典嵌套的方式存储树信息
    del featlist[axis]  # 删除当前特征
    valuelist = set(dataSet.iloc[:, axis])  # 提取最佳切分列所有属性值
    for value in valuelist:  # 对每一个属性值递归建树
        myTree[bestfeat][value] = createTree(mySplit(dataSet, axis, value))
    return myTree

def classify(inputTree,labels, testVec):
    firstStr = next(iter(inputTree)) #获取决策树第一个节点
    secondDict = inputTree[firstStr] #下一个字典
    featIndex = labels.index(firstStr) #第一个节点所在列的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict :
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def acc_classify(train,test):
    inputTree = createTree(train) #根据测试集生成一棵树
    print('tree',inputTree)
    labels = list(train.columns) #数据集所有的列名称
    result = []
    for i in range(test.shape[0]): #对测试集中每一条数据进行循环
        testVec = test.iloc[i,:-1] #测试集中的一个实例
        classLabel = classify(inputTree,labels,testVec) #预测该实例的分类
        result.append(classLabel) #将分类结果追加到result列表中
    test['predict']=result #将预测结果追加到测试集最后一列
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean() #计算准确率
    print(f'模型预测准确率为{acc}')
    return test

if __name__ == '__main__':

    # np.save('myTree.npy', mytree)
    # read_myTree = np.load('myTree.npy',allow_pickle=True).item()
    dataset = DataSet()
    # train = dataset
    # test = dataset.iloc[:3, :]
    # acc_classify(train, test)
    # print(test)

    # 特征
    Xtrain = dataset.iloc[:, :-1]
    # 标签
    Ytrain = dataset.iloc[:, -1]
    labels = Ytrain.unique().tolist()
    Ytrain = Ytrain.apply(lambda x: labels.index(x))  # 将本文转换为数字
    # 绘制树模型
    clf = DecisionTreeClassifier()
    clf = clf.fit(Xtrain, Ytrain)
    tree.export_graphviz(clf)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graphviz.Source(dot_data)
    # 给图形增加标签和颜色
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=['age','work','house','credibility'],
                                    class_names=['loan', 'not loan'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graphviz.Source(dot_data)
    # 利用render方法生成图形
    graph = graphviz.Source(dot_data)
    graph.render("loan")
    graph.view()





