# coding=utf-8
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# 加载鸢尾花数据集
iris_X, iris_y = datasets.load_iris(return_X_y=True)
# 数据预处理：按列归一化
iris_X = preprocessing.scale(iris_X)
# 切分数据集：测试集 30%
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_y, test_size=0.3, random_state=0)
# AdaBoost 分类模型
from sklearn import ensemble

model = ensemble.AdaBoostClassifier()
# 模型训练
model.fit(iris_X_train, iris_y_train)
# 模型预测
iris_y_pred = model.predict(iris_X_test)
# 模型评估
# 混淆矩阵
print(confusion_matrix(iris_y_test, iris_y_pred))
print("adaboost方法准确率: %.3f" % accuracy_score(iris_y_test, iris_y_pred))