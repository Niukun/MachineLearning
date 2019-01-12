# -*- coding: UTF-8 -*-

from sklearn.datasets import load_iris
iris = load_iris()

print iris.data.shape
# print iris.DESCR

# 使用sklearn.model_selection里的train_test_split模块切分数据集，得到训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
print y_train

# 从sklearn.preprocessing里选择导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors里选择导入KNeighborsClassifier，即K近邻分类器
from sklearn.neighbors import KNeighborsClassifier

# 对训练和测试数据集进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 舒勇K近邻分类器对测试数据进行类别预测
knc = KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict = knc.predict(X_test)


# 评估
print 'The accuracy of K-Nearest Neighbor Classifier is:',knc.score(X_test,y_test)

from sklearn.metrics import classification_report
print classification_report(y_test,y_predict,target_names=iris.target_names)






























