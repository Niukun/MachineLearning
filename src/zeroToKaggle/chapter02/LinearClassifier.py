# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
####################### 数据准备#######################
#创建特征列表
column_names = ['Sample code number',
                'Clump Thickness',
                'Uniformity of Cell Size',
                'Uniformity of Cell Shape',
                'Marginal Adhesion',
                'Single Epithelial Cell Size',
                'Bare Nuclei',
                'Bland Chromatin',
                'Normal Nucleoli',
                'Mitoses',
                'Class'
                ]

# 从互联网上读取数据
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                   names=column_names)
# 将？替换为标准缺失值表示
data = data.replace(to_replace="?",value=np.nan)

# 再将带有缺失值的数据丢弃
data = data.dropna(how='any')

print data.shape

# 使用sklearn.model_selection里的train_test_split模块切分数据集，得到训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)

####################### 查验训练样本的数量和类别分布#######################
print y_train.value_counts()

# 查验测试样本的数量和类别分布
print y_test.value_counts()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,SGDClassifier

# 标准化数据，保证每个维度的特征数据方差为1，均值为0。使得预测结果不会被某些维度过大的特征值主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# 初始化LogisticRegrassion和SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()

# 调用LogisticRegression中的fit函数来训练模型参数
lr.fit(X_train,y_train)
# 使用训练好的迷行lr对X_test进行预测，结果存储在变量lr_y_predict中
lr_y_predict = lr.predict(X_test)

# 调用SGDClaccifier中的fit函数用来训练模型参数
sgdc.fit(X_train,y_train)
# 使用训练好的模型sgdc对X_test进行测试，结果存储在变量sgdc_y_predict中
sgdc_y_predict = sgdc.predict(X_test)
print lr_y_predict
print sgdc_y_predict

####################### 性能分析#######################
from sklearn.metrics import classification_report
# 使用逻辑斯蒂回归模型自带的评分函数score活得模型在测试集上的准确性结果
print 'Accuracy of LR Classifier',lr.score(X_test,y_test)
# 利用classification_report模块活得LogisticRegression其他三个指标的结果
print classification_report(y_test,lr_y_predict,target_names=['Bengin','Malignant'])

#使用随机梯度下降模型自带的评分函数score活得模型在测试集上的准确性结果
print 'Accuracy of SGD Classifier:',sgdc.score(X_test,y_test)
#利用classification_report模块活得SGDClassifier其他三个指标的结果
print classification_report(y_test,sgdc_y_predict,target_names=['Benign','Malignant'])









