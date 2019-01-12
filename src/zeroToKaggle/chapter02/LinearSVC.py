# -*- coding: UTF-8 -*-

#从sklearn.datasets里导入手写体数字加载器
from sklearn.datasets import load_digits
#从通过数据加载器活得手写体数字的数码图像数据并存储在digits变量中
digits = load_digits()

print digits.data.shape


# 使用sklearn.model_selection里的train_test_split模块切分数据集，得到训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
print y_train






