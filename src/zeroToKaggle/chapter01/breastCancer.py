# -*- coding: UTF-8 -*-
import pandas as pd

# 调用工具包里面的方法读取csv中的文件，保存训练集和测试集
df_train = pd.read_csv('../../../data/Breast-Cancer/breast-cancer-train.csv')
df_test = pd.read_csv('../../../data/Breast-Cancer/breast-cancer-test.csv')

# 选取'Clump Thickness','Cell Size'两列作为特征，构建测试集中的正负分类样本
df_test_negative = df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
df_test_positive = df_test.loc[df_test['Type']==1][['Clump Thickness','Cell Size']]

# 测试打印出来的结果最左侧一列是该数据在原集合中的行号
# print df_test_positive

# 导入工具包准备画图
import matplotlib.pyplot as plt

# 绘制良性肿瘤、恶性肿瘤，用不同的符号跟颜色表示
# scatter中参数分别是特征1值，特征2值，标记符号，标记大小，标记颜色
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='x',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='o',s=150,c='black')

# 绘制x,y轴
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
# plt.show()

import numpy as np

# 随机生成一组参数
intercept = np.random.random(1)
coef = np.random.random(2)

# 绘制出这条直线
lx = np.arange(0,12)
ly = (-intercept - lx*coef[0])/coef[1]
# plt.plot(lx,ly,c='yellow')

# 导入sklearn中的逻辑斯蒂回归分类器
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# 使用前十条训练样本学习直线的系数和结局
# lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][:10])

# 使用所有训练样本
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print 'Testing accuracy(10 traning samples):',lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type'])

# 得到真正的系数
intercept = lr.intercept_
coef = lr.coef_[0,:]

ly = (-intercept - coef[0]*lx)/coef[1]

plt.plot(lx,ly,c='green')


# 显示所有图形
plt.show()