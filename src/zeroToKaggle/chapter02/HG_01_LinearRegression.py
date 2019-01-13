# -*- coding: UTF-8 -*-
# 美国波士顿地区房地产数据
#TODO error need to be handled

from sklearn.datasets import load_boston
boston = load_boston()
# print boston.DESCR

# 数据分割
from sklearn.model_selection import train_test_split
import numpy as np
X = boston.data
y = boston.target
print y
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)

print 'The max target value is:',np.max(boston.target)
print 'The min target value is:',np.min(boston.target)
print 'The average target is:',np.mean(boston.target)
print 'y shape:',y.shape

# 数据标准化
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()

X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)

y_train = ss_y.fit_transform(y_train)
y_test = ss_y.fit_transform(y_test)

# 使用线性回归模型LinearRegression和SGDRegressor分别进行训练和预测
from sklearn.linear_model import LinearRegression
# 使用默认配置初始化线性回归器LinearRegression
lr = LinearRegression()
# 使用训练数据进行参数估计
lr.fit(X_train,y_train)
# 对测试数据进行回归预测
lr_y_predict = lr.predict(X_test)

from sklearn.linear_model import SGDRegressor
sgdc = SGDRegressor()
sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)

# 使用三种回归评价机制以及两种调用R-squared评价模块的方法，对本届模型的回归性能做出评价
print 'The value of default measurement of LinearRegression is:', lr.score(X_test,y_test)
print 'The value of default measurement of SGDRegressor is:', sgdc.score(X_test,y_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print 'The value of R-squared of LinearRegression is:', r2_score(y_test,lr_y_predict)
print 'The mean squared error of LinearRegression is:', mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))
print 'The mean absolute error of LinearRegression is:', mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict))

print 'The value of R-squared of LinearRegression is:', r2_score(y_test,sgdc_y_predict)
print 'The mean squared error of LinearRegression is:', mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdc_y_predict))
print 'The mean absolute error of LinearRegression is:', mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdc_y_predict))

































