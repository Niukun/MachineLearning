# -*- coding: UTF-8 -*-
from sklearn.svm import SVR
from sklearn.datasets import load_boston
boston = load_boston()

# 数据分割
from sklearn.model_selection import train_test_split
import numpy as np
X = boston.data
y = boston.target
print y
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=33,test_size=0.25)

#使用     线性核函数   配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel='liner')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)



#使用     多项式核函数      配置的支持向量机进行回归训练，并且对测试样本进行预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict = linear_svr.predict(X_test)


#使用     径向基核函数      配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict = linear_svr.predict(X_test)









