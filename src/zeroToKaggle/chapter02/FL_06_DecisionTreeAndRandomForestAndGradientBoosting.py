# -*- coding: UTF-8 -*-

import pandas as pd
titanic = pd.read_csv('../../../data/titanic.txt')

X = titanic[['pclass','age','sex']]
y = titanic['survived']

# 用平均年龄填充缺失的年龄信息
X['age'].fillna(X['age'].mean(),inplace=True)

# 分割训练集测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

# 对类别型特征进行转化，成为特征向量
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

###########################分别进行模型训练和预测#############################
#单一决策树
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_predict = dtc.predict(X_test)

#随机森林
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_predict = rfc.predict(X_test)

#梯度上升决策树
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_predict= gbc.predict(X_test)


########################性能评估######################
from sklearn.metrics import classification_report
#单一决策树
print 'Decision Tree:',dtc.score(X_test,y_test)
print classification_report(dtc_y_predict,y_test)

#随机森林
print 'Random Decision Tree:',rfc.score(X_test,y_test)
print classification_report(rfc_y_predict,y_test)

#梯度上升决策树
print 'Gradient Boosting Tree:',gbc.score(X_test,y_test)
print classification_report(gbc_y_predict,y_test)