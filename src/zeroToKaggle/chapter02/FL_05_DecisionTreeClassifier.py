# -*- coding: UTF-8 -*-

import pandas as pd
titanic = pd.read_csv('../../../data/titanic.txt')
print titanic.shape
# print titanic.head()
# print titanic.info()

X = titanic[['pclass','age','sex']]
y = titanic[['survived']]
print X.info()
# Data columns (total 3 columns):
# pclass    1313 non-null object
# age       633 non-null float64
# sex       1313 non-null object
# dtypes: float64(1), object(2)

# 1.age这个数据列只有633条数据，需要补充完整
# 2.sex与pclass两列都是类别型的，需要转化为数值，用0/1代替
# 首先补充age里面数据，使用平均值或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(),inplace=True)
print X.info()

# 数据分割
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)

# 使用sklearn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

# 转换特征后，我们发现凡是类别型的特征都单独剥离出来，独成一列特征，数值型的则保持不变
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.fit_transform(X_test.to_dict(orient='record'))
print vec.feature_names_

# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict = dtc.predict(X_test)

# 性能
from sklearn.metrics import classification_report
print dtc.score(X_test,y_test)
print classification_report(y_predict,y_test,target_names=['died','survived'])





















