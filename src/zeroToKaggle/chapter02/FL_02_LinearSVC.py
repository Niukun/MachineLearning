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

# 从sklearn.preprocessing里导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.svm里导入基于现行假设的支持向量机分类器LinearSVC
from sklearn.svm import LinearSVC

# 对训练集和测试集进行标准化
ss=StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC()
lsvc.fit(X_train,y_train)
#利用训练好的模型对测试数据集进行预测
y_predict = lsvc.predict(X_test)

#############################性能评估#################################3
#使用模型自带的评估函数进行准确性测评
print 'Accutacy of LinearSVC is:', lsvc.score(X_test,y_test)
# 使用sklearn.metric里面的classification_report模块对预测结果作更加详细的分析
from sklearn.metrics import classification_report
print classification_report(y_test,y_predict,target_names=digits.target_names.astype(str))


