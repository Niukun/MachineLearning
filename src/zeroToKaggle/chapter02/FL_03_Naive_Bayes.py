# -*- coding: UTF-8 -*-
# //TO-DO

#从sklearn.datasets里面导入新闻数据抓取器fetch_20newgroups
from sklearn.datasets import fetch_20newsgroups
import gzip
#该数据需要即时从网上下载
with gzip.open('../../../data/20news-bydate.tar.gz','rb') as f:
    file_content = f.read()
    # print file_content
str = 'Some people might Invisible Pink Unicorn does not exist?'
print str
# print len(news.data)



# 使用sklearn.model_selection里的train_test_split模块切分数据集，得到训练集和测试集
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# print y_train


from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
str = vec.fit_transform(str)
print str


















































