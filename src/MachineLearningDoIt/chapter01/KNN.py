# -*- coding: UTF-8 -*-

from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
# K近邻分类实现
def classify(inX,dataSet,labels,k):
    dataSetSize =  dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    print('distances is: ', distances)
    sortedDistances = distances.argsort()
    print('sortedDistances is: ', sortedDistances)
    classCount = {}
    for i in range(k):
        print('sortedDistances[i] is: ', sortedDistances[i])
        voteLabel = labels[sortedDistances[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    print('classCount is: ', classCount)
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    print('sortedClassCount is: ', sortedClassCount)
    return sortedClassCount[0][0]
dataSet,labels = createDataSet()
result = classify([1,1],dataSet,labels,3)
print(result)
print('=======================')
result = classify([0,0],dataSet,labels,3)
print(result)