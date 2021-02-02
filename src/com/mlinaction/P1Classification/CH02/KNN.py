import operator

from numpy import array, tile


# 需要的方法不要先写，后面可以用快捷键： alt + Enter


def creaeDateSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]);
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = creaeDateSet()
print(group)
print(labels)
mat = array([[1,2], [3, 4],[5,6]])
print(mat)


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgeter(1),reverse=True)
    return sortedClassCount[0][0]

print(classify0([0,0], group,labels,3))