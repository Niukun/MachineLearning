# -*- coding: UTF-8 -*-

from numpy import *
randMat = mat(random.rand(4,4))
print('randMat is: \n', randMat)
print('randMat.I is: \n', randMat.I)
print('randMat * randMat.I is: \n', randMat * randMat.I)


print('randMat * randMat.I - sys(4)is: \n', randMat * randMat.I-eye(4))