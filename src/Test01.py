import sys



cat = [1,2,3,4,"aaa",[11,22,33,[111,222,3333]]]

def test(llist):
    for i in llist:
        if(isinstance(i,list)):
            test(i)
        else:
            print(i)
# test(sys.path)
a=2
b=a
a=6
print(a)#6
print(b)#2