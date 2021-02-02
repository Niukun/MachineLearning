from sklearn.datasets import load_iris

def dataSet():
    iris = load_iris()
    print(iris["data"].shape)
    return None

if __name__ == '__main__':
    dataSet()
