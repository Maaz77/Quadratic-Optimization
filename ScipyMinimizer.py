from sklearn import datasets
import numpy as np
from scipy.optimize import minimize
def loadData():
    iris = datasets.load_iris()
    iris = iris.data
    iris = np.cov(iris.T)
    return iris

def model(x):
    A =  loadData()
    y = [1, 2, 3, 4]
    return x.T.dot(A).dot(x)+x.T.dot(y)


def norm_constraint(x):
    return x.T.dot(x) - 1



con = {'type': 'eq', 'fun': norm_constraint}
cons = ([con])
x0 = np.array([1.,2.,0.,0.])

solutionn = minimize(model,x0,method='SLSQP',constraints=cons)

print(solutionn)

