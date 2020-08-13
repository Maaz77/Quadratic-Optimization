
import numpy as np


def ternarysearch(func, left, right):
    left = float(left)
    right = float(right)
    for i in range(100):
        left_third = ((2 * left) + right) / 3
        right_third = (left + (2 * right)) / 3
        if func(left_third) > func(right_third):
            (left, right) = (left_third, right)
        else:
            (left, right) = (left, right_third)

    return ((left + right) / 2, func((left + right) / 2))




def binarysearch(func, left, right, dir):
    left = float(left)
    right = float(right)
    
    for i in range(100):

        mid = (left + right) / 2
        value = func(mid)
        
        if dir == 0:
            if value < 1:
                left = mid
            else:
                right = mid
        if (dir == 1):
            if value < 1:
                right = mid
            else:
                left = mid

    return (mid, value)


def loadData():
    
    dataset = np.loadtxt('dataset.txt')
    n = int(dataset[0])
    
    return dataset[1 : n+1].reshape(n,1) , dataset[n+1 : ].reshape(n,n) ,  n











def main(i):
    
    
    
    y , A ,n  = loadData()
    
    
    landa, v = np.linalg.eig(A)
    
    landa = landa.reshape(n, 1)
    
    
    
    def function(x):
        
        x = float(x)
        
        
        ytilada = y.T.dot(v)
        
        ytilada = ytilada.T #????ghablan error nemidad idk
        
        temp = 0
        
        
        for i in range(np.shape(landa)[0]):
            temp += pow((ytilada[i] / (x - (2 * landa[i]))), 2)
        
        return temp
    
    def solution(x):
        
        term = (np.eye(len(landa), len(landa)) * x) - 2 * np.diagflat(landa)
        
        for i in range(len(landa)):
            if (term[i][i] != 0):
                term[i][i] = 1. / term[i][i]
        
        return v.dot(term).dot(v.T).dot(y)

    def model(x):
    
        return x.T.dot(A).dot(x) + x.T.dot(y)
    
    def solve_model():
        
        eigvals = []
        for i in range(len(landa)):
            eigvals.append(landa[i])
        
        eigvals.sort()
        
        candidate_solutions = []
        candidate_rslts = []
        
        candidate_solutions.append(binarysearch(function, -1000, 2 * eigvals[0], 0)[0])
        candidate_solutions.append(binarysearch(function, 2 * eigvals[-1], 1000, 1)[0])
        #print(candidate_solutions[0],candidate_solutions[1])
        
        
        # for i in range(1, len(eigvals) - 2):
        
        #     localmin = ternarysearch(function, eigvals[i] * 2, eigvals[i + 1] * 2)
        #     #print localmin
        #     if (localmin[1] == 1):
        #         candidate_solutions.append(localmin[0])
        
        
        #     elif (localmin[1] < 1):
        #         candidate_solutions.append(binarysearch(function, eigvals[i] * 2, localmin[0], 0)[0])
        #         candidate_solutions.append(binarysearch(function, localmin[0], 2*eigvals[i + 1], 1)[0])
        
        
        for i in range(len(candidate_solutions)):
            sol = solution(candidate_solutions[i])
            sol /=np.linalg.norm(sol)     #why ??????????
            #print model(sol)
            candidate_rslts.append(model(sol))
        
        
        minindex = candidate_rslts.index(min(candidate_rslts))
        maxindex = candidate_rslts.index(max(candidate_rslts))
        
        #print("len : ",len(candidate_rslts),"min index : ",minindex,"max index : ",maxindex)
        
        # if(len(candidate_solutions) > 2):
        #     print("here is N : ",n)
        print( minindex, maxindex)
        print (candidate_rslts[minindex])


    solve_model()


main(0)
