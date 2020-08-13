import os
from sklearn import datasets
from tqdm import tqdm
import numpy as np

def createdataset(n,symm):

    file1 = open('/Users/maaz/Desktop/optimization/SSM/SSM/dataset.txt','w')
    file2 = open('/Users/maaz/Desktop/optimization/OptimizationCode/dataset.txt','w')
    file3 = open('/Users/maaz/Desktop/optimization/GradientMethod/GradientMethod/dataset.txt', 'w')


    print >> file1,n
    print >> file2,n
    print >> file3,n

    if(symm ==1 ):
        data = datasets.make_spd_matrix(n) #Generate a random symmetric, positive-definite matrix.

    else:
        data = np.random.rand(n,n)
        data = data+data.T


    c = np.random.rand(n,1)
    #c = np.arange(1,n+1)
    c = c.reshape(n,1)

    for i in c:
        print  >> file1,i[0] 
        print  >> file2,i[0]
        print  >> file3,i[0]

    for item in data:
        for elements in item:
            print >> file3, elements
            print >> file1,elements
            print >> file2, elements

    file1.close()
    file2.close()
    file3.close()


def report():

    n = 5
    SSMpath = "cd /Users/maaz/Desktop/optimization/SSM/SSM"
    GDNTpath  = "cd /Users/maaz/Desktop/optimization/GradientMethod/GradientMethod"

    os.system(GDNTpath + "\n gcc -framework Accelerate  -o main.o main.c")
    os.system(SSMpath + "\n gcc -o main.o main.c SSM.c")
    
    for j in tqdm(range(52)):
        

        # createdataset(n,1)

        # os.system(GDNTpath + "\n ./main.o")
        # os.system(SSMpath + "\n ./main.o")

        # myfile = open("/Users/maaz/Desktop/report.txt", 'a')
        # myfile.write("..........................................................\n")
        # myfile.close()

        createdataset(n, 0)

        os.system(GDNTpath + "\n ./main.o")

        os.system(SSMpath + "\n ./main.o")


        # myfile = open("/Users/maaz/Desktop/report.txt", 'a')
        # myfile.write("######################################\n")
        # myfile.close()

        n+=10

report()
#createdataset(500,0)
