import matplotlib.pyplot as plt
import numpy as np 
import math


dataset = np.loadtxt('/Users/maaz/Desktop/diagramreport.txt')

dataset = dataset.reshape(1,156)

Gradientdim = []
SSMdim = []
Gradientdata =  []
SSMdata = []


i = 0
h = 0
p = 1
z = 2

while(i < len(dataset[0])):
    
    if (i == h):
        h+=3
        Gradientdim.append(dataset[0][i])
        SSMdim.append(dataset[0][i]- 2)
    elif(i == p):
        p+=3
        Gradientdata.append(math.log(dataset[0][i],10))
    elif(i == z):
        z+=3
        SSMdata.append(math.log(dataset[0][i],10))
    i+=1

        

  

plt.bar(Gradientdim,Gradientdata,width = 2,label="Gradient method", color='r')
plt.bar(SSMdim,SSMdata, width = 2 ,label="SSM method", color='g')


plt.legend()
plt.xlabel('Dimension')
plt.ylabel('log(time)')

plt.title('Compare time performance of SSM method and Gradient method')

plt.show()