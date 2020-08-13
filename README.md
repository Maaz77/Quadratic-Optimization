# Quadratic-Optimization

**_Here are some codes to compare the time performance of two distinct methods for solving a quadratic optimization problem with norm constraint._**

## min xAx + b*x ,   s.b to |x| = 1 , A is N * N matrix , b and x are vectors  
#### 1. SSM method: 
The proposed solution is iterative and based on the trust region method. For further information check the following publications: 

a) "GLOBAL CONVERGENCE OF SSM FOR MINIMIZING A QUADRATIC OVER A SPHERE"

b) " MINIMIZING A QUADRATIC OVER A SPHERE∗ WILLIAM W. HAGER†"

related source code is available in the following link: http://users.clas.ufl.edu/hager/


#### 2. Gradient method :

The method finds some candidate solutions for the problem, thereafter selects the global solution among them. This method is implemented in both C language and python.


 
 
