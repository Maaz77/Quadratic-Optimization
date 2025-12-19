# Quadratic-Optimization

This repository contains code to compare the **runtime performance** of two methods for solving a **quadratic optimization problem on the unit sphere**. 

## Problem Formulation

We consider the constrained optimization problem:

$$
\min_{x \in \mathbb{R}^N} \; x^\top A x + b^\top x
\quad \text{s.t.} \quad \|x\|_2 = 1 .
$$

where:
- $A \in \mathbb{R}^{N \times N}$,
- $b \in \mathbb{R}^N$,
- $x \in \mathbb{R}^N$.
## Methods Implemented

### 1) SSM Method (Trust-Region-Based)

An iterative approach based on the trust-region method. For details, see:

- **“Global Convergence of SSM for Minimizing a Quadratic over a Sphere”**
- **William W. Hager, “Minimizing a Quadratic over a Sphere”**

Related reference implementation and materials are available here:  
http://users.clas.ufl.edu/hager/

### 2) Gradient-Based Method (Gradient + Eigenvalues)

A method that leverages gradient information and eigenvalue computations. Implemented in:
- **C**
- **Python**
 
