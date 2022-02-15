#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:16:43 2021

@author: Tamirat Temesgen Dufera
"""

"""
Experiment for comparing the performance of the network as a fucntion of numbers of neuron.
The code is for system of differential equations with two unknowns. 
"""

# use the outgrad backage for the coputations of gradient
import functions 
import autograd.numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
   
np.random.seed(0)                      # to obtain a consistent solution

#define the right side function of the system of differential equations
def right_side(x, y):
    f1 = np.cos(x)+(y[0])**2+y[1]-(1+x**2+np.sin(x)*np.sin(x))
    f2 = 2*x-(1+x*x)*np.sin(x) + y[0]*y[1]
    return np.array([f1,f2])

# the initial values 
a1 = 0
a2=  1.0

# the analytical solution is given by 
def analytic(x):
    an_sol = np.zeros((2, len(x)))
    an_sol[0] = np.sin(x)
    an_sol[1] = 1+x*x  
    return an_sol     
              

#====================Experiment 1: numbers of neorons========================================
"""
Experiment for comparing the performance of the network as a fucntion of numbers of neuron.
The code is for system of differential equations with two unknowns. 
"""
# Input Data
a , b= [0.0, 1.0]               # the domain        
m = 11                          # number of sample points between 0 and 1.

x = np.linspace(a,b, m)         # generate m sample points

X = np.zeros((1,m))             # form a matrix containing the sample points
X[0] = x

# Define the neural network sturctue, loop over different sizes
H = [4, 17, 60, 150, 200]            # different sizes for the hidden layer
ITs = 100                            # number of iterations 
tol = 1e-04                          # the tollerance

fig1 = plt.figure()
for h in  H:      
    # The training, measuring the time for the convergence corresponding to the size of neurons.
    tic = time.time()
    param1, param2, Cost, Iter = functions.ode_nn_model1(X, h ,a1, a2, a, right_side,  tol, ITs, iter = 0, print_cost = True)
    toc = time.time()
    print("time : " + str((toc-tic))+ "s")
    
    #plot    
    plt.plot(Iter, Cost, label='$ n_{h}$ = ' + str(h)+ ', cost  : ' + str(round(Cost[-1],4)) + ', time = ' + str(round((toc-tic),2)) + ' s')
    
    
plt.legend( )
plt.title("Convergence of cost function for different number of neurons" )
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.savefig('fig2.eps') 
plt.show()





    
    
