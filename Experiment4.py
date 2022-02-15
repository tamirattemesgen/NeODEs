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
import pandas as pd
import subprocess
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
              


#=============Experiment 4: Grid effect RK4 vs ANN================================
"""
ANN verses RK4 -experiment on the size of grid points
"""
# Descretize the domain             
t0, tend = [0.0, 4.0]     

# diffent size of grid points between a and b.
M = [11, 16, 21, 26]

# store the absulte errors due to ANN and RKR 
ANN_er = []
RK4_er = []

fig5 = plt.figure(figsize= (10,6))
for m in range(len(M)):
    
    t = np.linspace(t0,tend, M[m])         # generate the  points
    x = (t-min(t))/(max(t)-min(t))         # normalize data
    X = np.zeros((1,M[m]))             
    X[0] = x
    T = np.zeros((1,M[m]))             
    T[0]= t
    h = 60
    ITs1 = 1500
    tol1 = 1e-06
    
    param1, param2 = functions.ode_nn_model(X, h, a1, a2, a, right_side, tol1, ITs1, iter = 0, print_cost = True)
    
    # Using the learned parameters we compute the ANN soutions
    def Nt1(X, param1):
        W11 = param1["W11"]
        b11 = param1["b11"]
        W12 = param1["W12"]
        b12 = param1["b12"] 
        Z1 = np.dot(W11,X) + b11
        A1 = functions.sigmoid(Z1)
        Z2 = np.dot(W12,A1)+ b12
        A2 = Z2
        return np.array(a1 + (X-t0)*A2[0])       # the first component
        
    def Nt2(X, param2):
        W21 = param2["W21"]
        b21 = param2["b21"]
        W22 = param2["W22"]
        b22 = param2["b22"] 
        Z1 = np.dot(W21,X) + b21
        A1 = functions.sigmoid(Z1)
        Z2 = np.dot(W22,A1)+ b22
        A2 = Z2
        return np.array(a2 + (X-t0)*A2[1])          # the second component
    
    # ANN solution
    y1 = Nt1(T, param1)
    y2 = Nt2(T, param2)
    
    
    # For comparision compute the exact solution
    exact = analytic(t)
    
    # RK4 solution
    u = np.array([0.0, 1.0])
    time, RK = functions.RungeKutta4(right_side, t0, tend, u, M[m])

    # We plot the trajectories    
    plt.subplot(2, 2, m+1)
    
    plt.plot(t, y1[0], 'g-o')
    plt.plot(t, y2[0], '-o')
    plt.plot(t, exact[0],'blue', lw=2)
    plt.plot(t, exact[1],'blue', lw=2)
    plt.plot(time, RK[:, 0],'r', lw=2)
    plt.plot(time, RK[:, 1],'-r',lw=2)  
    plt.title(str(M[m]) + " grid points")
    plt.legend(['ANN $y_1$','ANN $y_2$', 'Exact $y_1$','Exact $y_2$', 'RK4 $y_1$', 'RK4 $y_2$'])

    # the error at the end point
    ANN_er.append(round(abs(y2[0]-exact[1])[-1], 3))
    RK4_er.append(round(abs(exact[1]- RK[:,1])[-1],3))
    
# save the plot in folder Figures
plt.savefig('../results/fig5.eps')
plt.show()
 
# for a data frame
data = {'ANN_error': ANN_er, 'RK4_error': RK4_er }
summary_table = pd.DataFrame(data= data , index = M )

# the following code save the table as latex and pdf
filename = '../results/out1.tex'
pdffile =  '../results/out1.pdf', 
    
template = r'''\documentclass{{standalone}}
   \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    '''
    
with open(filename, 'w') as f:
    f.write(template.format(summary_table.to_latex()))
    
subprocess.call(['pdflatex', filename])

print(summary_table)



    
    
