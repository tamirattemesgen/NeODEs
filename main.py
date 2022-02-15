#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 11:16:43 2021

@author: tamirat
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


#================Experiment 2: one vs two hidden layers=============================
"""
Experiment for comparing one hidden layer vs two hidden layers.
The code is for system of differential equations with two uknowns.
"""
# Input Data
a , b= [0.0, 1.0]               # the domain        
m = 11                          # number of sample points between a and b.

x = np.linspace(a,b, m)         # generate m sample points

X = np.zeros((1,m))             # form a matrix containing the sample points
X[0] = x

H = [4, 60]                     # different sizes for the hidden layer1

ITs = 100                       # number of iterations 
tol = 1e-04                     # the tollerance
# For one hidden layer
fig2 = plt.figure()
for h in  H:            
    # The training, measuring the time for the convergence corresponding to the size of neurons.
    tic = time.time()
    param1, param2, Cost, Iter = functions.ode_nn_model1(X, h, a1, a2, a, right_side, tol, ITs, iter = 0, print_cost = True)
    toc = time.time()
    print("time : " + str((toc-tic))+ "s")
    
    #plot
    plt.plot(Iter, Cost, label='$ n_{h}$ =' + str(h)+ ', cost  = ' + str(round(Cost[-1], 4)) + ', time = ' + str(round((toc-tic),2)) + ' s')
# For two hidden layers

H1 = [4, 60]
H2 = [4, 60]

for h1 in range(0, len(H1)):
    for h2 in range(0, len(H2)):             
        # The training, measuring the time for the convergence corresponding to the size of neurons.    
        tic = time.time()
        param1, param2, Cost, Iter = functions.ode_nn_model2(X, H1[h1], H2[h2],  a1, a2, a, right_side, tol, ITs, iter = 0, print_cost = True)
        toc = time.time()
        print("time of computation: " + str((toc-tic))+ "s")            
    
        #plots the out come at the end of iteration, with error and time inforamtion
        plt.plot(Iter, Cost, label='$[n_{h1},n_{h2}]$ ='+str([H1[h1],H2[h2]]) + ' , cost  = ' + str(round(Cost[-1],3)) + ' , time = ' + str(round((toc-tic),2)) + ' s')

plt.legend()
plt.title("One vs Two Hidden Layer" )
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.savefig('fig3.eps')
plt.show()

     
#=============Experiment 3: Implementation and comparision with analytical solution=====================
"""
This is the code for implementation of the ANN.
A system of differential equations with two unkown. Compared with analytical solution.
"""

# Input Data
a, b =  [0.0, 1.0]               # the domain                       
m =11                            # number of sample points between a and b
t = np.linspace(a,b, m)          # generate the  points
x = (t-min(t))/(max(t)-min(t))   # normalize the data
X = np.zeros((1,m))              # we will generalize the model for system
X[0] = x
T = np.zeros((1,m))
T[0]= t

# Define the neural network sturctue 
h = 60   
ITs = 30000    # number of iteration
tol = 1e-06    # tollerance
ANN_er = []    # for storing neural network solution
RK4_er = []    # for storing Runge-Kutta solution

param1, param2, Cost, Iter = functions.ode_nn_model1(X, h, a1, a2, a, right_side, tol, ITs, iter = 0, print_cost = True)
    
# Using the learned parameters we compute the ANN soutions
def Yt1(X, param1):
    # retrieve each parameters 
    W11 = param1["W11"]
    b11 = param1["b11"]
    W12 = param1["W12"]
    b12 = param1["b12"] 
    Z1 = np.dot(W11,X) + b11
    A1 = functions.sigmoid(Z1)
    Z2 = np.dot(W12,A1)+ b12
    A2 = Z2
    return np.array(a1 + (X-a)*A2[0])      # the first component

def Yt2(X, param2):
    # retrive parameters
    W21 = param2["W21"]
    b21 = param2["b21"]
    W22 = param2["W22"]
    b22 = param2["b22"] 
    Z1 = np.dot(W21,X) + b21
    A1 = functions.sigmoid(Z1)
    Z2 = np.dot(W22,A1)+ b22
    A2 = Z2
    return np.array(a2 + (X-a)*A2[1])          # the second component

# compute the ANN solution
y1 = Yt1(T, param1)
y2 = Yt2(T, param2)
    
    
# For comparision compute the exact solution
exact = analytic(t)


# We plot the trajectories
fig3=plt.figure(figsize= (10,8))
plt.subplot(2, 1, 1)    
plt.plot(t, y1[0], 'g--o')
plt.plot(t, y2[0], 'r--o')
plt.plot(t, exact[0],'blue', lw=2)
plt.plot(t, exact[1],'blue', lw=2)
plt.legend(['ANN $y_1$','ANN $y_2$', 'Exact $y_1$','Exact $y_2$'])

# absulte error   
err_1= abs(y1[0]-exact[0]) 
err_2= abs(y2[0]-exact[1])
plt.subplot(2,1,2)
plt.plot(t, err_1,'-go')
plt.plot(t, err_2,'-r*')
plt.legend(['error $y_1$', 'error $y_2$'])
# save the plot
plt.savefig('fig4.eps')
plt.show()


# create table for soutions and corresponding error    
data = {'ANN $y_1$': y1[0], 'Analytic $y_1$':exact[0],  'ANN $y_2$': y2[0],'Analyitc $y_2$': exact[1]}
data_err = {'error $y_1$': err_1, 'error $y_2$': err_2 }

# for a data frame
summary_table = pd.DataFrame(data= data , index = t)
summary_err = pd.DataFrame(data = data_err, index = t)

# the following code save the table as latex and pdf
filename = ['out_ex1.tex','err.tex']
pdffile =  ['out_ex1.pdf', 'err.pdf']
    
template = r'''\documentclass{{standalone}}
   \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    '''
    
with open(filename[0], 'w') as f:
    f.write(template.format(summary_table.to_latex()))
    
subprocess.run(['pdflatex', filename[0]])

with open(filename[1], 'w') as f:
    f.write(template.format(summary_err.to_latex()))
    
subprocess.run(['pdflatex', filename[1]])


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
plt.savefig('fig5.eps')
plt.show()
 
# for a data frame
data = {'ANN_error': ANN_er, 'RK4_error': RK4_er }
summary_table = pd.DataFrame(data= data , index = M )

# the following code save the table as latex and pdf
filename = 'out1.tex'
pdffile =  'out1.pdf', 
    
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



    
    