#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:28:36 2021

@author: tamirat
This code contains function used for simulation: activation function, parameters initialization for different networks, 
forward propagations, update rule using Adam method, model for the two architect and finite difference Runge-kuta method.
"""
# use the outgrad backage for the coputations of gradient
import autograd.numpy as np
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')
   
np.random.seed(0)                      # to obtain a consistent solution



# activation function
def sigmoid(x):
    return np.tanh(x)

#Initialize the model's parameters, this is for network of single hidden layer    
def initialize_P(n_x, n_h, n_y):
    """
    Argument:
    n_x -- number of input in the training
    n_h -- number units in the hidden layer
    n_y -- number of output, which will be the number of dependent variable in the system.
    
    Returns:
    P1 --   Python dictionary containing parameters:
                    W11 -- weight matrix of shape (n_h, n_x)
                    b11 -- bias vector of shape (n_h, 1)
                    W12 -- weight matrix of shape (n_y, n_h)
                    b12 -- bias vector of shape (n_y, 1)
    P2 --   Python dictionary containing parameters:
                    W21 -- weight matrix of shape (n_h, n_x)
                    b21 -- bias vector of shape (n_h, 1)
                    W22 -- weight matrix of shape (n_y, n_h)
                    b22 -- bias vector of shape (n_y, 1)
    V1,V2,M1,M2- the same shape as P1 and P2
    """    
    W11 = np.random.randn(n_h, n_x)*0.01
    b11 = np.zeros((n_h,1))
    W12 = np.random.randn(n_y,n_h)*0.01
    b12 = np.zeros((n_y,1))
    
    P1 = {"W11": W11,
          "W12": W12,
          "b11": b11,
          "b12": b12}
    
    W21 = np.random.randn(n_h, n_x)*0.01
    b21 = np.zeros((n_h,1))
    W22 = np.random.randn(n_y,n_h)*0.01
    b22 = np.zeros((n_y,1))
    
    P2 = {"W21": W21,
          "W22": W22,
          "b21": b21,
          "b22": b22}        
            
    v_W11 = np.zeros((n_h, n_x))
    v_b11 = np.zeros((n_h,1))
    v_W12 = np.zeros((n_y,n_h))
    v_b12 = np.zeros((n_y,1))
    
    V1= {"v_W11": v_W11,
        "v_W12": v_W12,
        "v_b11": v_b11,
        "v_b12": v_b12}      
    
    
    v_W21 = np.zeros((n_h, n_x))
    v_b21 = np.zeros((n_h,1))
    v_W22 = np.zeros((n_y,n_h))
    v_b22 = np.zeros((n_y,1))
    
    V2= {"v_W21": v_W21,
        "v_W22": v_W22,
        "v_b21": v_b21,
        "v_b22": v_b22}
    
    m_W11 = np.zeros((n_h, n_x))
    m_b11 = np.zeros((n_h,1))
    m_W12 = np.zeros((n_y,n_h))
    m_b12 = np.zeros((n_y,1))
    
    M1= {"m_W11": m_W11,
        "m_W12": m_W12,
        "m_b11": m_b11,
        "m_b12": m_b12}        
    
    m_W21 = np.zeros((n_h, n_x))
    m_b21 = np.zeros((n_h,1))
    m_W22 = np.zeros((n_y,n_h))
    m_b22 = np.zeros((n_y,1))         
    
    M2= {"m_W21": m_W21,
        "m_W22": m_W22,
        "m_b21": m_b21,
        "m_b22": m_b22}

    
    return P1, P2, V1, V2, M1, M2

# Initialize_parameters this is for network of two hidden layers
def initialize_P2(n_x, n_h1, n_h2, n_y):
    """
    Argument:
    n_x -- number of input in the training
    n_h1 -- number of hidden units in the first hidden layer
    n_h2 -- number of hidden units in the second hidden layer
    n_y -- number of output, which will be the number of dependent variable in the system.
    
    Returns:
    P1 --   Python dictionary containing your parameters:
                    W11 -- weight matrix of shape (n_h1, n_x)
                    b11 -- bias vector of shape (n_h1, 1)
                    W12 -- weight matrix of shape (n_h2, n_h1)
                    b12 -- bias vector of shape (n_h2, 1)
                    W13 -- weight matrix of shape (n_y, n_h2)
                    b13 -- bias vector of shape (n_y, 1)
                    
    P2 --   Python dictionary containing your parameters:
                    W21 -- weight matrix of shape (n_h1, n_x)
                    b21 -- bias vector of shape (n_h1, 1)
                    W22 -- weight matrix of shape (n_h2, n_h1)
                    b22 -- bias vector of shape (n_h2, 1)
                    W23 -- weight matrix of shape (n_y, n_h2)
                    b23 -- bias vector of shape (n_y, 1)
    V1,V2,M1,M2- the same shape as P1 and P2
    """    
    W11 = np.random.randn(n_h1, n_x)*0.01
    b11 = np.zeros((n_h1, 1))
    W12 = np.random.randn(n_h2, n_h1)*0.01
    b12 = np.zeros((n_h2, 1))
    W13 = np.random.randn(n_y, n_h2)*0.01
    b13 = np.zeros((n_y, 1))
    
                
    P1 = {"W11":W11,
          "W12": W12,
          "b11": b11,
          "b12": b12,
          "W13": W13,
          "b13": b13}
    
    W21 = np.random.randn(n_h1, n_x)*0.01
    b21 = np.zeros((n_h1, 1))
    W22 = np.random.randn(n_h2, n_h1)*0.01
    b22 = np.zeros((n_h2, 1))
    W23 = np.random.randn(n_y, n_h2)*0.01
    b23 = np.zeros((n_y, 1))
    
    P2 = {"W21": W21,
          "W22": W22,
          "b21": b21,
          "b22": b22,
          "W23": W23,
          "b23": b23}
    
                
    v_W11 = np.zeros((n_h1, n_x))
    v_b11 = np.zeros((n_h1,1))
    v_W12 = np.zeros((n_h2,n_h1))
    v_b12 = np.zeros((n_h2,1))
    v_W13 = np.zeros((n_y,n_h2))
    v_b13 = np.zeros((n_y,1))
    
    V1 = {"v_W11": v_W11,
         "v_W12": v_W12,
         "v_b11": v_b11,
         "v_b12": v_b12,
         "v_W13": v_W13,
         "v_b13": v_b13}
    
    v_W21 = np.zeros((n_h1, n_x))
    v_b21 = np.zeros((n_h1,1))
    v_W22 = np.zeros((n_h2,n_h1))
    v_b22 = np.zeros((n_h2,1))
    v_W23 = np.zeros((n_y,n_h2))
    v_b23 = np.zeros((n_y,1))            
    
    V2= {"v_W21": v_W21,
        "v_W22": v_W22,
        "v_b21": v_b21,
        "v_b22": v_b22,
        "v_W23": v_W23,
        "v_b23": v_b23}
    
    
    m_W11 = np.zeros((n_h1, n_x))
    m_b11 = np.zeros((n_h1,1))
    m_W12 = np.zeros((n_h2,n_h1))
    m_b12 = np.zeros((n_h2,1))
    m_W13 = np.zeros((n_y,n_h2))
    m_b13 = np.zeros((n_y,1))
    
    M1= {"m_W11": m_W11,
        "m_W12": m_W12,
        "m_b11": m_b11,
        "m_b12": m_b12,
        "m_W13": m_W13,
        "m_b13": m_b13}
    
    m_W21 = np.zeros((n_h1, n_x))
    m_b21 = np.zeros((n_h1,1))
    m_W22 = np.zeros((n_h2,n_h1))
    m_b22 = np.zeros((n_h2,1))
    m_W23 = np.zeros((n_y,n_h2))
    m_b23 = np.zeros((n_y,1))
                
    M2= {"m_W21": m_W21,
        "m_W22": m_W22,
        "m_b21": m_b21,
        "m_b22": m_b22,
        "m_W23": m_W23,
        "m_b23": m_b23}           
    
    return P1, P2, V1, V2, M1, M2

# Forward propagation for single hidden layer neural network
def feed_forward(X, P1, P2, a1, a2, a, right_side):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing parameters (output of initialization function)
        a1, a2 : initial values,
        a  : initial point of the domain [a, b]   
        right_side: the right side function
        Returns:
        cost, grads
        """    
       # define the trial solutions 
        def Yt1(X, P1):
            W11 = P1["W11"]
            b11 = P1["b11"]
            W12 = P1["W12"]
            b12 = P1["b12"] 
            Z1 = np.dot(W11,X) + b11
            A1 = sigmoid(Z1)
            Z2 = np.dot(W12,A1)+ b12
            A2 = Z2                                  
            return a1 + (X-a)*(A2)[0]             # the first component
        def Yt2(X, P2):
            W21 = P2["W21"]
            b21 = P2["b21"]
            W22 = P2["W22"]
            b22 = P2["b22"] 
            Z1 = np.dot(W21,X) + b21
            A1 = sigmoid(Z1)
            Z2 = np.dot(W22,A1)+ b22
            A2 = Z2                                    
            return a2 + (X-a)*(A2)[1]              # the second component
        
        # define the loss fucntion
        
        def loss(P1,P2):
            # derivative of each components 
            dy1t = egrad(Yt1, argnum=0)(X, P1)
            dy2t = egrad(Yt2, argnum=0)(X, P2)
            
            cost1 = dy1t - right_side(X, [Yt1(X,P1), Yt2(X,P2)])[0]
            cost2 = dy2t - right_side(X, [Yt1(X,P1), Yt2(X,P2)])[1]
            cost = 0.5*np.sum(cost1**2+cost2**2) # total cost
            return cost
        
        grad1 = egrad(loss,argnum=0)        # the gradient of the cost w.r.t the parameters
        grad2 = egrad(loss,argnum=1)
        
        return loss, grad1, grad2
# Forward propagation: for two hidden layers
def feed_forward2(X, P1, P2, a1, a2, a, right_side):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing parameters     
    a1, a2, -- initial values        
    a -- the firs point(t0)
    right_side: the right side function corresponding the system of ode
    Returns: cost, grads
    """    
   # calculation the trial solutions 
    def Yt1(X, P1):
        W11 = P1["W11"]
        b11 = P1["b11"]
        W12 = P1["W12"]
        b12 = P1["b12"]
        W13 = P1["W13"]
        b13 = P1["b13"]        
        Z1 = np.dot(W11,X) + b11
        A1 = sigmoid(Z1)
        Z2 = np.dot(W12,A1)+ b12
        A2 = sigmoid(Z2)
        Z3 = np.dot(W13,A2)+ b13
        A3 = Z3                                  
        return a1 + (X-a)*(A3)[0]             # the first component
    def Yt2(X, P2):
        W21 = P2["W21"]
        b21 = P2["b21"]
        W22 = P2["W22"]
        b22 = P2["b22"]
        W23 = P2["W23"]
        b23 = P2["b23"]        
        Z1 = np.dot(W21,X) + b21
        A1 = sigmoid(Z1)
        Z2 = np.dot(W22,A1)+ b22
        A2 = sigmoid(Z2)
        Z3 = np.dot(W23,A2) + b23
        A3 = Z3                                    
        return a2 + (X-a)*(A3)[1]              # the second component
    
    # define the loss fucntion            
    def loss(P1,P2):
        # derivative of each components 
        dy1t = egrad(Yt1, argnum=0)(X, P1)
        dy2t = egrad(Yt2, argnum=0)(X, P2)
        
        cost1 = dy1t - right_side(X, [Yt1(X,P1), Yt2(X,P2)])[0]
        cost2 = dy2t - right_side(X, [Yt1(X,P1), Yt2(X,P2)])[1]
        cost = 0.5*np.sum(cost1**2+cost2**2) # total cost
        return cost
    
    grad1 = egrad(loss,argnum=0)        # the gradient of the cost w.r.t the parameters
    grad2 = egrad(loss,argnum=1)
    
    return loss, grad1, grad2

    
# Updating parameter using Adam algorithm, for the single hidden layers    
def update_Adam(P1, P2, V1, V2, M1, M2, grad1, grad2, eta = 0.01, beta1 = 0.9, beta2 =0.999, l=1, eps = 1e-8):
    """
    Updates parameters using the Adam update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads      -- python dictionary containing gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    v_W11 = V1["v_W11"]
    v_W12 = V1["v_W12"]
    v_b11 = V1["v_b11"]
    v_b12 = V1["v_b12"]
    
    v_W21 = V2["v_W21"]
    v_W22 = V2["v_W22"]
    v_b21 = V2["v_b21"]
    v_b22 = V2["v_b22"]
    
    m_W11 = M1["m_W11"]
    m_W12 = M1["m_W12"]
    m_b11 = M1["m_b11"]
    m_b12 = M1["m_b12"]
    
    m_W21 = M2["m_W21"]
    m_W22 = M2["m_W22"]
    m_b21 = M2["m_b21"]
    m_b22 = M2["m_b22"]
    
    # Retrieve each parameter from the dictionary "P1"
    W11 = P1["W11"]
    b11 = P1["b11"]
    W12 = P1["W12"]
    b12 = P1["b12"]
    
    # Retrieve each gradient from the dictionary "grad1"
    dW11 = grad1["W11"]
    db11 = grad1["b11"]
    dW12 = grad1["W12"]
    db12 = grad1["b12"]    
    
    m_W11 = beta1*m_W11+(1-beta1)*dW11
    m_W12 = beta1*m_W12+(1-beta1)*dW12
    m_b11 = beta1*m_b11+(1-beta1)*db11
    m_b12 = beta1*m_b12+(1-beta1)*db12   
        
    v_W11 = beta2*v_W11+ (1-beta2)*dW11**2
    v_W12 = beta2*v_W12+(1-beta2)*dW12**2
    v_b11 = beta2*v_b11+(1-beta2)*db11**2
    v_b12 = beta2*v_b12+(1-beta2)*db12**2
        
    # Update rule for each parameter
    W11 = W11-(eta/np.sqrt((v_W11/(1.0-beta2**l))+eps))*(m_W11/(1-beta1**l))
    b11 = b11-(eta/np.sqrt((v_b11/(1.0-beta2**l))+eps))*(m_b11/(1-beta1**l))
    W12 = W12-(eta/np.sqrt((v_W12/(1.0-beta2**l))+eps))*(m_W12/(1-beta1**l))
    b12 = b12-(eta/np.sqrt((v_b12/(1.0-beta2**l))+eps))*(m_b12/(1-beta1**l))
    
    P1 = {"W11": W11,                  
          "b11": b11,
          "W12": W12,
          "b12": b12}
    
    # Retrieve each parameter from the dictionary "P2"
    W21 = P2["W21"]
    b21 = P2["b21"]
    W22 = P2["W22"]
    b22 = P2["b22"]
    
    # Retrieve each gradient from the dictionary "grad2"
    dW21 = grad2["W21"]
    db21 = grad2["b21"]
    dW22 = grad2["W22"]
    db22 = grad2["b22"]
    
    m_W21 = beta1*m_W21+ (1-beta1)*dW21
    m_W22 = beta1*m_W22+(1-beta1)*dW22
    m_b21 = beta1*m_b21+(1-beta1)*db21
    m_b22 = beta1*m_b22+(1-beta1)*db22
    
    v_W21 = beta2*v_W21+ (1-beta2)*dW21**2
    v_W22 = beta2*v_W22+(1-beta2)*dW22**2
    v_b21 = beta2*v_b21+(1-beta2)*db21**2
    v_b22 = beta2*v_b22+(1-beta2)*db22**2    
       
    # Update rule for each parameter
    W21 = W21-(eta/np.sqrt(v_W21/(1-beta2**l)+eps))*(m_W21/(1-beta1**l))
    b21 = b21-(eta/np.sqrt((v_b21/(1-beta2**l))+eps))*(m_b21/(1-beta1**l))
    W22 = W22-(eta/np.sqrt((v_W22/(1-beta2**l))+eps))*(m_W22/(1-beta1**l))
    b22 = b22-(eta/np.sqrt((v_b22/(1-beta2**l))+eps))*(m_b22/(1-beta1**l))
    
    l = l +1
    
    P2 = {"W21": W21,                  
          "b21": b21,
          "W22": W22,
          "b22": b22}
    
    
    V1= {"v_W11": v_W11,
        "v_W12": v_W12,
        "v_b11": v_b11,
        "v_b12": v_b12}
    
    V2= {"v_W21": v_W21,
        "v_W22": v_W22,
        "v_b21": v_b21,
        "v_b22": v_b22}
    
    M1= {"m_W11": m_W11,
        "m_W12": m_W12,
        "m_b11": m_b11,
        "m_b12": m_b12}
    
    M2= {"m_W21": m_W21,
        "m_W22": m_W22,
        "m_b21": m_b21,
        "m_b22": m_b22}

    
    return P1, P2, V1, V2, M1, M2, l

# Update parameter using Adam algorithm, for two hidden layers
def update2_Adam(P1, P2, V1, V2,M1, M2, grad1, grad2, eta = 0.01, beta1 = 0.9, beta2 =0.999, l=1, eps = 1e-8):
    """
    Updates parameters using the Adam update rule            
    Arguments:
    parameters      -- python dictionary containing parameters 
    grad1 and grad2 -- python dictionary containing gradients 
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    v_W11 = V1["v_W11"]
    v_W12 = V1["v_W12"]
    v_b11 = V1["v_b11"]
    v_b12 = V1["v_b12"]
    v_W13 = V1["v_W13"]
    v_b13 = V1["v_b13"]
    
    v_W21 = V2["v_W21"]
    v_W22 = V2["v_W22"]
    v_b21 = V2["v_b21"]
    v_b22 = V2["v_b22"]
    v_W23 = V2["v_W23"]
    v_b23 = V2["v_b23"]
    
    m_W11 = M1["m_W11"]
    m_W12 = M1["m_W12"]
    m_b11 = M1["m_b11"]
    m_b12 = M1["m_b12"]
    m_W13 = M1["m_W13"]
    m_b13 = M1["m_b13"]
    
    m_W21 = M2["m_W21"]
    m_W22 = M2["m_W22"]
    m_b21 = M2["m_b21"]
    m_b22 = M2["m_b22"]
    m_W23 = M2["m_W23"]
    m_b23 = M2["m_b23"]
    
    # Retrieve each parameter from the dictionary "P1"
    W11 = P1["W11"]
    b11 = P1["b11"]
    W12 = P1["W12"]
    b12 = P1["b12"]
    W13 = P1["W13"]
    b13 = P1["b13"]  
    
    # Retrieve each gradient from the dictionary "grad1"
    dW11 = grad1["W11"]
    db11 = grad1["b11"]
    dW12 = grad1["W12"]
    db12 = grad1["b12"] 
    dW13 = grad1["W13"]
    db13 = grad1["b13"]  
    
    m_W11 = beta1*m_W11+(1-beta1)*dW11
    m_W12 = beta1*m_W12+(1-beta1)*dW12
    m_b11 = beta1*m_b11+(1-beta1)*db11
    m_b12 = beta1*m_b12+(1-beta1)*db12  
    m_W13 = beta1*m_W13+(1-beta1)*dW13
    m_b13 = beta1*m_b13+(1-beta1)*db13
        
    v_W11 = beta2*v_W11+ (1-beta2)*dW11**2
    v_W12 = beta2*v_W12+(1-beta2)*dW12**2
    v_b11 = beta2*v_b11+(1-beta2)*db11**2
    v_b12 = beta2*v_b12+(1-beta2)*db12**2
    v_W13 = beta2*v_W13+(1-beta2)*dW13**2
    v_b13 = beta2*v_b13+(1-beta2)*db13**2
        
    # Update rule for each parameter
    W11 = W11-(eta/np.sqrt((v_W11/(1.0-beta2**l))+eps))*(m_W11/(1-beta1**l))
    b11 = b11-(eta/np.sqrt((v_b11/(1.0-beta2**l))+eps))*(m_b11/(1-beta1**l))
    W12 = W12-(eta/np.sqrt((v_W12/(1.0-beta2**l))+eps))*(m_W12/(1-beta1**l))
    b12 = b12-(eta/np.sqrt((v_b12/(1.0-beta2**l))+eps))*(m_b12/(1-beta1**l))
    W13 = W13-(eta/np.sqrt((v_W13/(1.0-beta2**l))+eps))*(m_W13/(1-beta1**l))
    b13 = b13-(eta/np.sqrt((v_b13/(1.0-beta2**l))+eps))*(m_b13/(1-beta1**l))
    
    P1 = {"W11":W11,
          "W12": W12,
          "b11": b11,
          "b12": b12,
          "W13": W13,
          "b13": b13}
    
    # Retrieve each parameter from the dictionary "P2"
    W21 = P2["W21"]
    b21 = P2["b21"]
    W22 = P2["W22"]
    b22 = P2["b22"]
    W23 = P2["W23"]
    b23 = P2["b23"] 
    
    # Retrieve each gradient from the dictionary "grad2"
    dW21 = grad2["W21"]
    db21 = grad2["b21"]
    dW22 = grad2["W22"]
    db22 = grad2["b22"]
    dW23 = grad2["W23"]
    db23 = grad2["b23"]
    
    m_W21 = beta1*m_W21+ (1-beta1)*dW21
    m_W22 = beta1*m_W22+(1-beta1)*dW22
    m_b21 = beta1*m_b21+(1-beta1)*db21
    m_b22 = beta1*m_b22+(1-beta1)*db22
    m_W23 = beta1*m_W23+(1-beta1)*dW23
    m_b23 = beta1*m_b23+(1-beta1)*db23
    
    v_W21 = beta2*v_W21+ (1-beta2)*dW21**2
    v_W22 = beta2*v_W22+(1-beta2)*dW22**2
    v_b21 = beta2*v_b21+(1-beta2)*db21**2
    v_b22 = beta2*v_b22+(1-beta2)*db22**2  
    v_W23 = beta2*v_W23+(1-beta2)*dW23**2
    v_b23 = beta2*v_b23+(1-beta2)*db23**2
       
    # Update rule for each parameter
    W21 = W21-(eta/np.sqrt(v_W21/(1-beta2**l)+eps))*(m_W21/(1-beta1**l))
    b21 = b21-(eta/np.sqrt((v_b21/(1-beta2**l))+eps))*(m_b21/(1-beta1**l))
    W22 = W22-(eta/np.sqrt((v_W22/(1-beta2**l))+eps))*(m_W22/(1-beta1**l))
    b22 = b22-(eta/np.sqrt((v_b22/(1-beta2**l))+eps))*(m_b22/(1-beta1**l))
    W23 = W23-(eta/np.sqrt((v_W23/(1-beta2**l))+eps))*(m_W23/(1-beta1**l))
    b23 = b23-(eta/np.sqrt((v_b23/(1-beta2**l))+eps))*(m_b23/(1-beta1**l))
    
    l = l +1
    
    P2 = {"W21": W21,
          "W22": W22,
          "b21": b21,
          "b22": b22,
          "W23": W23,
          "b23": b23}
        
    V1= {"v_W11": v_W11,
         "v_W12": v_W12,
         "v_b11": v_b11,
         "v_b12": v_b12,
         "v_W13": v_W13,
         "v_b13": v_b13}

    V2= {"v_W21": v_W21,
         "v_W22": v_W22,
         "v_b21": v_b21,
         "v_b22": v_b22,
         "v_W23": v_W23,
         "v_b23": v_b23}
        
    M1= {"m_W11": m_W11,
         "m_W12": m_W12,
         "m_b11": m_b11,
         "m_b12": m_b12,
         "m_W13": m_W13,
         "m_b13": m_b13}
        
    M2= {"m_W21": m_W21,
         "m_W22": m_W22,
         "m_b21": m_b21,
         "m_b22": m_b22,
         "m_W23": m_W23,
         "m_b23": m_b23}

    
    return P1, P2, V1, V2, M1, M2, l    


# the model, for single hidden layers
def ode_nn_model1(X, h,  a1, a2, a, right_side,tol, ITs, iter = 0, print_cost = False):
    """
    Arguments:
    X   -- dataset of shape (1, m)
    h -- size of the hidden layer
    tol -- tollerance
    ITs, maximum iterations
    
    iter -- Number of iterations initialized to be 0.
    
    Returns:
    parameters -- parameters learned by the model.
    number of iteration and corresponding cost
    """
    n_x = X.shape[0]
    n_y =  2
    
    # Initialize parameters        
    P1, P2, V1, V2 , M1, M2 = initialize_P(n_x, h, n_y)
    
    # feed forward
    loss, grad1, grad2 = feed_forward(X, P1, P2, a1, a2, a, right_side)
    cost  = loss(P1, P2)
    
    # store cost and number of iteration for plotting the convergence
    Cost = []
    Iter = []

    # condition for stopping
    while cost>tol and iter < ITs:            
        # Feed forward propagation:        
        loss, grad1, grad2 = feed_forward(X, P1, P2, a1, a2, a, right_side)
        
        # Cost function. 
        cost  = loss(P1, P2)
        
        # gradients
        grad1 = grad1(P1, P2) 
        grad2 = grad2(P1, P2)
        
        # parameters update.          
        P1, P2, V1, V2, M1, M2, t  = update_Adam(P1, P2, V1, V2, M1, M2, grad1, grad2)
        iter += 1  
            
        # Print the cost every 10 iterations
        if print_cost and iter % 10 == 0:
            print ("Cost after iteration %i: %f" %(iter, cost))
        Cost.append(cost)
        Iter.append(iter)

    return P1, P2, np.array(Cost), Iter  

# Model for the two hidden layers
def ode_nn_model2(X, n_h1, n_h2, a1, a2, a, right_side, tol, ITs, iter = 0, print_cost = False):
    """
    Arguments:
    X   -- dataset of shape (1, m)
    n_h1, n_h2 -- size of the hidden layer one and two respectively
    iter -- Number of iterations, initialized to be zero
    tol -- tollerance
    ITs, maximum iteration
    Returns:
    parameters -- parameters learned by the model.
        """
    n_x = X.shape[0]
    n_y = 2
        
    # Initialize parameters            
    P1, P2, V1, V2, M1, M2 = initialize_P2(n_x, n_h1, n_h2, n_y)
    
    # feed forward
    loss, grad1, grad2 = feed_forward2(X, P1, P2, a1, a2, a, right_side)
    cost  = loss(P1, P2)
    
    # store cost and number of iteration for plotting the convergence
    Cost = []
    Iter = []

    # condition for stopping
    while cost>tol and iter < ITs:                            
        # Feed forward propagation:        
        loss, grad1, grad2 = feed_forward2(X, P1, P2, a1, a2, a, right_side)
          
        # Cost function. 
        cost  = loss(P1, P2)
           
        # gradients
        grad1 = grad1(P1, P2) 
        grad2 = grad2(P1, P2)
            
        # parameter update.            
        P1, P2, V1, V2, M1, M2, t= update2_Adam(P1, P2, V1, V2, M1, M2, grad1, grad2)
        iter += 1  
               
        # Print the cost every 10 iterations
        if print_cost and iter % 10 == 0:
            print ("Cost after iteration %i: %f" %(iter, cost))
        Cost.append(cost)
        Iter.append(iter)

    return P1, P2, np.array(Cost), Iter 
# the model
def ode_nn_model(X, h,  a1, a2, a, right_side, tol, ITs, iter = 0, print_cost = False):
    """
    Arguments:
    X   -- dataset of shape (1, m)
    n_h -- size of the hidden layer
    iter -- Number of iterations, initialized to be zero.
    
    Returns:
    parameters -- parameters learned by the model.
    """
    n_x = X.shape[0]
    n_y =  2
    
    # Initialize parameters    
    P1, P2, V1, V2 , M1, M2 = initialize_P(n_x, h, n_y)
    
    # feed forward
    loss, grad1, grad2 = feed_forward(X, P1, P2, a1, a2, a, right_side)
    cost  = loss(P1, P2)

    # conditions for stopping           
    while cost> tol and iter < ITs:
    
        # Feed forward propagation:        
        loss, grad1, grad2 = feed_forward(X, P1, P2, a1, a2, a, right_side)
        
        # Cost function. 
        cost  = loss(P1, P2)
       
        
        # gradients
        grad1 = grad1(P1, P2) 
        grad2 = grad2(P1, P2)
        
        # Gradient descent parameter update.            
        P1, P2, V1, V2, M1, M2, t = update_Adam(P1, P2, V1, V2, M1, M2, grad1, grad2, eta = 0.001)
        iter += 1   
        # Print the cost every 10 iterations
        if print_cost and iter % 10 == 0:
            print ("Cost after iteration %i: %f" %(iter, cost))

    return P1, P2


# function for implementation of RK4
def RungeKutta4(f, a, b, u, N):
    """
    Inputs:
    ========
    f    : the right side fucntion corresoponding to the differential equations
    a, b : end point of the domian
    u    : the unknown
    N    : number of points between a and b.
    
    Output:
    =========
    Time : discrete points
    result: the solution
    """
    h = (b - a)/(N-1)
    result = []
    Time = []
    result.append(u)
    Time.append(a)
    t = a
    for k in range(1, N):
        k1 = f(t,u)
        k2 = f(t+h/2.0, u+h*k1/2.0)
        k3 = f(t+h/2.0, u+h*k2/2.0)
        k4 = f(t+h, u+h*k3)
        u = u+h*(k1+2*k2+2*k3+k4)/6.0
        t = t+h
        result.append(u)
        Time.append(t)
    return np.array(Time), np.array(result)