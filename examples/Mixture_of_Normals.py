#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:36:51 2018

@author: isaaclavine
"""

#import ad
# import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
import seaborn as sns
#from functools import partial


#import sghmc_algorithm


## Example #1:
## Sampling from a mixture of normals in 1-D
## SAMPLING MODEL: x ~ 0.5 * N(mu1, 1) + 0.5 * N(mu2, 1)
## PRIORS: p(mu1) = p(mu2) = N(0,10)

def log_prior(theta):
    return(-(1/(2*10))*theta.T@theta)
    
    
def log_lik(theta, x):
    return(np.log(0.5 * np.exp(-0.5*(theta[0]-x)**2) + 0.5* np.exp(-0.5*(theta[1]-x)**2)))
    

def U(theta, x, n, batch_size):
    return(-log_prior(theta) - (n/batch_size)*sum(log_lik(theta, x)))
    
    
# Setup the data
p = 2 #dimension of theta
theta = np.array([-3.0, 3.0]).reshape(p,-1)
n = 10000
x = np.array([np.random.normal(theta[0], 1, (n,1)),
              np.random.normal(theta[1], 1, (n,1))]).reshape(-1,1)

# Plot the density of x
sns.kdeplot(x.reshape(-1), bw=1)
    
# Automatic differentiation to get the jacobian
gradU = jacobian(U, argnum=0)

# Now for sampling

# Initialize theta
#theta_0 = np.random.normal(size=(p,1))
theta_0 = theta

# Initialize hyper parameters:

# learning rate
eta = 0.01/n * np.eye(p)

# Friction rate
alpha = 0.1 * np.eye(p)

# Arbitrary guess at covariance of noise from mini-batching the data
V = np.eye(p)*1
niter = 100
batch_size=1000


# Run sampling algorithm
samps = sghmc(gradU, eta, niter, alpha, theta_0, V, x, batch_size)

# Plot the density of each theta
sns.kdeplot(samps[0,:])

# Plot the joint density
sns.kdeplot(samps[0,:], samps[1,:])



### Easy test example:
def U(theta, x, n=1, batch_size=1):
    return(-2*theta**2 + theta**4)    
n = 10000
x = np.array([np.random.normal(0, 1, (n,1))]).reshape(-1,1)
gradU = jacobian(U, argnum=0)
theta_0 = np.array([0.0])
niter = 200
p=1
eta = 0.01 * np.eye(p)
alpha = 0.05 * np.eye(p)
V = np.eye(p)*4
batch_size = 100
samps = sghmc(gradU, eta, niter, alpha, theta_0, V, x, batch_size)

sns.kdeplot(samps.reshape(-1))



