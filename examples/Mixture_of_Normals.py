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

#import sghmc_algorithm


## Example #1:
## Sampling from a mixture of normals in 1-D
## SAMPLING MODEL: x ~ 0.5 * N(mu1, 1) + 0.5 * N(mu2, 1)
## PRIORS: p(mu1) = p(mu2) = N(0,1)

def log_prior(theta):
    return(-0.5*theta.T @ theta)
    
    
def log_lik(theta, x):
    return(-0.5*(np.log(.5) * np.sum((x-theta[0])**2) + np.log(0.5)*np.sum((x-theta[1]**2))))
    

def U(theta, x):
    return(-log_prior(theta) - log_lik(theta, x))
    
def batch_data(data, batch_size):
    n = data.shape[0]
    p = data.shape[1]
    if n % batch_size != 0:
        n = (n // batch_size) * batch_size
    ind = np.arange(n)
    np.random.shuffle(ind)
    n_batches = n // batch_size
    data = data[ind].reshape(batch_size, p, n_batches)
    return(data, n_batches)
    
    
# Setup the data
p = 2 #dimension of theta
theta = np.array([-1, 1]).reshape(p,-1)
n = 100
x = np.array([0.5 * np.random.normal(theta[0], 1, (n,1)),
              0.5 * np.random.normal(theta[1], 1, (n,1))]).reshape(-1,1)

# Automatic differentiation to get the jacobian
gradU = jacobian(U)


# Now for sampling

# Initialize theta
theta_0 = np.random.normal(size=(p,1))

# Initialize hyper parameters:

# learning rate
eta = 0.1 * np.eye(p)

# Friction rate
alpha = 0.1 * np.eye(p)

# Arbitrary guess at covariance of noise from mini-batching the data
V = np.eye(p)
niter = 100


# Run sampling algorithm
samps = sghmc(gradU, eta, niter, alpha, theta_0, V, x, 20)





