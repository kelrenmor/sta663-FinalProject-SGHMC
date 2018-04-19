#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:36:51 2018

@author: isaaclavine
"""

import ad
# import numpy as np
import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import jacobian


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
    
    
# Setup the data
theta = np.array([-1, 1])
n = 100
x = np.array([0.5 * np.random.normal(theta[0], 1, (n,1)),
              0.5 * np.random.normal(theta[1], 1, (n,1))]).reshape(-1,1)

# Automatic differentiation to get the jacobian
gradU = jacobian(U)


# Now for sampling

# Initialize theta
theta = np.random.normal(size=(2,1))

# Initialize hyper parameters:

# learning rate
eta = 0.1 * np.eye(2)

# Friction
alpha = 0.1 * np.eye(2)