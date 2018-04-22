#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:36:51 2018

@author: isaaclavine
"""

import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
import seaborn as sns

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
       
# Automatic differentiation to get the gradient
gradU = jacobian(U, argnum=0)

# Set random seed
np.random.seed(1234)
# Set up the data
p = 2 #dimension of theta
theta = np.array([-3.0, 3.0]).reshape(p,-1)
n = 10000
x = np.array([np.random.normal(theta[0], 1, (n,1)),
              np.random.normal(theta[1], 1, (n,1))]).reshape(-1,1)

# Plot the empirical distribution of x (sampled from mixture of normals)
sns.kdeplot(x.reshape(-1), bw=1)

## Initialize parameters and sample 

# Initialize mean parameters
#theta_0 = np.random.normal(size=(p,1))
theta_0 = theta # initialize at "true" value for testing

# Initialize tuning parameters:
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

# See below that the mean estimates are VERY precice
# using this sampler, e.g. for \mu_0 we have samples in the range of
# roughly 2.96 to 3.04, and for \mu_1 from -3.05 to -2.98 

# Plot the density of each mean in mixture
sns.kdeplot(samps[0,:]) # should be -3
sns.kdeplot(samps[1,:]) # should be 3

# Plot the joint density
sns.kdeplot(samps[0,:], samps[1,:])


################################################################################

# Use the hmc function from the pyhmc (at all default settings)
# to sample from this distribution.
# See https://pythonhosted.org/pyhmc/ for function details.
from pyhmc import hmc

# define your probability distribution
# note some fiddling to get dimensions to be compatible
def logprob(theta):
    logp = np.sum(U(theta, x=x, n=n, batch_size=n))
    gradu = gradU(theta, x=x, n=n, batch_size=n).reshape((-1,))
    return logp, gradu

# run the HMC sampler 
# ideally would use same theta_0 and niter as SGHMC, 
# but computing the full gradient is prohibtively slow!!
samps_hmc = hmc(logprob, x0=theta_0.reshape((-1)), n_samples=100) 
# plot the samples from the HMC algorithm (jointly)
sns.kdeplot(samps[0,:], samps[1,:])
