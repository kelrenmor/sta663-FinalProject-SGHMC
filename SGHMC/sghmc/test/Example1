#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
import seaborn as sns
from sghmc.sghmc_algorithm import sghmc

### Easy test example based on Figure 1 from the paper
### See pg. 6 of Chen et. al. for their results/comparison

# Log likelihood function
def U(theta):
    return(-2*theta**2 + theta**4)
# True gradient
gradU = jacobian(U, argnum=0)
# Noisy gradient, based on what they do in the paper for Fig 1
def noisy_gradU(theta, x, n, batch_size):
    '''Noisy gradient \Delta\tilde{U}(\theta)=\Delta U(\theta)+N(0,4)
    Extra args (x, n, batch_size) for compatibility with sghmc()'''
    return -4*theta + 4*theta**3 + np.random.normal(0,2)
# Set random seed
np.random.seed(1234)
# Don't actually need 'data' in this example, just use
# it as a place-holder to fit into our function.
n = 100
x = np.array([np.random.normal(0, 1, (n,1))]).reshape(-1,1)
# Set up start values and tuning params
theta_0 = np.array([0.0]) # Initialize theta
p = theta_0.shape[0]
eta = 0.001 * np.eye(p) # make this small
alpha = 0.01 * np.eye(p)
V = np.eye(p)
batch_size = n # since we're not actually using the data, don't need to batch it
niter = 500000 # Lots of iterations
# run SGHMC sampler
samps_sghmc = sghmc(noisy_gradU, eta, niter, alpha, theta_0, V, x, batch_size)
# plot the samples from the algorithm
sns.kdeplot(samps_sghmc.reshape(-1))

################################################################################

# Use the hmc function from the pyhmc (at all default settings)
# to sample from this distribution.
# See https://pythonhosted.org/pyhmc/ for function details.
from pyhmc import hmc
# define your probability distribution
def logprob(theta):
    logp = -2*theta**2 + theta**4
    grad = -4*theta + 4*theta**3
    return logp, grad
# run the HMC sampler (use same theta_0 and niter as SGHMC)
samps_hmc = hmc(logprob, x0=theta_0, n_samples=niter) 
# plot the samples from the HMC algorithm
sns.kdeplot(samps_hmc.reshape(-1))

# The hmc() function with that many iterations goes kind of insane... 
# How about less samples? Also redefine logprob(theta) to use funs from above
def logprob(theta):
    return U(theta, x, n, batch_size), gradU(theta, x, n, batch_size).reshape(1)
# run the HMC sampler (use same theta_0 as above but fewer samples)
samps_hmc = hmc(logprob, x0=theta_0, n_samples=500) # NOPE! Still looks bad.
# plot the samples from the HMC algorithm
sns.kdeplot(samps_hmc.reshape(-1))
