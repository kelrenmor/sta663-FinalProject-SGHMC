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
import pystan
from sghmc.sghmc_algorithm import sghmc

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
       
#def U2(theta, **kwargs):
#    return(-log_prior(theta) - (n/batch_size)*sum(log_lik(theta, x)))
    
#kwargs = {"n":n, "batch_size":batch_size, "x":x}

#U2(theta, **kwargs)
    
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
sns.kdeplot(samps[0,:]) # MLE is -3, prior shrinks this to 0 a little
sns.kdeplot(samps[1,:]) # MLE is 3, prior shrinks this to 0 a little

# plot the samples from the algorithm and save to a file
kdeplt = sns.kdeplot(samps[0,:], samps[1,:]) # Plot the joint density
fig = kdeplt.get_figure()
fig.savefig('MixNorm_a.png')


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
# plot the samples from the algorithm and save to a file
kdeplt = sns.kdeplot(samps[0,:], samps[1,:]) # FIGURE 2b FOR PAPER
fig = kdeplt.get_figure()
fig.savefig('MixNorm_b.png')

################################################################################

# Use the hmc function from the pystan package (at all default settings)
# to sample from this distribution.

# Stan code for a mixtures of normals model with unknown mean:
# (Used http://modernstatisticalworkflow.blogspot.com/2016/10/finite-mixture-models-in-stan.html)
stan_mix_code = '''
data {
  int N;
  vector[N] y;
  int n_groups;
  vector<lower = 0>[n_groups] sigma; // known SDs
  vector<lower=0>[n_groups]  weights; // known weight vector
}
parameters {
  vector[n_groups] mu; // unknown means
}
model {
  vector[n_groups] contributions;
  // priors
  mu ~ normal(0, 10);  
  
  // likelihood
  for(i in 1:N) {
    for(k in 1:n_groups) {
      contributions[k] = log(weights[k]) + normal_lpdf(y[i] | mu[k], sigma[k]);
    }
    target += log_sum_exp(contributions);
  }
}
'''

# Set up the data
p = 2 # dimension of theta
theta = np.array([-3.0, 3.0]).reshape(p,-1)
n = 200 # smaller for test
x = np.array([np.random.normal(theta[0], 1, (n,1)),
              np.random.normal(theta[1], 1, (n,1))]).flatten()

# Explicitly define std deviation and weights
sigma = np.array([1,1])
weights = np.array([0.5,0.5]) # equal probability

mix_dat = {'N': len(x),
               'y': x,
               'n_groups': p,
               'sigma': sigma,
               'weights': weights}

sm = pystan.StanModel(model_code=stan_mix_code)
fit = sm.sampling(data=mix_dat, iter=1000, chains=4)

# return a dictionary of arrays
la = fit.extract(permuted=True)  # return a dictionary of arrays
mu = la['mu']

# plot the samples from the algorithm and save to a file
kdeplt = sns.kdeplot(mu[:,0], mu[:,1])  # FIGURE 2c FOR PAPER
fig = kdeplt.get_figure()
fig.savefig('MixNorm_b.png')

