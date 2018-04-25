import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
import seaborn as sns
import pystan
from sghmc.sghmc_algorithm import sghmc

### Easy test example based on Figure 1 from the paper
### See pg. 6 of Chen et. al. for their results/comparison

# Log likelihood function
def Ex1_sghmc():
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
    
    # plot the samples from the algorithm and save to a file
    kdeplt = sns.kdeplot(samps_sghmc.reshape(-1)) # Plot the joint density
    fig = kdeplt.get_figure()
    fig.savefig('Example1_a.png')
    return(samps_sghmc)

################################################################################

# Use the hmc function from the pyhmc (at all default settings)
# to sample from this distribution.
# See https://pythonhosted.org/pyhmc/ for function details.
from pyhmc import hmc
def Ex1_hmc():
    def U(theta):
        return(-2*theta**2 + theta**4)
    # True gradient
    gradU = jacobian(U, argnum=0)
    # Noisy gradient, based on what they do in the paper for Fig 1
    def noisy_gradU(theta, x, n, batch_size):
        '''Noisy gradient \Delta\tilde{U}(\theta)=\Delta U(\theta)+N(0,4)
        Extra args (x, n, batch_size) for compatibility with sghmc()'''
        return -4*theta + 4*theta**3 + np.random.normal(0,2)
    # define your probability distribution
    def logprob(theta):
        logp = -2*theta**2 + theta**4
        grad = -4*theta + 4*theta**3
        return logp, grad
    # run the HMC sampler (use same theta_0 and niter as SGHMC)
    np.random.seed(1234)
    # Don't actually need 'data' in this example, just use
    # it as a place-holder to fit into our function.
    n = 100
    x = np.array([np.random.normal(0, 1, (n,1))]).reshape(-1,1)
    # Set up start values and tuning params
    theta_0 = np.array([0.0]) # Initialize theta
    batch_size = n # since we're not actually using the data, don't need to batch it
    niter = 500000 # Lots of iterations
    samps_hmc = hmc(logprob, x0=theta_0, n_samples=niter) 
    # plot the samples from the HMC algorithm
    sns.kdeplot(samps_hmc.reshape(-1))

    # The hmc() function with that many iterations goes kind of insane... 
    # How about less samples? Also redefine logprob(theta) to use funs from above
    #def logprob(theta):
        #return U(theta, x, n, batch_size), gradU(theta, x, n, batch_size).reshape(1)
    # run the HMC sampler (use same theta_0 as above but fewer samples)
   #samps_hmc = hmc(logprob, x0=theta_0, n_samples=500) # NOPE! Still looks bad.
    # plot the samples from the HMC algorithm
    
    # plot the samples from the algorithm and save to a file
    kdeplt = sns.kdeplot(samps_hmc.reshape(-1)) # Plot the joint density
    fig = kdeplt.get_figure()
    fig.savefig('Example1_b.png')
    
    return(samps_hmc)

################################################################################

### Make figure showing the truth compared to our samples
def Ex1_truth():
    # define exp(-U(theta)), the true density
    def expU(theta):
        return np.exp(-1*U(theta))/5.36516 # norm constant from Wolfram Alpha

    samps_sghmc = np.load("samps_sghmc.npy") # load data run with tons of iters from C++ code
    test_theta = np.linspace(samps_sghmc.min(),samps_sghmc.max(),200)
    
    # just truth
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(test_theta,expU(test_theta))
    fig.savefig('Example1_truth.png')
    
    # truth and C++ samples
    #fig, ax = plt.subplots()
    #sns.kdeplot(samps_sghmc[0,:], ax=ax)  # plot samples (blue)
    #plt.plot(test_theta,expU(test_theta)) # plot true density (orange)

################################################################################

# Stan code for our Example 1 :
import pystan
def Ex1_stan():
    stan_ex1_code = '''
    functions {
      // Define log probability density function
      real Ex1_lpdf(real theta) {return -1* (-2*theta^2 + theta^4);}
    }
    data {
    }
    parameters {
      real theta;
    }
    model {
      // Sample theta
      theta ~ Ex1_lpdf();
    }
    
    '''
    ex1_dat = {}
    
    sm = pystan.StanModel(model_code=stan_ex1_code)
    fit = sm.sampling(data=ex1_dat, iter=100000, chains=4)
    
    # return a dictionary of arrays
    la = fit.extract(permuted=True)  # return a dictionary of arrays
    thet_samps = la['theta']
    
    # PLOT and save output
    kdeplt = sns.kdeplot(thet_samps)
    fig = kdeplt.get_figure()
    fig.savefig('Example1_c.png')
    
    samps_stan = thet_samps
    return(samps_stan)
