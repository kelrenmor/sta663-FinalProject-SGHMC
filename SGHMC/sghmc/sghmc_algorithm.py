import autograd.numpy as np


def is_pos_def(X):
    '''Check whether a matrix X is pos definite.
    Returns True or False, depending on X.
    '''
    return np.all(np.linalg.eigvals(X) > 0)

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

def sghmc(gradU, eta, niter, alpha, theta_0, V_hat, dat, batch_size):
    '''Define SGHMC as described in the paper
    Tianqi Chen, Emily B. Fox, Carlos Guestrin 
    Stochastic Gradient Hamiltonian Monte Carlo 
    ICML 2014.

    The inputs are:
    gradU = gradient of U
    eta = eps^2 M^(-1)
    niter = number of samples to generate
    alpha = eps M^(-1) C
    theta_0 = initial val of parameter(s) to be sampled
    V_hat = estimated covariance matrix of stoch grad noise
    See paper for more details

    The return is:
    A np.array of positions of theta.'''

    ### Initialization and checks ###
    # get dimension of the thing you're sampling
    p = len(theta_0)
    # set up matrix of 0s to hold samples
    n = dat.shape[0]
    theta_samps = np.zeros((p, niter*(n // batch_size)))
    # fix beta_hat as described on pg. 6 of paper
    beta_hat = 0.5 * V_hat @ eta
    # We're sampling from a N(0, 2(alpha - beta_hat) @ eta)
    # so this must be a positive definite matrix
    Sigma = 2 * (alpha - beta_hat) @ eta
    if not is_pos_def( Sigma ): 
        print("Error: (alpha - beta_hat) eta not pos def")
        return
    # Need batch size to be <= the amount of data
    if (batch_size > n): 
        print("Error: batch_size must be <= number of data points")
        return

    # initialize nu and theta 
    nu = np.random.multivariate_normal(np.zeros(p), eta).reshape(p,-1)
    theta = theta_0
    
    # loop through algorithm to get niter samples
    it = 0
    for i in range(niter):
        dat_resh, nbatches = batch_data(dat, batch_size)
        
        # Resample momentum every epoch
        nu = np.random.multivariate_normal(np.zeros(p), eta).reshape(p,-1)
        
        for batch in range(nbatches):
            gradU_batch = gradU(theta, dat_resh[:,:,batch], n, batch_size).reshape(p,-1)
            nu = nu - eta @ gradU_batch - alpha @ nu + \
                 np.random.multivariate_normal(np.zeros(p), Sigma).reshape(p, -1)
            theta = theta + nu
            theta_samps[:,it] = theta.reshape(-1,p)
            it = it + 1
        
    return theta_samps

def sghmc_cleaned(gradU, eta, niter, alpha, theta_0, V_hat, dat, batch_size):
    '''Define SGHMC as described in the paper
    Tianqi Chen, Emily B. Fox, Carlos Guestrin 
    Stochastic Gradient Hamiltonian Monte Carlo 
    ICML 2014.

    The inputs are:
    gradU = gradient of U
    eta = eps^2 M^(-1)
    niter = number of samples to generate
    alpha = eps M^(-1) C
    theta_0 = initial val of parameter(s) to be sampled
    V_hat = estimated covariance matrix of stoch grad noise
    See paper for more details

    The return is:
    A np.array of positions of theta.'''

    ### Initialization and checks ###
    # get dimension of the thing you're sampling
    p = len(theta_0)
    # set up matrix of 0s to hold samples
    n = dat.shape[0]
    theta_samps = np.zeros((p, niter*(n // batch_size)))
    # fix beta_hat as described on pg. 6 of paper
    beta_hat = 0.5 * V_hat @ eta
    # We're sampling from a N(0, 2(alpha - beta_hat) @ eta)
    # so this must be a positive definite matrix
    Sigma = 2 * (alpha - beta_hat) @ eta
    Sig_chol = np.linalg.cholesky(Sigma)
    if not is_pos_def( Sigma ): 
        print("Error: (alpha - beta_hat) eta not pos def")
        return
    # Need batch size to be <= the amount of data
    if (batch_size > n): 
        print("Error: batch_size must be <= number of data points")
        return

    # initialize nu and theta 
    nu = np.random.multivariate_normal(np.zeros(p), eta).reshape(p,-1)
    theta = theta_0

    # set up for Chol decomp for MV normal sampling of nu every epoch
    eta_chol = np.linalg.cholesky(eta)

    # loop through algorithm to get niter samples
    it = 0
    for i in range(niter):
        dat_resh, nbatches = batch_data(dat, batch_size)
        
        # Resample momentum every epoch
        nu = np.sqrt(eta_chol) @ np.random.normal(size=p).reshape(p,-1) # sample from MV normal
        
        for batch in range(nbatches):
            gradU_batch = gradU(theta, dat_resh[:,:,batch], n, batch_size).reshape(p,-1)
            nu = nu - eta @ gradU_batch - alpha @ nu + \
                 Sig_chol @ np.random.normal(size=p).reshape(p,-1) # sample from MV normal
            theta = theta + nu
            theta_samps[:,it] = theta.reshape(-1,p)
            it = it + 1
        
    return theta_samps
