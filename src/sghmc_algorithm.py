def is_pos_def(X):
    '''Check whether a matrix X is pos definite.
    Returns True or False, depending on X.
    '''
    return np.all(np.linalg.eigvals(X) > 0)

def sghmc(gradU, eta, niter, alpha, theta_0, V_hat, data):
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
    dim_theta = len(theta_0)
    # set up matrix of 0s to hold samples
    theta_samps = np.zeros((dim_theta, niter))
    # fix beta_hat as described on pg. 6 of paper
    beta_hat = 0.5 * V_hat @ eta
    # We're sampling from a N(0, 2(alpha - beta_hat) @ eta)
    # so this must be a positive definite matrix
    Sigma = 2 * (alpha - beta_hat) @ eta
    if not is_pos_def( Sigma ): 
        print("Error: (alpha - beta_hat) eta not pos def")
        return

    # initialize nu and theta 
    nu = np.random.multivariate_normal(np.zeros(dim_theta), eta).reshape(dim_theta,-1)
    theta = theta_0
    
    # loop through algorithm to get niter samples
    for i in range(niter):
        nu = nu - eta @ gradU(theta, data) - alpha @ nu + \
             np.random.multivariate_normal(np.zeros(nu.shape[0]), Sigma).reshape(nu.shape[0], -1)
        theta = theta + nu
        theta_samps[:,i] = theta.reshape(-1,theta.shape[0])

    return theta_samps