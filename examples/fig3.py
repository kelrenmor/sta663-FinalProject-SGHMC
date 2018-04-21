### From their Figure 3 example at these links:
#https://github.com/tqchen/ML-SGHMC/blob/master/matlab/figure3/sghmc.m
#https://github.com/tqchen/ML-SGHMC/blob/master/matlab/figure3/figure3b.m

# WHY DOESN'T THIS WORK???

import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
import seaborn as sns

def sghmc( gradU, eta, L, alpha, x, V ):
    m = len(x)
    samps = np.zeros( (m, L) )
    beta = V * eta * 0.5

    if beta > alpha:
        print('too big eta');

    sigma = np.sqrt( 2 * eta * (alpha-beta) )
    p = np.random.normal(size=( m, 1 )) * np.sqrt( eta )
    momentum = 1 - alpha

    for t in range(L):
        p = p * momentum - gradU( x ) * eta + np.random.normal(size=(2,1)) * sigma
        x = x + p
        samps[:,t] = x.reshape(-1,2)
    
    return samps

V = 1
# covariance matrix
rho = 0.9
covS = np.array([ [1, rho], [rho, 1] ])
invS = np.linalg.inv(covS)
# intial x
x = np.zeros(2).reshape(2,1)
alpha = 0.05
etaSGHMC = 0.05
# number of steps 
L = 1000
nset = 5

def log_norm(X):
    return(-(1/2)*np.log(np.linalg.det(covS))-(1/2)*X.T@invS@X)

# Automatic differentiation to get the jacobian
gradUTrue = jacobian(log_norm, argnum=0)

# make sure grad gives what it should
print(-invS@np.array([1.0,1.0]))
print(gradUTrue(np.array([1.0,1.0])))

# define noisy gradient like they do
def gradUNoise(X):
    return -invS@X  + np.random.normal(size=( 2, 1 ))

'''
for i = 1 : nset
    dscale = (0.6^(i-1));
    eta = etaSGHMC * dscale*dscale;
    dsghmc = sghmc( gradUNoise, eta, L, alpha*dscale, x, V );
    covESGHMC(:,:,i) = dsghmc * dsghmc' / L;
    meanESGHMC(:,i) = mean( dsghmc, 2 );
    SGHMCeta(i) = eta;
    SGHMCauc(i) = mean(aucTime( dsghmc, 1 ));
'''

i = 3
dscale = (0.6**(i-1))
eta = etaSGHMC * dscale*dscale
dsghmc = sghmc( gradUNoise, eta, L, alpha*dscale, x, V )

# look at first dimension (should be normal)
sns.kdeplot(dsghmc[0,:])