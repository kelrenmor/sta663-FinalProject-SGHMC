<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['../../eigen3']
setup_pybind11(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <stdexcept>
#include <algorithm> // std::random_shuffle
#include <random>

#include <Eigen/LU>
#include <Eigen/Dense>

namespace py = pybind11;
using std::default_random_engine;
using std::normal_distribution;
        
// start random number engine with fixed seed
default_random_engine re{1234};
// set up random normal rnorm to work like in python
normal_distribution<double> norm(0, 1); // mean and standard deviation
auto rnorm = bind(norm, re);

// fill xs with draws from N(0,1) and return this n x 1 dim vector
Eigen::MatrixXd rnorm_vec(int n) {
    Eigen::MatrixXd xs = Eigen::MatrixXd::Zero(n, 1);
    for (int i=0; i<n; i++) {xs(i,0) = rnorm();}
    return xs;
}
    
// get noisy gradient of Fig1 example from Chen et. al. paper
Eigen::MatrixXd gradU_noisyFig1(Eigen::MatrixXd theta) {
    Eigen::MatrixXd xs = -4*theta.array() + 4*theta.array().pow(3) + 2*rnorm_vec(theta.rows()).array();
    return xs;
} 
    
// get gradient for mixture of normals example
Eigen::MatrixXd gradU_mixNormals(Eigen::MatrixXd theta, Eigen::MatrixXd x, int n, int batch_size) {
    int p = theta.rows();
    Eigen::ArrayXd c_0 = theta(0,0) - x.array();
    Eigen::ArrayXd c_1 = theta(1,0) - x.array();
    Eigen::ArrayXd star = 0.5 * (-0.5 * c_0.pow(2)).exp() + 0.5 * (-0.5 * c_1.pow(2)).exp();
    Eigen::ArrayXd star_prime;
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(p, 1);
    for (int i=0; i<p; i++) {
        star_prime = 0.5 * (-0.5 * (theta(i,0) - x.array()).pow(2)).exp() * (theta(i,0) - x.array());
        grad(i,0) = -theta(i,0)/10 - (n/batch_size)*(star_prime/star).sum();
    }
    return grad;
} 

/* 
SGHMC algorithm as described in the paper by Chen et al.
Note that users specify the choice of which gradient function to use by gradU_choice, 
but the gradient function itself must be available in this file
*/
Eigen::MatrixXd sghmc(std::string gradU_choice, Eigen::MatrixXd eta, int niter, Eigen::MatrixXd alpha, Eigen::MatrixXd theta_0, Eigen::MatrixXd V_hat, Eigen::MatrixXd dat, int batch_size){
    // Initialization and checks
    int p = theta_0.rows(); // dimension of the thing you're sampling
    int n = dat.rows();     // number of data observations
    int p_dat = dat.cols(); // 2nd dimension of data
    int nbatches = n / batch_size; // how many batches data will be broken into
    // Set up dat_temp and dat_batch for use in loop below, as well as gradU_batch
    Eigen::MatrixXd dat_temp = dat;
    Eigen::MatrixXd dat_batch = Eigen::MatrixXd::Zero(batch_size, p_dat);
    Eigen::MatrixXd gradU_batch = Eigen::MatrixXd::Zero(p, 1);
    // set up matrix of 0s to hold samples
    Eigen::MatrixXd theta_samps = Eigen::MatrixXd::Zero(p, niter*(n/batch_size));
    // vector to hold indices for shuffling
    std::vector<int> ind;
    // fix beta_hat as described on pg. 6 of paper
    Eigen::MatrixXd beta_hat = 0.5 * V_hat * eta;
    // We're sampling from a N(0, 2(alpha - beta_hat) @ eta)
    // so this must be a positive definite matrix
    Eigen::MatrixXd Sigma = 2.0 * (alpha - beta_hat) * eta;
    Eigen::LLT<Eigen::MatrixXd> lltOfSig(Sigma); // compute the Cholesky decomposition of Sigma
    if(lltOfSig.info() == Eigen::NumericalIssue){ // check if we got error, and break out if so
        return theta_samps; // will just give back all-0s
    }
    Eigen::MatrixXd Sig_chol = lltOfSig.matrixL(); // get L in Chol decomp
    if(batch_size > n){ // Need batch size to be <= the amount of data
        return theta_samps; // will just give back all-0s
    }
    // initialize more things! (nu and theta, to be specific)
    Eigen::LLT<Eigen::MatrixXd> lltOfeta(eta); // compute the Cholesky decomposition of eta
    Eigen::MatrixXd eta_chol = lltOfeta.matrixL(); // get L in Chol decomp
    Eigen::MatrixXd nu = eta_chol * rnorm_vec(p); // initialize nu
    Eigen::MatrixXd theta = theta_0; // initialize theta 
    
    // loop through algorithm to get niter*batch_size samples
    int big_iter = 0;
    for (int it=0; it<niter; it++) {
        
        // shuffle rows of dat to get dat_temp 
        Eigen::VectorXi indices = Eigen::VectorXi::LinSpaced(dat.rows(), 0, dat.rows());
        std::random_shuffle(indices.data(), indices.data() + dat.rows());
        dat_temp = indices.asPermutation() * dat;
        
        // Resample momentum every epoch
        nu = eta_chol * rnorm_vec(p); // sample from MV normal
        
        // loop through the batches
        int count_lower = 0;
        int count_upper = batch_size;
        for (int b=0; b<nbatches; b++){
            int batch_ind = 0;
            for (int ind_temp=count_lower; ind_temp<count_upper; ind_temp++){
                dat_batch.row(batch_ind) = dat_temp.row(ind_temp);
                batch_ind += 1;
            }
            count_lower += batch_size; // add batch size to each iterator
            count_upper += batch_size;
            if (gradU_choice == "fig1"){
                gradU_batch = gradU_noisyFig1(theta);
            } else if (gradU_choice == "mixture_of_normals"){
                gradU_batch = gradU_mixNormals(theta, dat_batch, n, batch_size);
            } else {
                return theta_samps; // will just give back all-0s
            }
            nu = nu - eta * gradU_batch - alpha * nu + Sig_chol * rnorm_vec(p);
            theta = theta + nu;
            theta_samps.col(big_iter) = theta;
            big_iter += 1;
        }
    }

    return theta_samps;
}

PYBIND11_MODULE(wrap, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("sghmc", &sghmc);
}