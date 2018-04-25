# sta663-FinalProject-SGHMC
This repository houses the final project for STA 663 (Duke University Spring 2018), an implementation of stochastic gradient Hamiltonian Monte Carlo (SGHMC) as described by Chen et. al. in their 2014 paper "Stochastic Gradient Hamiltonian Monte Carlo".

The package associated with our work is housed in the SGHMC directory. The README file in the package documentation contains installation instructions and points the user to the examples. The following features are available in this package:
- an open source license (LICENSE.txt)
- Python and C++ implementations of the SGHMC algorithm (found in sghmc/sghmc_algorithm.py and sghmc/test/wrap.cpp, respectively)
- code to generate Examples 1 and 2 from the paper (found in sghmc/test/Example1.py and sghmc/test/Mixture_of_Normals.py, respectively)
- a Jupyter notebook containing the optimization work and results found in the optimization section of the final paper (found in sghmc/test/optimization_work.ipynb)
- a code to run all of the relevant examples (found in bin/run_examples.py)

The paper for our final project is housed in the paper directory, along with the figures generated in the package functions. 

The original paper upon which the project is based is available in the root folder of the Git repo.


