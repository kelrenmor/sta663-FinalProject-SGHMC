#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:33:22 2018

@author: isaaclavine and kellymoran
"""

### Run example 1:

from sghmc.test import Example1

# get the samples from Example 1, plus save an output graph
samps_sghmc = Example1.Ex1_sghmc()

# get the samples from Example 1, plus save an output graph
samps_hmc = Example1.Ex1_hmc()

# get the samples from Example 1, plus save an output graph
samps_stan = Example1.Ex1_stan()


### Run Mixture of Normals:

from sghmc.test import Mixture_of_Normals

# get the samples from Example 1, plus save an output graph
samps_sghmc = Mixture_of_Normals.MoN_sghmc()
kdeplt = sns.kdeplot(samps_sghmcs[0,:], samps_sghmc[1,:]) # Plot the joint density
# Note: Had an issue where plots from within the function look
# different than plots outside

# get the samples from Example 1, plus save an output graph
samps_hmc = Mixture_of_Normals.MoN_hmc()

# get the samples from Example 1, plus save an output graph
samps_stan = Mixture_of_Normals.MoN_stan()