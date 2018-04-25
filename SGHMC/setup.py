#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:04:53 2018

@author: isaaclavine
"""
from distutils.core import setup

setup(
    name='SGHMC',
    version='0.1.0',
    author='Kelly Moran and Isaac Lavine',
    author_email='kelly.moran@duke.edu',
    packages=['sghmc', 'sghmc.test'],
#    scripts=['bin/Mixture_of_Normals.py','bin/Example1.py'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Implementation of SGHMC algorithm.',
    long_description=open('README.txt').read(),
    #package_data={  # Optional
    #    'housing': ['ss15husa.csv'],
    #},
    
    #entry_points={  # Optional
    #    'console_scripts': [
    #        'sample=sample:main',
    #    ],
    #},
                
                
    install_requires=[
        "autograd >= 1.2",
        "seaborn >= 0.7.0",
        "matplotlib >= 2.0.0"
    ],
    include_package_data=True
)