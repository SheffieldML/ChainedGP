#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from setuptools import setup, Extension

# Version number
version = '0.0.1'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# compile_flags = ["-march=native", '-fopenmp', '-O3', ]
compile_flags = ['-fopenmp', '-O3']

ext_mods = [Extension(name='quad_cython',
                      sources=['chained_gp/cython/quad_cython.c', 'chained_gp/cython/quad_utils.c'],
                      include_dirs=[np.get_include()],
                      extra_compile_args=compile_flags)]
                      # extra_link_args=['-lgomp'])]

setup(name = 'ChainedGP',
      version = version,
      author = 'Alan Saul, James Hensman, Aki Vehtari, Neil Lawrence',
      author_email = "alan.daniel.saul@gmail.com",
      description = ("Chained Gaussian processes"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes",
      url = "http://www.alansaul.com",
      ext_modules=ext_mods,
      packages = ["chained_gp",
                  ],
      package_dir={'chained_gp': 'chained_gp'},
      py_modules = ['chained_gp.__init__'],
      long_description="Chained Gaussian process models",
      install_requires=['numpy>=1.7', 'scipy>=0.12', 'GPy', 'theano'],
      extras_require = {'docs':['matplotlib >=1.3','Sphinx','IPython', 'seaborn', 'descartes']},
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )

