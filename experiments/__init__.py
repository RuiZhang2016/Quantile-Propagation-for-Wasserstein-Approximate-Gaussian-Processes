# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Examples for GPy.

The examples in this package usually depend on `pods <https://github.com/sods/ods>`_ so make sure 
you have that installed before running examples.
"""
import sys
import os

# for Mac OS
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
else:
    pass
ROOT_PATH = os.path.split(os.getcwd())[0]

datanames = {0: 'ionosphere', 1: 'breast_cancer', 2: 'crabs', 3: 'pima',
             4: 'sonar',5:'glass', 6: 'wine12', 7: 'wine23', 8: 'wine13'}

if os.path.isdir(ROOT_PATH+'/res'):
    pass
else:
    os.mkdir(ROOT_PATH+'/res')

if os.path.isdir(ROOT_PATH+'/res/objs'):
    pass
else:
    os.mkdir(ROOT_PATH+'/res/objs')


if os.path.isdir(ROOT_PATH+'/res/paper_clf'):
    pass
else:
    os.mkdir(ROOT_PATH+'/res/paper_clf')

if os.path.isdir(ROOT_PATH+'/res/paper_poi'):
    pass
else:
    os.mkdir(ROOT_PATH+'/res/paper_poi')


if os.path.isdir(ROOT_PATH+'/data/split_data/'):
    pass
else:
    os.mkdir(ROOT_PATH + '/data/split_data/')

