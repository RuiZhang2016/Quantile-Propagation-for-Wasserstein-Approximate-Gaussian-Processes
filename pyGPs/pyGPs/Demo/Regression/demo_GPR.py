from __future__ import print_function
#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [dan dot marthaler at gmail dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGPs.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 18/02/2014
#================================================================================

import sys
import pickle
import os


# for Mac OS
if sys.platform == 'darwin':
    import matplotlib

    matplotlib.use('TkAgg')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    os.environ['proj'] = '/home/rzhang/PycharmProjects/WGPC'
sys.path.append(os.environ['proj'] + '/pyGPs')
sys.path.append(os.environ['proj'])

import matplotlib.pyplot as plt
import pyGPs
import numpy as np
from pyGPs.Demo.Classification.GPC_experiments import *
# This demo will not only introduce GP regression model,
# but provides a gerneral insight of our tourbox.

# You may want to read it before reading other models.
# current possible models are:
#     pyGPs.GPR          -> Regression
#     pyGPs.GPC          -> Classification
#     pyGPs.GPR_FITC     -> Sparse GP Regression
#     pyGPs.GPC_FITC     -> Sparse GP Classification
#     pyGPs.GPMC         -> Muli-class Classification



print('')
print('---------------------GPR DEMO-------------------------')

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
demoData = np.load('regression_data.npz')
x = demoData['x']            # training data
y = demoData['y']            # training target
z = demoData['xstar']        # test data

f = lambda x: 0.3*x+np.cos(3*x)
x = np.linspace(-2,2,30).reshape((-1,1))
y = np.array([f(i[0]) for i in x]).reshape((-1,1))
y[10] += 2
z = np.linspace(-2,5,20).reshape((-1,1))
zy = np.array([f(i[0]) for i in z]).reshape((-1,1))

#----------------------------------------------------------------------
# A five-line example
#----------------------------------------------------------------------
print('Basic Example')
modelEP = pyGPs.GPR()          # model
# modelEP.useInference('EP')
modelEP.useLikelihood('Laplace_EP')
modelQP = pyGPs.GPR()
f = lambda x:x
# modelQP.useInference('QP',f,f)
modelQP.useLikelihood('Laplace_QP')
# f1, f2 = None, None #interp_fs()
# model.useInference('QP',f1,f2)
# model.setOptimizer('BFGS')
# model.useLikelihood('Laplace_QP')

fig = plt.figure(figsize=(14,6))
models = [modelEP,modelQP]
for i in range(len(models)):
    model = models[i]
    print('Inference Method: ', model.inffunc.name)
    print('Before Optimization')
    model.setData(x,y)
    m = pyGPs.mean.Zero() # pyGPs.mean.Const() + pyGPs.mean.Linear()
    model.setPrior(mean=m)

    # model.predict(z)             # predict test cases (before optimization)
    # model.plot()                 # and plot result
    model.optimize(x, y)           # optimize hyperparamters (default optimizer: single run minimize)
    # print('After Optimization')
    model.predict(z)               # predict test cases
    ax = fig.add_subplot(1,2,i+1)
    model.plot(ax)                   # and plot result
    ax.set_title('Laplace-{}'.format(model.inffunc.name))
    print('negative log marginal likelihood: ', model.nlZ)
plt.show()
#----------------------------------------------------------------------
# Now lets do another example to get more insight to the toolbox
#----------------------------------------------------------------------
# print('More Advanced Example (using a non-zero mean and Matern7 kernel)')
# model = pyGPs.GPR()           # start from a new model
# model.useLikelihood('Laplace_QP')
# Specify non-default mean and covariance functions
# SEE doc_kernel_mean for documentation of all kernels/means
# m = pyGPs.mean.Zero()  # pyGPs.mean.Const() + pyGPs.mean.Linear()
# k = pyGPs.cov.Matern(d=7) # Approximates RBF kernel
# model.setPrior(mean=m,kernel=k)
# print(model.inffunc.name)


# Specify optimization method (single run "Minimize" by default)
# @SEE doc_optimization for documentation of optimization methods
#model.setOptimizer("RTMinimize", num_restarts=30)
#model.setOptimizer("CG", num_restarts=30)
#model.setOptimizer("LBFGSB", num_restarts=30)

# Instead of getPosterior(), which only fits data using given hyperparameters,
# optimize() will optimize hyperparamters based on marginal likelihood
# the deafult mean will be adapted to the average value of the training labels.
# ..if you do not specify mean function by your own.
# model.optimize(x, y)

# There are several properties you can get from the model
# For example:
#   model.nlZ
#   model.dnlZ.cov
#   model.dnlZ.lik
#   model.dnlZ.mean
#   model.posterior.sW
#   model.posterior.alpha
#   model.posterior.L
#   model.covfunc.hyp
#   model.meanfunc.hyp
#   model.likfunc.hyp
#   model.ym (predictive means)
#   model.ys2 (predictive variances)
#   model.fm (predictive latent means)
#   model.fs2 (predictive latent variances)
#   model.lp (log predictive probability)
# print('Optimized negative log marginal likelihood:', round(model.nlZ,3))


# Predict test data
# output mean(ymu)/variance(ys2), latent mean(fmu)/variance(fs2), and log predictive probabilities(lp)
# ym, ys2, fmu, fs2, lp = model.predict(x)


# Set range of axis for plotting
# NOTE: plot() is a toy method only for 1-d data
# model.plot()
# model.plot(axisvals=[-1.9, 1.9, -0.9, 3.9]))


#----------------------------------------------------------------------
# A bit more things you can do
#----------------------------------------------------------------------

# [For all model] Speed up prediction time if you know posterior in advance
# model.getPosterior(z,zy)    # already known before
# print(model.nlZ)

# ym, ys2, fmu, fs2, lp = model.predict_with_posterior(post,z)
# ...other than model.predict(z)


# [Only for Regresstion] Specify noise of data (sigma=0.1 by default)
# You don't need it if you optimize it later anyway
# model.setNoise(log_sigma=np.log(0.1))

print('--------------------END OF DEMO-----------------------')

