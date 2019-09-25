from __future__ import print_function

import sys
#

# for Mac OS
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    sys.path.append('/Users/ruizhang/PycharmProjects/WGPC/pyGPs')
else:
    sys.path.append('/home/rzhang/PycharmProjects/WGPC/pyGPs')
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

import pyGPs
from read_data import *
from core.generate_table import *
from GPC_experiments import *
from scipy import interpolate
# To have a gerneral idea,
# you may want to read demo_GPR, demo_kernel and demo_optimization first!
# Here, the focus is on the difference of classification model.

print('')
print('---------------------GPC DEMO-------------------------')

#----------------------------------------------------------------------
# Load demo data (generated from Gaussians)
#----------------------------------------------------------------------
# GPC target class are +1 and -1
demoData = np.load('classification_data.npz')
x = demoData['x']            # training data
xmean = np.mean(x)
xstd = np.std(x)
y = demoData['y']            # training target
z = demoData['xstar']        # test data
# only needed for 2-d contour plotting 
x1 = demoData['x1']          # x for class 1 (with label -1)
x2 = demoData['x2']          # x for class 2 (with label +1)     
t1 = demoData['t1']          # y for class 1 (with label -1)
t2 = demoData['t2']          # y for class 2 (with label +1)
p1 = demoData['p1']          # prior for class 1 (with label -1)
p2 = demoData['p2']          # prior for class 2 (with label +1)

## By Rui
def preproc(x, m, s):
    return (x-m)/s
# x = preproc(x,xmean,xstd)
# z = preproc(z,xmean,xstd)
# x1 = preproc(x1,xmean,xstd)
# x2 = preproc(x2,xmean,xstd)

#----------------------------------------------------------------------
# First example -> state default values
#----------------------------------------------------------------------
# print('Basic Example - Data')
# model = pyGPs.GPC()  # binary classification (default inference method: EP)
# # model.inffunc = pyGPs.inf.QP()
# # model.setOptimizer('BFGS')
# model.plotData_2d(x1,x2,t1,t2,p1,p2)
# print('Basic Example - Posterior')
# model.getPosterior(x, y)     # fit default model (mean zero & rbf kernel) with data
# print('Basic Example - Optimize')
# model.optimize(x, y, numIterations=10)     # optimize hyperparamters (default optimizer: single run minimize)
# print('Basic Example - Predict')
# model.predict(z)             # predict test cases
#
# print('Basic Example - Prediction')
# model.plot(x1,x2,t1,t2)

#----------------------------------------------------------------------
# GP classification example
#----------------------------------------------------------------------
# print('More Advanced Example')
# Start from a new model
model = pyGPs.GPC()

# Analogously to GPR
k = pyGPs.cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
model.setPrior(kernel=k)

model.getPosterior(x, y)
# print("Negative log marginal liklihood before:", round(model.nlZ,7))
model.optimize(x, y)
print("Negative log marginal liklihood optimized:", round(model.nlZ,7))

# Prediction
n = z.shape[0]
ymu, ys2, fmu, fs2, lp = model.predict(z, ys=np.ones((n,1)))
model.plot(x1,x2,t1,t2)
# pyGPs.GPC.plot() is a toy method for 2-d data
# plot log probability distribution for class +1
# model.plot(x1,x2,t1,t2)
# print(x1.shape,x2.shape,z.shape,x.shape)

def compute_I(ys, ps, ys_train):
    p1 = np.mean([e if e == 1 else 0 for e in ys_train])
    p2 = 1-p1
    H = -p1*np.log2(p1)-p2*np.log2(p2)
    assert len(ys) == len(ps)
    Is = (ys+1)/2*np.log2(ps)+(1-ys)/2*np.log2(1-ps)+H
    print((ys+1)/2*np.log2(ps)+(1-ys)/2*np.log2(1-ps))
    return np.mean(Is)

def compute_E(ys,ps):
    return np.mean([100 if (ps[i] > 0.5)^(ys[i] == 1) else 0 for i in range(len(ps))])
# I = compute_I(y,np.exp(lp),y)
# E = compute_E(y,np.exp(lp))
# print(I,E)
# table1 = WR_table('/home/rzhang/PycharmProjects/WGPC/res/WD_GPC/sigma_new_1.csv', 'r')
# table2 = WR_table('/home/rzhang/PycharmProjects/WGPC/res/WD_GPC/sigma_new_-1.csv', 'r')
# x = [i * 0.001 - 5 for i in range(10000)]
# y = [0.4 + 0.001 * i for i in range(4601)]
# f1 = interpolate.interp2d(y, x, table1, kind='linear')
# f2 = interpolate.interp2d(y, x, table2, kind='linear')
#
# model = pyGPs.GPC()
# f1, f2 = interp_fs()
# model.useInference('QP',f1,f2)
# model.optimize(x, y)
# n = z.shape[0]
# ymu, ys2, fmu, fs2, lp = model.predict(z, ys=np.ones((n,1)))
# model.plot(x1,x2,t1,t2)
# # Analogously to GPR
# k = pyGPs.cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
# model.setPrior(kernel=k)
#
# model.getPosterior(x, y)
# print("Negative log marginal liklihood before:", round(model.nlZ,7))
# model.optimize(x, y)
# print("Negative log marginal liklihood optimized:", round(model.nlZ,7))
#
# # Prediction

# print(np.exp(lp))
# I = np.mean(lp)
# E = np.mean([0 if np.exp(e)>0.5 else 1 for e in lp])
# print(I,E)

#
# print('--------------------END OF DEMO-----------------------')




