from __future__ import print_function

import sys
import copy

# for Mac OS
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    sys.path.append('/Users/ruizhang/PycharmProjects/Wasserstein-GPC/pyGPs')
else:
    sys.path.append('/home/rzhang/PycharmProjects/WGPC/pyGPs')


import pyGPs
import numpy as np
np.random.seed(0)
from read_data import *


def preproc(x, m, s):
    return (x-m)/s

def compute_Is(ys, ps, ys_train):
    p1 = np.mean([e if e == 1 else 0 for e in ys_train])
    p2 = 1-p1
    H = -p1*np.log2(p1)-p2*np.log2(p2)
    assert len(ys) == len(ps)
    n = len(ps)
    Is = (ys+1)/2*np.log2(ps)+(1-ys)/2*np.log2(1-ps)+H
    return Is

def compute_E(ys,ps):
    return np.mean([1 if (ps[i] > 0.5)^(ys[i] == 1) else 0 for i in range(len(ps))])

def experiments_ionosphere():
    data = read_ionosphere()
    np.random.shuffle(data)
    n = len(data)
    l = int(n / 10)
    for i in range(10):
        # split data
        test = data[i * l:i * l + l]
        train = np.vstack((data[:i * l], data[i * l + l:]))
        x_train = train[:,:-1]
        y_train = train[:, -1]
        x_test = test[:,:-1]
        y_test = test[:, -1]
        n_features = x_train.shape[1]

        #preprocess
        xmean = np.mean(x_train)
        xstd = np.std(x_train)
        x_train = preproc(x_train,xmean,xstd)
        x_test = preproc(x_test,xmean,xstd)

        data1 = data[data[:, -1] == 1]
        data2 = data[data[:, -1] == -1]
        x1 = preproc(data1[:, :-1],xmean,xstd)
        y1 = data1[:, -1]
        x2 = preproc(data2[:, :-1],xmean,xstd)
        y2 = data2[:, -1]

        model = pyGPs.GPC()
        # kernel
        k = pyGPs.cov.RBFard(log_ell_list=[0.5]*n_features, log_sigma=1.)
        model.setPrior(kernel=k)

        model.getPosterior(x_train, y_train)
        print("Negative log marginal liklihood before:", round(model.nlZ, 3))
        model.optimize(x_train, y_train)
        print("Negative log marginal liklihood optimized:", round(model.nlZ, 3))

        # Prediction
        ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=y_test.reshape((-1,1)))
        Is = compute_Is(y_test,np.exp(lp),y_train)
        E = compute_E(y_test,np.exp(lp))
        print('EP I E: {} {}'.format(np.mean(Is),E))

        print('new model using QP and initilized with the optimal hyp from EP')
        model.useInference('QP')

        model = pyGPs.GPC()
        # kernel
        k = pyGPs.cov.RBFard(log_ell_list=[0.5] * n_features, log_sigma=1.)
        model.setPrior(kernel=k)

        model.getPosterior(x_train, y_train)
        print("Negative log marginal liklihood before:", round(model.nlZ, 7))
        model.optimize(x_train, y_train, numIterations=6)
        print("Negative log marginal liklihood optimized:", round(model.nlZ, 7))

        ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=y_test.reshape((-1, 1)))
        Is = compute_Is(y_test, np.exp(lp), y_train)
        E = compute_E(y_test, np.exp(lp))
        print('QP I E: {} {}'.format(np.mean(Is), E))

if __name__ == '__main__':
    experiments_ionosphere()