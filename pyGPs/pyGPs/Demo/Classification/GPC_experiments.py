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
from core.generate_table import *
from scipy import interpolate

def preproc(x, m, s):
    return (x-m)/s

def compute_I(ys, ps, ys_train):
    p1 = np.mean([e if e == 1 else 0 for e in ys_train])
    p2 = 1-p1
    H = -p1*np.log2(p1)-p2*np.log2(p2)
    assert len(ys) == len(ps)
    Is = (ys+1)/2*np.log2(ps)+(1-ys)/2*np.log2(1-ps)+H
    return np.mean(Is)

def compute_E(ys,ps):
    return np.mean([1 if (ps[i] > 0.5)^(ys[i] == 1) else 0 for i in range(len(ps))])

def interp_fs():
    table1 = WR_table('/home/rzhang/PycharmProjects/WGPC/res/WD_GPC/sigma_new_1.csv', 'r')
    table2 = WR_table('/home/rzhang/PycharmProjects/WGPC/res/WD_GPC/sigma_new_-1.csv', 'r')
    x = [i * 0.001 - 5 for i in range(10000)]
    y = [0.4 + 0.001 * i for i in range(4601)]
    f1 = interpolate.interp2d(y, x, table1, kind='cubic')
    f2 = interpolate.interp2d(y, x, table2, kind='cubic')
    return f1, f2

def run(x_train,y_train,x_test,y_test,f1,f2):
    n_features = x_train.shape[1]
    xmean = np.mean(x_train, axis=0)
    xstd = np.std(x_train, axis=0)
    x_train = preproc(x_train, xmean, xstd)
    x_test = preproc(x_test, xmean, xstd)

    # define models
    modelEP = pyGPs.GPC()
    modelQP = pyGPs.GPC()
    modelQP.useInference('QP', f1, f2)
    k = pyGPs.cov.RBFard(log_ell_list=[2] * n_features, log_sigma=1.)  # kernel
    modelEP.setPrior(kernel=k)
    modelQP.setPrior(kernel=k)

    # EP
    modelEP.getPosterior(x_train, y_train)
    nlZEP1 = modelEP.nlZ
    modelEP.optimize(x_train, y_train)
    nlZEP2 = modelEP.nlZ

    ymu, ys2, fmu, fs2, lp = modelEP.predict(x_test, ys=y_test.reshape((-1, 1)))
    IEP = compute_I(y_test, np.exp(lp), y_train)
    EEP = compute_E(y_test, np.exp(lp))

    # QP
    modelQP.getPosterior(x_train, y_train)
    nlZQP1 = modelQP.nlZ
    modelQP.optimize(x_train, y_train, numIterations=40)
    nlZQP2 = modelQP.nlZ

    ymu, ys2, fmu, fs2, lp = modelQP.predict(x_test, ys=y_test.reshape((-1, 1)))
    IQP = compute_I(y_test, np.exp(lp), y_train)
    EQP = compute_E(y_test, np.exp(lp))

    # print results
    print("Negative log marginal liklihood before and after optimization")
    print("EP: {}, {}".format(round(nlZEP1, 7), round(nlZEP2, 7)))
    print("QP: {}, {}".format(round(nlZQP1, 7), round(nlZQP2, 7)))
    print('I E: EP {} {} QP {} {}'.format(IEP, EEP, IQP, EQP))


def experiments_ionosphere():
    data = read_ionosphere()
    data = np.delete(data, 1, axis=1)
    np.random.shuffle(data)
    n, n_features = data.shape
    n_features -= 1
    l = int(n / 10)

    f1,f2 = interp_fs()

    for i in range(10):
        # split data
        test = data[i * l:i * l + l]
        train = np.vstack((data[:i * l], data[i * l + l:]))
        x_train = train[:,:-1]
        y_train = train[:, -1]
        x_test = test[:,:-1]
        y_test = test[:, -1]
        run(x_train,y_train,x_test,y_test,f1,f2)

def experiments_ionosphere():
    data = read_ionosphere()
    data = np.delete(data, 1, axis=1)
    np.random.shuffle(data)
    n= data.shape[0]
    l = int(n / 10)

    f1,f2 = interp_fs()

    for i in range(10):
        # split data
        test = data[i * l:i * l + l]
        train = np.vstack((data[:i * l], data[i * l + l:]))
        x_train = train[:,:-1]
        y_train = train[:, -1]
        x_test = test[:,:-1]
        y_test = test[:, -1]
        run(x_train,y_train,x_test,y_test,f1,f2)

def experiments_breast_cancer():
    data = read_breast_cancer()
    np.random.shuffle(data)
    n = data.shape[0]
    l = int(n / 10)

    f1, f2 = interp_fs()

    for i in range(10):
        # split data
        test = data[i * l:i * l + l]
        train = np.vstack((data[:i * l], data[i * l + l:]))
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]
        run(x_train, y_train, x_test, y_test, f1, f2)

if __name__ == '__main__':
    # experiments_ionosphere()
    experiments_breast_cancer()
