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
        n = x_test.shape[0]
        ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=np.ones((n, 1)))

        # pyGPs.GPC.plot() is a toy method for 2-d data
        # plot log probability distribution for class +1

        # model.plot(x1, x2, y1, y2)


if __name__ == '__main__':
    experiments_ionosphere()