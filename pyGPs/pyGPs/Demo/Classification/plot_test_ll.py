import sys
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

import numpy as np
import pyGPs
import matplotlib.pyplot as plt
from read_data import *
from GPC_experiments import *
# from . import __init__

def plot(x_train, y_train, x_test, y_test, f1, f2):
    n_features = x_train.shape[1]
    n_test = len(x_test)
    xmean = np.mean(x_train, axis=0)
    xstd = np.std(x_train, axis=0)

    # define models

    # model.useInference('QP',f1,f2)
    p1 = np.linspace(-1,4,6)
    p2 = np.linspace(-1,4,6)
    posts = []
    posts2 = []
    for p1e in p1:
        for p2e in p2:
            model = pyGPs.GPC()
            # model.useInference('QP', f1, f2)
            k = pyGPs.cov.RBFard(log_ell_list=[p2e] * n_features, log_sigma=p1e)  # kernel
            model.setPrior(kernel=k)
            model.getPosterior(x_train,y_train,False)
            posts += [-model.nlZ]

            model = pyGPs.GPC()
            # model.useInference('QP', f1, f2)
            k = pyGPs.cov.RBFard(log_ell_list=[p2e] * n_features, log_sigma=p1e)  # kernel
            model.setPrior(kernel=k)
            model.getPosterior(x_test, y_test, False)
            posts2 += [-model.nlZ]
    posts = np.array(posts).reshape((len(p2),len(p1)))
    posts2 = np.array(posts2).reshape((len(p2), len(p1)))
    X,Y = np.meshgrid(p2,p1)
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    axs[0].contourf(X,Y,posts)
    axs[1].contourf(X,Y,posts2)
    plt.show()


if __name__ == '__main__':
    dic = read_ionosphere()
    nr,nc = dic.shape
    train_ids = [i for i in range(200)]
    x_train = np.array([dic[i,:-1] for i in train_ids])
    y_train = np.array([[dic[i,-1]] for i in train_ids])
    x_test = np.array([dic[i, :-1] for i in range(nr) if i not in train_ids])
    y_test = np.array([dic[i, -1] for i in range(nr) if i not in train_ids])
    f1, f2 = None, None
    # f1, f2 =interp_fs()
    plot(x_train, y_train, x_test, y_test, f1, f2)





