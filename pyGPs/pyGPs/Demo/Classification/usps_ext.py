from __future__ import print_function
import pickle,sys,os
# from . import __init__
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    sys.path.append('/Users/ruizhang/PycharmProjects/WGPC/pyGPs')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    sys.path.append('/home/rzhang/PycharmProjects/WGPC/pyGPs')
    os.environ['proj'] = '/home/rzhang/PycharmProjects/WGPC'
import pyGPs
import numpy as np

# np.random.seed(10230)
# from .read_data import *
from core.generate_table import *
from scipy import interpolate
from mpl_toolkits.mplot3d import axes3d

def preproc(x, m, s):
    return (x - m) / s


def compute_I(ys, ps, ys_train):
    p1 = np.mean([e if e == 1 else 0 for e in ys_train])
    p2 = 1 - p1
    H = -p1 * np.log2(p1) - p2 * np.log2(p2)
    assert ys.shape == ps.shape
    Is = (ys + 1) / 2 * np.log2(ps) + (1 - ys) / 2 * np.log2(1 - ps) + H
    return np.mean(Is)

def compute_testll(ys, ps):
    p1 = np.mean([e if e == 1 else 0 for e in ys])
    p2 = 1 - p1
    assert ys.shape == ps.shape
    Is = (ys + 1) / 2 * np.log2(ps) + (1 - ys) / 2 * np.log2(1 - ps)
    return np.mean(Is)

def compute_E(ys, ps):
    return np.mean([100 if (ps[i] > 0.5) ^ (ys[i] == 1) else 0 for i in range(len(ps))])

def read_usps():
    import h5py
    path = os.environ['proj'] + '/data/usps.h5'

    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    X,y= np.vstack((X_tr,X_te)), np.hstack((y_tr,y_te))
    X_split = [X[np.where(y == i)[0]] for i in range(10)]
    return X_split


def run(x_train,y_train,x_test,y_test,f1,f2,dataname,expid):

    # print(x_train[1:10, 1:10], np.std(x_train, axis=0), np.std(x_test, axis=0))
    n_features = x_train.shape[1]
    n_test = len(x_test)
    # xmean = np.mean(x_train, axis=0)
    # xstd = np.std(x_train, axis=0)
    # x_train = preproc(x_train, xmean, xstd)
    # x_test = preproc(x_test, xmean, xstd)


    # define models
    modelEP = pyGPs.GPC()
    modelQP = pyGPs.GPC()
    # modelEP.useLikelihood('Heaviside')
    # modelQP.useLikelihood('Heaviside')
    # modelEP.setOptimizer('BFGS')
    if not f1 is None and not f2 is None:
        modelQP.useInference('QP', f1, f2)
    kEP = pyGPs.cov.RBFard(log_ell_list=[0.01] * n_features, log_sigma=1.)  # kernel
    kQP = pyGPs.cov.RBFard(log_ell_list=[0.01] * n_features, log_sigma=1.)  # kernel
    modelEP.setPrior(kernel=kEP)
    modelQP.setPrior(kernel=kQP)

    # calculations of EP and QP
    models =  [modelEP, modelQP]
    Es = []
    Is = []
    lps = []
    for i in range(2):
        model = models[i]
        try:
            # model.getPosterior(x_train, y_train)
            model.optimize(x_train, y_train.reshape((-1, 1)), numIterations=1)
        except Exception as e:
            print(e)
            Is += [None]
            Es += [None]
            continue

        ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=y_test)
        lp = lp.flatten()
        print('lp:', np.exp(lp))
        y_test = y_test.flatten()
        lp2 = (1 + y_test) / 2 * lp + (1 - y_test) / 2 * (np.log(1 - np.exp(lp)))
        print('lp2:', np.exp(lp2))
        lps += [lp2]
        Is += [np.nansum(lp2)]
        # I = compute_I(y_test, np.exp(lp.flatten()), y_train)
        # pred = np.exp(lp.flatten())
        # y_test = y_test>0
        # y_score_bin_mean, empirical_prob_pos = model.reliability_curve(y_test>0,pred,bins=10)
        # scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        # line_style = '-' if i == 0 else '-.'
        # plt.plot(y_score_bin_mean[scores_not_nan],
        #          empirical_prob_pos[scores_not_nan],linestyle=line_style,label=model.inffunc.name)
        Es += [compute_E(y_test, np.exp(lp))]

    # plt.plot(np.linspace(0,1,20),np.linspace(0,1,20),'-.')
    # plt.xlabel('Predictive Probability')
    # plt.ylabel('Empirical Probability')
    # plt.legend()
    # plt.savefig(os.environ['proj'] + "/res/{}_rely_diag_{}.pdf".format(dataname,expid))
    # plt.show()

    # print results
    np.save(os.environ['proj'] + "/res/lps_{}_2.npy".format(expid), lps)
    f = open(os.environ['proj'] + "/res/{}_output_2.txt".format(dataname), "a")
    f.write('{} Es: EP {} QP {}; Is: EP {} QP {} \n'.format(expid, Es[0], Es[1], Is[0], Is[1]))
    f.close()


def read_output_table(file_path):
    with open(file_path,'r') as file:
        lines = file.readlines()
        lines = np.array([l.replace('\n','').split() for l in lines])
        lines = np.array([[float(l[-3]),float(l[-1])] for l in lines if 'Es:' in l])
        return lines

if __name__ == '__main__':
    X_split = read_usps()
    ns = [len(X_split[i]) for i in range(10)]
    f1, f2 = lambda x:x, lambda x:x
    for i in range(9):
        for j in range(i+1,10):
            y_ij = np.hstack((np.ones(len(X_split[i])),0-np.ones(len(X_split[j]))))
            X_ij = np.vstack((X_split[i],X_split[j]))
            n = len(X_ij)
            train_ids = np.random.choice(n,int(n*0.05),replace=False)
            X_train = np.array([X_ij[ii] for ii in train_ids])
            y_train = np.array([y_ij[ii] for ii in train_ids])
            X_test = np.array([X_ij[ii] for ii in range(n) if not ii in train_ids])
            y_test = np.array([y_ij[ii] for ii in range(n) if not ii in train_ids])
            run(X_train, y_train, X_test, y_test, f1, f2, 'usps', '{}-{}'.format(i,j))



