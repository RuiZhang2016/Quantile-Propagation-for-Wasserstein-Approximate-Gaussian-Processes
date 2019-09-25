from __future__ import print_function

import sys

import os
# for Mac OS
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    os.environ['proj'] = '/home/rzhang/PycharmProjects/WGPC'
sys.path.append(os.environ['proj']+'/pyGPs')
sys.path.append(os.environ['proj'])

import pyGPs
import numpy as np
np.random.seed(0)

from core.util import *


def run(x_train,y_train,x_test,y_test,f1,f2,dataname, id):
    # print(x_train[1:10, 1:10], np.std(x_train, axis=0), np.std(x_test, axis=0))
    n_features = x_train.shape[1]
    n_test = len(x_test)
    xmean = np.mean(x_train, axis=0)
    xstd = np.std(x_train, axis=0)
    x_train = preproc(x_train, xmean, xstd)
    x_test = preproc(x_test, xmean, xstd)

    # return#, y_train, y_test)
    # define models
    model = pyGPs.GPC()
    # modelQP = pyGPs.GPC()
    # modelQP.useInference('QP', f1, f2)
    # modelEP.setOptimizer('CG')
    k = pyGPs.cov.RBFard(log_ell_list=[3] * n_features, log_sigma=1.)  # kernel
    model.setPrior(kernel=k)
    # k = pyGPs.cov.RBFard(log_ell_list=[np.log(n_features)/10] * n_features, log_sigma=1.)
    # modelQP.setPrior(kernel=k)

    print('EP')
    model.optimize(x_train, y_train.reshape((-1,1)), numIterations=40)
    ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=np.ones((n_test,1)))
    IEP = compute_I(y_test, np.exp(lp.flatten()), y_train)
    EEP = compute_E(y_test, np.exp(lp.flatten()))
    print(IEP,EEP)

    print('QP')
    model.useInference('QP',f1,f2)
    try:
        model.optimize(x_train, y_train.reshape((-1,1)), numIterations=40)
        ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=np.ones((n_test,1)))
        IQP = compute_I(y_test, np.exp(lp.flatten()), y_train)
        EQP = compute_E(y_test, np.exp(lp.flatten()))
    except Exception as e:
        print(e)
        IQP = -1000
        EQP = -1000
    # print results
    # print("Negative log marginal liklihood before and after optimization")
    # print("EP: {}, {}".format(round(nlZEP1, 7), round(nlZEP2, 7)))
    # print("QP: {}, {}".format(round(nlZQP1, 7), round(nlZQP2, 7)))
    # print('I E: EP {} {} QP {} {}'.format(IEP, EEP, IQP, EQP))
    f = open(os.environ['proj']+"/res/{}_output.txt".format(dataname), "a")
    # f.write("Negative log marginal liklihood before and after optimization:\n")
    # f.write("EP: {}, {}\n".format(round(nlZEP1, 7), round(nlZEP2, 7)))
    # f.write("QP: {}, {}\n".format(round(nlZQP1, 7), round(nlZQP2, 7)))
    f.write('{} I E {} {} {} {}\n'.format(id, IEP, EEP, IQP, EQP))
    f.close()

def experiments(f1,f2,exp_id):
    data_id, piece_id = divmod(exp_id,10)
    datanames = ['ionosphere','crabs','breast_cancer','pima','usps','sonar']
    dic = load_obj('{}_{}'.format(datanames[data_id],piece_id))
    print('finish loading ',datanames[data_id])
    run(dic['x_train'],dic['y_train'],dic['x_test'],dic['y_test'],f1,f2,datanames[data_id],exp_id)

def read_output_table(filename):
    with open(filename,'r') as file:
        lines = file.readlines()
        lines = [[float(e) for e in l.split(' ')[3:]] for l in lines]
        return lines




if __name__ == '__main__':
    f1, f2 = interp_fs()
    # exp_id = int(sys.argv[1])
    for exp_id in range(10,20):
        experiments(f1,f2,exp_id)
        # print(exp_id)
        # experiments(0,0,exp_id)
    # lines = read_output_table('/home/rzhang/PycharmProjects/WGPC/res/sonar_output.txt')
    # for l in lines:
    #     print(l)
    # print('I E: ', np.mean(lines,axis=0))
