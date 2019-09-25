from __future__ import print_function
import pickle
import pyGPs
import numpy as np


np.random.seed(0)
# from .read_data import *
from core.generate_table import *
from scipy import interpolate


def preproc(x, m, s):
    return (x - m) / s


def compute_I(ys, ps, ys_train):
    p1 = np.mean([e if e == 1 else 0 for e in ys_train])
    p2 = 1 - p1
    H = -p1 * np.log2(p1) - p2 * np.log2(p2)
    assert ys.shape == ps.shape
    Is = (ys + 1) / 2 * np.log2(ps) + (1 - ys) / 2 * np.log2(1 - ps) + H
    return np.mean(Is)


def compute_E(ys, ps):
    return np.mean([100 if (ps[i] > 0.5) ^ (ys[i] == 1) else 0 for i in range(len(ps))])


def interp_fs():
    table1 = WR_table(os.environ['proj'] + '/res/WD_GPC/sigma_new_1.csv', 'r')
    table2 = WR_table(os.environ['proj'] + '/res/WD_GPC/sigma_new_-1.csv', 'r')
    x = [i * 0.001 - 5 for i in range(10000)]
    y = [0.4 + 0.001 * i for i in range(4601)]
    f1 = interpolate.interp2d(y, x, table1, kind='cubic')
    f2 = interpolate.interp2d(y, x, table2, kind='cubic')
    return f1, f2


def run(x_train, y_train, x_test, y_test, f1, f2, dataname, id):
    n_features = x_train.shape[1]
    n_test = len(x_test)
    xmean = np.mean(x_train, axis=0)
    xstd = np.std(x_train, axis=0)
    x_train = preproc(x_train, xmean, xstd)
    x_test = preproc(x_test, xmean, xstd)

    # define models
    model = pyGPs.GPC()
    k = pyGPs.cov.RBFard(log_ell_list=[2] * n_features, log_sigma=1.)  # kernel
    model.setPrior(kernel=k)

    # EP
    # model.getPosterior(x_train, y_train)
    model.optimize(x_train, y_train, numIterations=40)
    ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=np.ones((n_test, 1)))
    # IEP = compute_I(y_test, np.exp(lp.flatten()), y_train)
    EEP = compute_E(y_test, np.exp(lp.flatten()))

    print('EEP: {}'.format(EEP))
    # QP
    model.useInference('QP', f1, f2)
    # modelQP.getPosterior(x_train, y_train)
    model.optimize(x_train, y_train, numIterations=40)

    ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=np.ones((n_test, 1)))
    # IQP =  compute_I(y_test, np.exp(lp.flatten()), y_train)
    EQP = compute_E(y_test, np.exp(lp.flatten()))

    # print results
    print('EQP: {}'.format(EQP))
    # f = open(os.environ['proj']+"/res/{}_output.txt".format(dataname), "a")
    # f.write("Negative log marginal liklihood before and after optimization:\n")
    # f.write("EP: {}, {}\n".format(round(nlZEP1, 7), round(nlZEP2, 7)))
    # f.write("QP: {}, {}\n".format(round(nlZQP1, 7), round(nlZQP2, 7)))
    # f.write('{} I E: EP {} {} QP {} {}\n'.format(id, IEP, EEP, IQP, EQP))
    # f.close()


def experiments(f1, f2, exp_id):
    data_id, piece_id = divmod(exp_id, 10)
    datanames = ['ionosphere', 'breast_cancer', 'crabs', 'pima', 'usps', 'sonar']
    dic = load_obj('{}_{}'.format(datanames[data_id], piece_id))
    run(dic['x_train'], dic['y_train'], dic['x_test'], dic['y_test'], f1, f2, datanames[data_id], exp_id)


def synthetic(f1, f2):
    print('generating data ...')
    n = 100
    data_n1 = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], int(n / 2))
    data_n1 = np.array([np.append(e, -1) for e in data_n1])
    data_p1 = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], int(n / 2))
    data_p1 = np.array([np.append(e, 1) for e in data_p1])
    data = np.vstack((data_n1, data_p1))
    train_id = np.random.choice(len(data), int(n * 0.7))
    train = np.array([data[i] for i in train_id])
    test = np.array([data[i] for i in range(n) if i not in train_id])
    print('done')

    exp_id = 0
    dataname = 'synthetic'
    run(train[:, 0:-1], train[:, -1].reshape((-1,1)), test[:, 0:-1], test[:, -1].reshape((-1,1)), f1, f2, dataname, exp_id)


def load_obj(name):
    with open(os.environ['proj'] + '/data/split_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    f1, f2 = interp_fs()
    # f1, f2 = None, None
    # exp_id = int(sys.argv[1])
    experiments(f1,f2,20)
    # synthetic(f1, f2)
    # Parallel(n_jobs=4)(delayed(experiments)(0,0,exp_id) for exp_id in range(60))
    # print(exp_id)
    # experiments(0,0,exp_id)
