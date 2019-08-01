from core.quantile import *
import numpy as np
from pynverse import inversefunc
import scipy.integrate as integrate
from scipy.special import erfinv
import csv
import os
from joblib import Parallel, delayed

def generate_table_QP(v):
    mu_range = np.linspace(-10,10,2001)
    sigma_range = np.linspace(0.4, 5, int((5-0.4)/0.01+1))

    print(mu_range[:10],mu_range[-10:],sigma_range[:10],sigma_range[-10:])
    table = []

    sqrt2 = np.sqrt(2)
    def loop(v,mu,sigma):
        inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y, accuracy=8)
        inf_sigma = sqrt2 * integrate.quad(lambda x: inverse_Fr(x) * erfinv(2 * x - 1), 0, 1)[0]
        return inf_sigma

    for mu in mu_range:
        row = []
        for sigma in sigma_range:
            inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y, accuracy=8)
            inf_sigma = np.sqrt(2)*integrate.quad(lambda x: inverse_Fr(x)*erfinv(2*x-1), 1e-14, 1-1e-14)[0]
            # wd2 = integrate.quad(lambda x: (inverse_Fr(x) - inf_mu - np.sqrt(2) * erfinv(2 * x - 1)) ** 2, 0, 1)[0]
            # inf_sigma = (sigma_q2 + 1 - wd2) / 2
            # inf_sigma = fit_gauss_wd_minus_wd(v,mu,sigma)
            row.append(inf_sigma)
            print('mu, sigma:',mu,sigma)
        table.append(row)
    with open('sigma_{}.csv'.format(v),'w') as wf:
        writer = csv.writer(wf)
        writer.writerows(table)

def generate_table_EP(v):
    mu_range = np.linspace(-5,4.999,10000)
    sigma_range = np.linspace(0.4, 5, int((5-0.4)/0.001+1))

    def loop(v,mu,sigmas):
        res = []
        for sigma in sigmas:
            _,inf_sigma = fit_gauss_kl(v,mu,sigma)
            res.append(inf_sigma)
        return res

    table = Parallel(n_jobs=7)(delayed(loop)(v,mu,sigma_range) for mu in mu_range)
    # table = np.array(table).reshape((len(mu_range),len(sigma_range)))
    WR_table('../res/WD_GPC/sigma_{}_ep.csv'.format(v),'w',table)

def WR_table(file,op,table=None):
    if op == 'r':
        with open(file,'r') as rf:
            reader = csv.reader(rf)
            lines = list(reader)
            z = [[float(e) for e in l] for l in lines]
            return np.array(z)
    elif op == 'w':
        with open(file, 'w') as wf:
            writer = csv.writer(wf)
            writer.writerows(table)
            return True
    else:
        raise NotImplementedError


def compress(v):
    table = WR_table('/Users/ruizhang/PycharmProjects/tmp/sigma_-5.000_{}.csv'.format(v),'r')
    for i in range(1,2000):
        sv = i*0.005-5
        filename = '/Users/ruizhang/PycharmProjects/tmp/sigma_{:.3f}_{}.csv'.format(sv,v)
        if os.path.exists(filename):
            z = WR_table(filename,'r')
            table = np.vstack((table, z))
        else:
            raise Exception(filename, ' not exists')
    assert WR_table('../res/WD_GPC/sigma_new_{}.csv'.format(v), 'w', table)
    print(table.shape)

def plot_table(v):
    plt.figure(figsize=(12,10))
    table = WR_table('../res/WD_GPC/sigma_new_{}.csv'.format(v),'r')
    # table = table[200:]
    plt.matshow(table)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 10))
    table2 = WR_table('../res/WD_GPC/sigma_{}_ep.csv'.format(v), 'r')
    # table2 = table2[200:][:-1]
    plt.matshow(table2)
    plt.colorbar()
    plt.show()

    assert table.shape == table2.shape, (table.shape,table2.shape)

    plt.figure(figsize=(12, 10))
    diff = np.abs(table-table2)
    plt.matshow(diff)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # generate_table_EP(1)
    # table = [[1.0002,2,3],[2.1,3.1,4.1]]
    # with open('sigma_{}.csv'.format(1),'w') as wf:
    #     writer = csv.writer(wf)
    #     writer.writerows(table)
    #
    # with open('sigma_{}.csv'.format(1),'r') as rf:
    #     reader = csv.reader(rf)
    #     lines = list(reader)
    #     print([[float(e) for e in l] for l in lines])
