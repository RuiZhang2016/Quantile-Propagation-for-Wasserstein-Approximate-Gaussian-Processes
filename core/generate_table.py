from quantile import *
import numpy as np
from pynverse import inversefunc
import scipy.integrate as integrate
from scipy.special import erfinv
import csv
import os
from joblib import Parallel, delayed

def generate_table_QP(v):
    mu_range = np.linspace(-5,4.999,10000)
    sigma_range = np.linspace(0.4, 5, int((5-0.4)/0.001+1))

    print(mu_range[:10],mu_range[-10:],sigma_range[:10],sigma_range[-10:])
    table = []

    sqrt2 = np.sqrt(2)
    def loop(v,mu,sigma):
        inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y, accuracy=8)
        inf_sigma = sqrt2 * integrate.quad(lambda x: inverse_Fr(x) * erfinv(2 * x - 1), 0, 1)[0]
        return inf_sigma

    table = Parallel(n_jobs=40)(delayed(loop)(v,mu,sigma) for mu in mu_range for sigma in sigma_range)
    table = np.array(table).reshape((len(mu_range),len(sigma_range)))
    with open('sigma_{}.csv'.format(v),'w') as wf:
        writer = csv.writer(wf)
        writer.writerows(table)

def generate_table_EP(v):
    mu_range = np.linspace(-5,4.999,10000)
    sigma_range = np.linspace(0.4, 5, int((5-0.4)/0.001+1))

    def loop(v,mu,sigmas):
        res =[]
        for sigma in sigmas:
            _,inf_sigma = fit_gauss_kl(v,mu,sigma)
            res.append(inf_sigma)
        return res

    table = Parallel(n_jobs=40)(delayed(loop)(v,mu,sigma_range) for mu in mu_range)
    #table = np.array(table).reshape((len(mu_range),len(sigma_range)))
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
    table = WR_table('/home/rzhang/PycharmProjects/WGPC/tmp/sigma_-10.00_{}.csv'.format(v),'r')
    for i in range(2,401):
        sv = i/20-10.05
        filename = '/home/rzhang/PycharmProjects/WGPC/tmp/sigma_{:.2f}_{}.csv'.format(sv,v)
        if os.path.exists(filename):
            z = WR_table(filename,'r')
            table = np.vstack((table, z))
            assert WR_table('../res/WD_GPC/sigma_{}.csv'.format(v),'w',table)
        else:
            raise Exception(filename, ' not exists')
    print(table.shape)

def plot_table(v):
    plt.figure(figsize=(12,10))
    table = WR_table('../res/WD_GPC/sigma_{}.csv'.format(v),'r')
    table = table[200:]
    plt.matshow(table)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 10))
    table2 = WR_table('../res/WD_GPC/sigma_{}_ep.csv'.format(v), 'r')
    table2 = table2[200:][:-1]
    plt.matshow(table2)
    plt.colorbar()
    plt.show()

    assert table.shape == table2.shape

    plt.figure(figsize=(12, 10))
    diff = np.abs(table-table2)
    plt.matshow(diff)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # compress(-1)
    # plot_table(1)
    generate_table_EP(1)


    # # generate_table(1)
    # table = [[1.0002,2,3],[2.1,3.1,4.1]]
    # with open('sigma_{}.csv'.format(1),'w') as wf:
    #     writer = csv.writer(wf)
    #     writer.writerows(table)
    #
    # with open('sigma_{}.csv'.format(1),'r') as rf:
    #     reader = csv.reader(rf)
    #     lines = list(reader)
    #     print([[float(e) for e in l] for l in lines])
