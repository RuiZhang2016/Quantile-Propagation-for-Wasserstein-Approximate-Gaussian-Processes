from core.quantile import *
import numpy as np
from pynverse import inversefunc
import scipy.integrate as integrate
from scipy.special import erfinv
import csv
import os

def generate_table(v):
    mu_range = np.linspace(-2,2,401)
    sigma_range = np.linspace(0.51, 5, int((5-0.51)/0.01+1))

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



if __name__ == '__main__':
    compress(-1)
    # compress(-1)
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