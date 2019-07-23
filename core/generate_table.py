from quantile import *
import numpy as np
from pynverse import inversefunc
import scipy.integrate as integrate
from scipy.special import erfinv
import csv

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

if __name__ == '__main__':
    # generate_table(1)
    table = [[1.0002,2,3],[2.1,3.1,4.1]]
    with open('sigma_{}.csv'.format(1),'w') as wf:
        writer = csv.writer(wf)
        writer.writerows(table)

    with open('sigma_{}.csv'.format(1),'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        print([[float(e) for e in l] for l in lines])