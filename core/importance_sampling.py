import numpy as np
from scipy.stats import norm
from quantile import fit_gauss_kl,fit_gauss_wd_sampling
from scipy.special import erfinv
import time


def fit_gauss_kl_IS(v, mu, sigma):
    J = 1000
    samples = np.random.normal(size=J)
    samples = samples*sigma+mu
    w = norm.cdf(v*samples)
    w /= np.sum(w)
    mean = w@samples

    variance = w@(samples**2)-mean**2
    return mean, np.sqrt(variance)


def fit_gauss_wd2_IS(v, mu, sigma):
    J = 20000
    samples = np.random.normal(size=J)
    samples = samples*sigma+mu
    samples = np.sort(samples)
    w = norm.cdf(v*samples)
    w /= np.sum(w)
    mean = w@samples
    tmp = 2*np.cumsum(w)-1
    _nugget0 = -1 + 1e-14
    _nugget1 = 1 - 1e-14
    tmp[tmp >= _nugget1] = _nugget1
    tmp[tmp <= _nugget0] = _nugget0
    variance = np.sqrt(2)*w@(samples*erfinv(tmp))
    return mean, variance


def main():
    # m,v = fit_gauss_kl_IS(1,1,2)
    # m1,v1 = fit_gauss_kl(1, 1, 2)
    # print(m,v,m1,v1)
    t0 = time.time()
    m,v = fit_gauss_wd2_IS(1,1,2)
    t1 = time.time()
    m1,v1 = fit_gauss_wd_sampling(1,1,2)
    t2 = time.time()
    print(m,v,m1,v1,t1-t0,t2-t1)

if __name__ == '__main__':
    main()

