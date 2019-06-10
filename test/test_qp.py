from unittest import TestCase
from core.quantile import *
import time
import numpy as np
from scipy.special import erfinv
import scipy.integrate as integrate
from pynverse import inversefunc
from scipy.stats import norm


class TestQp(TestCase):
    # def test_Fr(self):
    #     ms = [-2,-1,1,2]
    #     vs = [-2,-1,1,2]
    #     for x in np.linspace(-10,10,50):
    #         for m in ms:
    #             for v in vs:
    #                 # t = time.time()
    #                 res1 = Fr(x, m, v, 0, 1)
    #                 # print(time.time() - t, ' seconds')
    #                 # t = time.time()
    #                 res2 = Fr_MC(x, m, v, 0, 1)
    #                 # print(time.time() - t, ' seconds')
    #                 assert np.isclose(res1,res2),(x,' Fr, Fr_MC',res1,res2)
    #
    # def test_inverse_Fr(self):
    #     print('Test Inverse Func')
    #     ms = [-2, -1, 1, 2]
    #     vs = [-2, -1, 1, 2]
    #     for m in ms:
    #         for v in vs:
    #             func = lambda x: Fr(x, m, v, 0, 1)
    #             for x in np.linspace(0,1,10):
    #                 assert np.isclose(x,func(inversefunc(func, y_values=x)))
    #

    # def test_qp_fit_gauss2gauss(self):
    #     mus = np.linspace(-10,10,5)
    #     sigmas = np.linspace(1,10,5)
    #     for mu in mus:
    #         for sigma in sigmas:
    #             try:
    #                 inverse_Fr = lambda y: inversefunc(lambda x: norm.cdf(x, loc=mu, scale=sigma),
    #                                                    y_values=y,domain=[mu-4*sigma, mu+4*sigma],
    #                                                    accuracy=3)
    #                 inf_mu = integrate.quad(lambda x: inverse_Fr(x), 0, 1)[0]
    #                 C2 = np.sqrt(2) * integrate.quad(lambda x: erfinv(2 * x - 1) * inverse_Fr(x), 0, 1)[0]
    #                 inf_sigma = C2
    #                 # inf_sigma = cal_C2(0, v, mu, sigma)
    #                 assert np.isclose(inf_mu,mu),('inf_mu, mu:', inf_mu,mu)
    #                 assert np.isclose(inf_sigma, sigma), ('inf_sigma, sigma:', inf_sigma, sigma)
    #             except Exception as e:
    #                 print(e,mu,sigma)


    def test_qp_fit_gauss2gauss_MC(self):
        mus = np.linspace(-10,10,5)
        sigmas = np.linspace(1,10,5)
        for mu in mus:
            for sigma in sigmas:
                try:
                    xs_Fr = np.linspace(mu-5*sigma,mu+5*sigma,1024)
                    ys = np.array([norm.cdf(x,loc=mu,scale=sigma) for x in xs_Fr])
                    dys = ys[1:]-ys[:-1]
                    inf_mu = np.sum((xs_Fr[:-1]+xs_Fr[1:])*dys)*0.5
                    xs_erf = erfinv(2 * ys - 1)
                    prod = xs_Fr*xs_erf
                    prod = prod[~np.isnan(prod)]
                    C2 = np.sqrt(2)*np.sum((prod[:-1]+prod[1:])*dys)*0.5
                    inf_sigma = C2
                    # inf_sigma = cal_C2(0, v, mu, sigma)
                    assert np.isclose(inf_mu,mu,rtol=1e-4),('inf_mu, mu:', inf_mu,mu)
                    assert np.isclose(inf_sigma, sigma,rtol=1e-4), ('inf_sigma, sigma:', inf_sigma, sigma)
                except Exception as e:
                    print(e)
                    print(mu,inf_mu)
                    print(sigma, inf_sigma)

    # def test_fit_gauss_wd(self):
    #     ms = [-2, -1, 1, 2]
    #     vs = [-2, -1, 1, 2]
    #     mu, sigma = 1,1
    #     for m in ms:
    #         for v in vs:
    #             _,C2 = qp.fit_gauss_wd(m, v, mu, sigma)
    #             f2 = lambda x: np.sqrt(2)*erfinv(2*qp.Fr(x,m,v,mu,sigma)-1)\
    #                            *x*qp.pr(x,m,v,mu,sigma)
    #             xs = np.linspace(-10,10,100000)
    #             ys = np.array([f2(x) for x in xs])*(xs[1]-xs[0])/2
    #             C2_2 = np.sum(ys[:-1]+ys[1:])
    #             print(C2,C2_2)