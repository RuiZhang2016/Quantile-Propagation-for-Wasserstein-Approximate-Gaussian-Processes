from unittest import TestCase
import qp
import time
import numpy as np
from scipy.special import erfinv
import scipy.integrate as integrate

class TestQp(TestCase):
    def test_Fr(self):
        ms = [-2,-1,1,2]
        vs = [-2,-1,1,2]
        for x in np.linspace(-10,10,50):
            for m in ms:
                for v in vs:
                    # t = time.time()
                    res1 = qp.Fr(x, m, v, 0, 1)
                    # print(time.time() - t, ' seconds')
                    # t = time.time()
                    res2 = qp.Fr_MC(x, m, v, 0, 1)
                    # print(time.time() - t, ' seconds')
                    assert np.isclose(res1,res2),(x,' Fr, Fr_MC',res1,res2)

    def test_inverse_Fr(self):
        print('Test Inverse Func')
        from pynverse import inversefunc
        ms = [-2, -1, 1, 2]
        vs = [-2, -1, 1, 2]
        for m in ms:
            for v in vs:
                func = lambda x: qp.Fr(x, m, v, 0, 1)
                for x in np.linspace(0,1,10):
                    assert np.isclose(x,func(inversefunc(func, y_values=x)))

    def test_Fr_vectorisation(self):
        x = np.linspace(1,10,100)
        m, v, mu, sigma = 1, 2, 3, 4
        v = qp.Fr(x,m, v, mu, sigma)

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