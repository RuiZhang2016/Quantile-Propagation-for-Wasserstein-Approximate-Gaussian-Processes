import os,sys

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    sys.path.append('/Users/ruizhang/PycharmProjects/WGPC/pyGPs')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    os.environ['proj'] = '/home/rzhang/PycharmProjects/WGPC'
sys.path.append(os.environ['proj']+'/pyGPs')
sys.path.append(os.environ['proj'])

from unittest import TestCase
from core.quantile import *
import time
import numpy as np
from scipy.special import erfinv, hyp1f1,gamma,gammaincc,comb,erf
import scipy.integrate as integrate
from pynverse import inversefunc
from scipy.stats import norm
import pyGPs
from math import factorial

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


    # def test_qp_fit_gauss2gauss_MC(self):
    #     mus = np.linspace(-10,10,5)
    #     sigmas = np.linspace(1,10,5)
    #     for mu in mus:
    #         for sigma in sigmas:
    #             try:
    #                 xs_Fr = np.linspace(mu-5*sigma,mu+5*sigma,1024)
    #                 ys = np.array([norm.cdf(x,loc=mu,scale=sigma) for x in xs_Fr])
    #                 dys = ys[1:]-ys[:-1]
    #                 inf_mu = np.sum((xs_Fr[:-1]+xs_Fr[1:])*dys)*0.5
    #                 xs_erf = erfinv(2 * ys - 1)
    #                 prod = xs_Fr*xs_erf
    #                 prod = prod[~np.isnan(prod)]
    #                 C2 = np.sqrt(2)*np.sum((prod[:-1]+prod[1:])*dys)*0.5
    #                 inf_sigma = C2
    #                 inf_sigma = cal_C2(0, v, mu, sigma)
                    # assert np.isclose(inf_mu,mu,rtol=1e-4),('inf_mu, mu:', inf_mu,mu)
                    # assert np.isclose(inf_sigma, sigma,rtol=1e-4), ('inf_sigma, sigma:', inf_sigma, sigma)
                # except Exception as e:
                #     print(e)
                #     print(mu,inf_mu)
                #     print(sigma, inf_sigma)

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

    # def test_heaviside_quantile(self):
    #     mu = 1
    #     sigma = 1
    #     xplot = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
    #     fig = plt.figure()
    #     for label in [-1,1]:
    #         cdf0 = norm.cdf(0,loc=mu,scale = sigma)
    #         Z = (1+label)/2-label*cdf0
    #         print('cdf(0), Z: ',norm.cdf(0,loc=mu,scale = sigma),Z)
    #         p = lambda x: norm.pdf(x,loc=mu,scale=sigma)*(2*(x>=0)-1 == label)/Z
    #         F = lambda x: norm.cdf(x,loc=mu,scale=sigma)/Z-(label+1)/2*cdf0/Z if 2*(x>=0)-1 == label else (1-label)/2
    #         qt = lambda y: mu + np.sqrt(2)*sigma*erfinv(2*(y*Z+(label+1)/2*cdf0)-1) if y > 0 else float('-inf')
    #         ax = fig.add_subplot(1,2,int((label+1)/2)+1)
    #         ys = [F(x) for x in xplot]
    #         ax.plot(xplot,ys,label='F')
    #         ax.plot(xplot,[p(x) for x in xplot],label='p')
    #         ax.legend()
    #         ax.set_title('label={}'.format( int((label + 1) / 2)))
    #         for i in range(len(xplot)):
    #             x_2 = qt(ys[i])
    #             print('x, value, x_2: ', xplot[i],ys[i],x_2)
    #
    #     plt.show()

    # def test_comp_EP_and_QP(self):
    #     modelEP = pyGPs.GPC()
    #     modelQP = pyGPs.GPC()
    #
    #     f1,f2 = lambda x:x, lambda x:x
    #     modelQP.useInference('QP', f1, f2)
    #     x = np.array([[1,2],[3,4],[5,6],[7,8]])
    #     nf = x.shape[1]
    #     y = np.array([[1],[1],[-1],[-1]])
    #     k1 = pyGPs.cov.RBFard(log_ell_list=[0.01] *nf, log_sigma=1.)  # kernel
    #     modelEP.setPrior(kernel=k1)
    #     k2 = pyGPs.cov.RBFard(log_ell_list=[0.01] * nf, log_sigma=1.)  # kernel
    #     modelQP.setPrior(kernel=k2)
    #     print('EP opt:')
    #     modelEP.optimize(x, y, numIterations=1)
    #     # print(modelEP.posterior)
    #     print('QP opt:')
    #     modelQP.optimize(x, y, numIterations=1)
    #     assert np.allclose(modelEP.inffunc.last_ttau,modelQP.inffunc.last_ttau),(modelEP.inffunc.last_ttau,modelQP.inffunc.last_ttau)
    #     assert np.allclose(modelEP.posterior.alpha, modelQP.posterior.alpha)
    #     assert np.allclose(modelEP.posterior.sW, modelQP.posterior.sW)
    #     assert np.allclose(modelEP.posterior.L,modelQP.posterior.L)

        # x_test = np.array([[2,3],[6,7]])
        # y_test = np.array([[1],[-1]])
        # ymu1, ys21, fmu1, fs21, lp1 = modelEP.predict(x_test, ys=np.ones(y_test.shape))
        # ymu2, ys22, fmu2, fs22, lp2 = modelQP.predict(x_test, ys=np.ones(y_test.shape))
        # assert ymu1==ymu2
        # print('ymu',ymu1,ymu2)
        # print('ys2:',ys21,ys22)
        # print('fmu:',fmu1,fmu2)
        # print('fs2: ',fs21,fs22)
        # print('lp:',lp1,lp2)
        # lp = lp.flatten()
        # y_test = y_test.flatten()
        # lp2 = (1 + y_test) / 2 * lp + (1 - y_test) / 2 * (np.log(1 - np.exp(lp)))
        # lps += [lp2]
        # Is += [np.nansum(lp2)]

    # def test_square_poisson_Z(self):
    #     g = lambda f: f**2
    #     like = lambda f,y: f**y*np.exp(-f)/factorial(y)
    #     pr = lambda x, mu, sigma: norm.pdf(x, loc=mu, scale=sigma)
    #     for y in [0,1,2,3,5,7]:
    #         for mu in np.linspace(-10,10,10):
    #             for sigma in np.linspace(1,5,10):
    #                 Z_quad = quad(lambda x: pr(x,mu,sigma)*like(g(x),y),-np.inf,np.inf)[0]
    #                 sigma2 = sigma*sigma
    #                 alpha = 2*sigma2/(1+2*sigma2)
    #                 h = mu*mu/(1+2*sigma2)
    #                 Z_thm = alpha**(y+0.5)/np.sqrt(2*np.pi*sigma2)/factorial(y)/np.exp(h)*gamma(y+0.5)*hyp1f1(-y,0.5,-h/2/sigma2)
    #                 assert np.isclose(Z_quad,Z_thm), "y, mu, sigma, Z_quad, Z_thm: {}, {}, {}, {}, {}".format(y,mu,sigma,Z_quad, Z_thm)

    def test_square_poisson_qp_sigma(self):
        g = lambda f: f**2
        like = lambda f,y: f**y*np.exp(-f)/factorial(y)
        pr = lambda x, mu, sigma: norm.pdf(x, loc=mu, scale=sigma)
        for U in [-3,-1,0,1,3]:
            for y in [0,2,4,6]:
                for mu in np.linspace(-3,3,5):
                    for sigma in np.linspace(0.5,3,5):
                        sigma2 = sigma*sigma
                        alpha = 2*sigma2/(1+2*sigma2)
                        h = mu*mu/(1+2*sigma2)
                        A = 1/(alpha**(y+0.5)*gamma(y+0.5)*hyp1f1(-y,0.5,-h/2/sigma2))
                        beta = mu/(1+2*sigma2)
                        Z_thm = alpha**(y+0.5)/np.sqrt(2*np.pi*sigma2)/factorial(y)/np.exp(h)*gamma(y+0.5)*hyp1f1(-y,0.5,-h/2/sigma2)
                        assert np.isclose(A, 1/(Z_thm*np.sqrt(2*np.pi*sigma2)*factorial(y)*np.exp(h))), "A has problems"
                        F_quad = lambda x: quad(lambda x: pr(x,mu,sigma)*like(g(x),y),-np.inf,x)[0]/Z_thm
                        F_thm = lambda x: A/2*np.sum([comb(2*y,k)*beta**(2*y-k)*alpha**((k+1)/2)*((-1)**k*gamma((k+1)/2)+
                        np.sign(x-beta)**(k+1)*gamma((k+1)/2)*(1-gammaincc((1+k)/2,(x-beta)**2/alpha))) for k in range(2*y+1)])
                        assert np.isclose(F_quad(U),F_thm(U)), "{} {} {} {}".format(U, y, mu, sigma)

    def test_simpler_computation_for_square_poisson(self):
        sigma = 0.8
        U = 1  # require >= 0
        k= 1

        f = lambda x: x**k*np.exp(-x*x/sigma)
        int_f = lambda U: 0.5*sigma**((1+k)/2)*gamma((1+k)/2)*((1+(-1)**k)-gammaincc((1+k)/2,U*U/sigma))
        int_f_quad = lambda U: quad(f,-np.inf,U)[0]
        assert np.isclose(int_f(U),int_f_quad(U))




