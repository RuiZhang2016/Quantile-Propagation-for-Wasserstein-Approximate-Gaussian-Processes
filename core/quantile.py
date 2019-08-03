import numpy as np
from scipy.stats import norm
from scipy.special import owens_t
import scipy.integrate as integrate
import time
from scipy.special import erfinv
from pynverse import inversefunc
import matplotlib.pyplot as plt
np.random.seed(0)

def Fr(x, v, mu, sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)

    k = mu / sqrtsigma
    Z = norm.cdf(k / v)  # Z = norm.cdf((mu - m) / v / np.sqrt(1 + sigma2 / v2))

    A = 1 / Z
    h = (x - mu) / sigma
    rho = sigma / sqrtsigma  # rho = 1 / np.sqrt(1 + v2 / sigma2)
    cdfk = norm.cdf(k)
    res = [0] * len(x) if np.ndim(x) else [0]
    hs = h if np.ndim(h) > 0 else [h]
    for i in range(len(hs)):
        h = hs[i]
        eta = 0 if h * k > 0 or (h * k == 0 and h + k >= 0) else -0.5
        if k == 0 and h == 0:
            res[i] = A * (0.25 + 1 / np.sin(-rho))
        # OT1 = owens_t(h,(k+rho*h)/h/np.sqrt(1-rho**2))
        # OT2 = owens_t(k,(h+rho*k)/k/np.sqrt(1-rho**2))
        OT1 = my_owens_t(h, k, rho)
        OT2 = my_owens_t(k, h, rho)
        res[i] = A * (0.5 * norm.cdf(h) + 0.5 * v * cdfk - v * OT1 - v * OT2 + v * eta)
    return res[0]

def pr(x,v,mu,sigma):
    Z = norm.cdf(mu  / v / np.sqrt(1 + sigma ** 2))
    return norm.cdf(x/v)*norm.pdf(x,loc=mu,scale=sigma)/Z

def my_owens_t(x1,x2,rho):
    if x1 == 0 and x2>0:
        return 0.25
    elif x1 == 0 and x2<0:
        return -0.25
    else:
        return owens_t(x1,(x2/x1+rho)/np.sqrt(1-rho**2))

def Fr_MC(x,m,v,mu,sigma):
    Z = norm.cdf((mu - m) / v / np.sqrt(1 + sigma ** 2 / v ** 2))
    samples = np.linspace(-50,x,100000)
    values = norm.cdf((samples-m)/v)*norm.pdf(samples, loc=mu,scale = sigma)
    d = samples[1]-samples[0]
    integral = np.sum(values[:-1]+values[1:])/2*d
    return integral/Z


def cal_C2(v,mu,sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)
    z = mu / v / sqrtsigma
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / sqrtsigma
    inf_sigma2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)
    inf_sigma = np.sqrt(inf_sigma2)
    xs_Fr = np.linspace(inf_mu-4*inf_sigma,inf_mu+4*inf_sigma,256)
    d = xs_Fr[1]-xs_Fr[0]
    ys = np.array([Fr(x,v,mu,sigma) for x in xs_Fr])
    # print(ys)
    # print(erfinv(2*ys-1))
    xs_erf = erfinv(2*ys-1)
    prod = np.array(xs_erf)*np.array(xs_Fr)
    prod = prod[~np.isnan(prod)]
    return np.sum(np.sqrt(2)*0.5*(prod[:-1]+prod[1:]))*d





xs_norm = np.random.normal(size=20000)
# xs_disc = np.linspace(-1,1,20000)
def fit_gauss_wd_sampling(v, mu, sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)
    z = mu / v / sqrtsigma  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    Z = norm.cdf(z)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf / v / sqrtsigma  # inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    #inf_sigma2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)
    #inf_sigma = np.sqrt(inf_sigma2)

    xs_Fr = xs_norm*sigma+mu# xs_disc*3*inf_sigma+inf_mu# np.linspace(inf_mu-4*inf_sigma,inf_mu+4*inf_sigma,8200) # xs_norm*sigma+mu

    ys = np.array([Fr(x, v, mu, sigma) for x in xs_Fr])
    # ys = np.array(Fr(xs_Fr,v,mu,sigma))
    ys_2 = 2 * ys - 1
    _nugget0 = -1 + 1e-14
    _nugget1 = 1 - 1e-14
    ys_2[ys_2 >= _nugget1] = _nugget1
    ys_2[ys_2 <= _nugget0] = _nugget0

    xs_erf = erfinv(ys_2)
    prod = xs_Fr*xs_erf*norm.cdf(xs_Fr*v)

    C2 = np.sqrt(2) * np.mean(prod)/Z
    return inf_mu, C2

samples = np.linspace(-1,1,1024)
def fit_gauss_wd_nature(v, mu, sigma):
    print('mu,sigma: ',mu,sigma)
    sigma2 = sigma ** 2
    # v2 = 1
    z = mu / v / np.sqrt(1 + sigma2)  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf/v/np.sqrt(1+sigma2)# inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)

    # inverse_Fr = lambda y: inversefunc(lambda x: self._Fr(x, v, mu, sigma), y_values=y, accuracy=6)
    # inf_sigma =  self.sqrt2* integrate.quad(lambda x: inverse_Fr(x)*erfinv(2 * x - 1), 0, 1)[0]

    inf_sigma2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)# inf_sigma2 = sigma2 - sigma2 ** 2 * norm.pdf(z) / (v2 + sigma2) / norm.cdf(z) * (z + norm.pdf(z) / norm.cdf(z))
    inf_sigma = np.sqrt(inf_sigma2)
    xs_Fr = samples*5*inf_sigma + inf_mu # np.linspace(inf_mu - 5 * inf_sigma, inf_mu + 5 * inf_sigma, 512)

    ys = np.array([Fr(x, v, mu, sigma) for x in xs_Fr])
    #ys = np.array(Fr(xs_Fr, v, mu, sigma))
    _nugget0 = -1+1e-14
    _nugget1 = 1 - 1e-14
    dys = ys[1:] - ys[:-1]
    ys = 2 * ys - 1
    ys[ys >= _nugget1] = _nugget1
    ys[ys <= _nugget0] = _nugget0
    xs_erf = erfinv(ys)
    prod = xs_Fr * xs_erf
    #
    inf_sigma = np.sqrt(2) * np.nansum((prod[:-1] + prod[1:]) * dys) * 0.5
    return inf_mu, inf_sigma

def fit_gauss_wd_minus_wd(v, mu, sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)
    z = mu / v / sqrtsigma  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    Z = norm.cdf(z)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf / v / sqrtsigma  # inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    sigma_q2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)
    inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y, accuracy=6)
    wd2 = integrate.quad(lambda x: (inverse_Fr(x)-inf_mu-np.sqrt(2)*erfinv(2*x-1)) ** 2, 0, 1)[0]
    inf_sigma = (sigma_q2+1-wd2)/2
    return inf_mu, inf_sigma

def fit_gauss_wd_integral(v, mu, sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)
    z = mu / v / sqrtsigma  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf / v / sqrtsigma  # inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y, accuracy=6)
    inf_sigma = np.sqrt(2)*integrate.quad(lambda x: inverse_Fr(x)*erfinv(2*x-1), 0, 1)[0]

    return inf_mu, inf_sigma


def fit_gauss_kl(v,mu,sigma):
    sigma2 = sigma**2
    z = mu/v/np.sqrt(1+sigma2)
    inf_mu = mu+sigma2*norm.pdf(z)/norm.cdf(z)/v/np.sqrt(1+sigma2)
    inf_sigma2 = sigma2-sigma2**2*norm.pdf(z)/(1+sigma2)/norm.cdf(z)*(z+norm.pdf(z)/norm.cdf(z))
    return inf_mu,np.sqrt(inf_sigma2)


if __name__ == '__main__':
    v,mu,sigma = 1,1,10
    t1 = time.time()
    inf_mu_wd,inf_sigma_wd = fit_gauss_wd_nature(v,mu,sigma)
    t2 = time.time()
    inf_mu_kl, inf_sigma_kl = fit_gauss_kl(v, mu, sigma)
    t3 = time.time()
    print('wd time, kl time: ',t2-t1,t3-t2)
    print('wd: mu ',inf_mu_wd,' sigma:',inf_sigma_wd)
    print('kl: mu ', inf_mu_kl, ' sigma:', inf_sigma_kl)
    xplot = np.linspace(mu-5,mu+5,100)
    yplot_true = pr(xplot,v,mu,sigma)
    wd_pdf = lambda x: norm.pdf(x,loc=inf_mu_wd,scale=inf_sigma_wd)
    kl_pdf = lambda x: norm.pdf(x, loc=inf_mu_kl, scale=inf_sigma_kl)
    inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y,accuracy=6)
    L2_wd = integrate.quad(lambda x: (inverse_Fr(x)-inf_mu_wd-inf_sigma_wd*np.sqrt(2)*erfinv(2*x-1)) ** 2, 0, 1)[0]
    L2_kl = integrate.quad(lambda x: (inverse_Fr(x)-inf_mu_kl-inf_sigma_kl*np.sqrt(2)*erfinv(2*x-1)) ** 2, 0, 1)[0]
    print('wd quantile L2, kl quantile L2: ', L2_wd, L2_kl)
    print('sigma_q^2-sigma^*2: ', inf_sigma_kl**2-inf_sigma_wd**2)

    t1 = time.time()
    inf_mu_wd, inf_sigma_wd_minus_wd = fit_gauss_wd_minus_wd(v, mu, sigma)
    t2 = time.time()
    inf_mu_wd, inf_sigma_wd_integral = fit_gauss_wd_integral(v, mu, sigma)
    t3 = time.time()
    print('wd minus wd time, wd integral time: ', t2 - t1, t3 - t2)
    print('sigma: minus wd ', inf_sigma_wd_minus_wd, ' integral: ', inf_sigma_wd_integral)

    L2_wd_minus_wd = \
    integrate.quad(lambda x: (inverse_Fr(x) - inf_mu_wd - inf_sigma_wd_minus_wd * np.sqrt(2) * erfinv(2 * x - 1)) ** 2, 0,
                   1)[0]
    L2_wd_integral = \
    integrate.quad(lambda x: (inverse_Fr(x) - inf_mu_kl - inf_sigma_wd_integral * np.sqrt(2) * erfinv(2 * x - 1)) ** 2, 0,
                   1)[0]
    print('wd quantile L2: minus wd, integral: ', L2_wd_minus_wd, L2_wd_integral)
    print('sigma_q^2-sigma^*2: minus wd, integral: ', inf_sigma_kl ** 2 - inf_sigma_wd_minus_wd ** 2,
          inf_sigma_kl ** 2 - inf_sigma_wd_integral ** 2)

    # plt.plot(xplot,yplot_true,label='True')
    # plt.plot(xplot, wd_pdf(xplot), label='wd')
    #
    # plt.plot(xplot, kl_pdf(xplot), label='kl')
    # plt.legend()
    # plt.show()
    #
    # xplot = np.linspace(mu - 5, mu + 10, 100)
    # yplot_true = np.array([Fr(e, v, mu, sigma) for e in xplot])
    # yplot_wd = norm.cdf(xplot, loc=inf_mu_wd, scale=inf_sigma_wd)
    # yplot_kl = norm.cdf(xplot, loc=inf_mu_kl, scale=inf_sigma_kl)
    # plt.plot(xplot, yplot_true, label='True')
    # plt.plot(xplot, yplot_wd, label='wd')
    # plt.plot(xplot, yplot_kl, label='kl')
    # plt.legend()
    # plt.show()

    # xplot = np.linspace(1e-6,1-1e-6,1024)
    # yplot = []
    # for x in xplot:
    #     yplot.append(inverse_Fr(x)*erfinv(2*x-1))
    #     print(yplot[-1])
    # plt.semilogy(xplot,yplot)
    # # plt.semilogy(xplot, erfinv(2*xplot-1))
    # plt.show()
