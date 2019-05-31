import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')

import numpy as np
from scipy.stats import norm
from scipy.special import owens_t
import scipy.integrate as integrate
import time
from scipy.special import erfinv
from pynverse import inversefunc
import matplotlib.pyplot as plt

def Fr(x,m,v,mu,sigma):
    Z = norm.cdf((mu-m)/v/np.sqrt(1+sigma**2/v**2))
    A = 1/Z
    k = (mu-m)/np.sqrt(sigma**2+v**2)
    h = (x-mu)/sigma
    rho = 1/np.sqrt(1+v**2/sigma**2)
    # print('Z: {}, \nA: {}, \nk: {}, \nh: {}, \nrho: {}'.format(Z,A,k,h,rho))

    eta = 0 if h*k>0 or (h*k==0 and h+k >= 0) else -0.5
    # eta = -0.5+((h * k > 0) + (h * k == 0)*(h + k >= 0))*0.5
    # res = np.zeros(len(x))
    # if k == 0 and 0 in h:
    #     res += (h==0)*A * (0.25 + 1 / np.sin(-rho))
    if k == 0 and h ==0:
        return A*(0.25+1/np.sin(-rho))
    # OT1 = owens_t(h,(k+rho*h)/h/np.sqrt(1-rho**2))
    # OT2 = owens_t(k,(h+rho*k)/k/np.sqrt(1-rho**2))
    OT1 = my_owens_t(h,k,rho)
    OT2 = my_owens_t(k,h,rho)
    if v>0:
        return A*(0.5*norm.cdf(h)+0.5*norm.cdf(k)-OT1-OT2+eta)
    if v<0:
        return A * (0.5 * norm.cdf(h) - 0.5 * norm.cdf(k) + OT1 + OT2 - eta)

def pr(x,m,v,mu,sigma):
    Z = norm.cdf((mu - m) / v / np.sqrt(1 + sigma ** 2 / v ** 2))
    return norm.cdf((x-m)/v)*norm.pdf(x,loc=mu,scale=sigma)/Z

def my_owens_t(x1,x2,rho):
    if x1 == 0 and x2>0:
        return 0.25
    elif x1 == 0 and x2<0:
        return -0.25
    else:
        return owens_t(x1,(x2+rho*x1)/x1/np.sqrt(1-rho**2))

def Fr_MC(x,m,v,mu,sigma):
    Z = norm.cdf((mu - m) / v / np.sqrt(1 + sigma ** 2 / v ** 2))
    samples = np.linspace(-50,x,100000)
    values = norm.cdf((samples-m)/v)*norm.pdf(samples, loc=mu,scale = sigma)
    d = samples[1]-samples[0]
    integral = np.sum(values[:-1]+values[1:])/2*d
    return integral/Z

def fit_gauss_wd(m,v,mu,sigma):
    z = (mu-m)/v/np.sqrt(1+sigma**2/v**2)
    inf_mu = mu+sigma**2*norm.pdf(z)/norm.cdf(z)/v/np.sqrt(1+sigma**2/v**2)
    inverse_Fr = lambda y: inversefunc(lambda x: Fr(x,m,v,mu,sigma),y_values=y,domain=[inf_mu-8*sigma, inf_mu+8*sigma],accuracy=14)
    C2 = np.sqrt(2)*integrate.quad(lambda x: erfinv(2*x-1)*inverse_Fr(x),0,1)[0]
    inf_sigma = C2
    return inf_mu,inf_sigma

def fit_gauss_kl(m,v,mu,sigma):
    sigma2 = sigma**2
    v2 = v**2
    z = (mu-m)/v/np.sqrt(1+sigma2/v2)
    inf_mu = mu+sigma2*norm.pdf(z)/norm.cdf(z)/v/np.sqrt(1+sigma2/v2)
    inf_sigma2 = sigma2-sigma2**2*norm.pdf(z)/(v2+sigma2)/norm.cdf(z)*(z+norm.pdf(z)/norm.cdf(z))
    return inf_mu,np.sqrt(inf_sigma2)

if __name__ == '__main__':
    m,v,mu,sigma = 1,2,3,40
    inf_mu_wd,inf_sigma_wd = fit_gauss_wd(m,v,mu,sigma)
    inf_mu_kl, inf_sigma_kl = fit_gauss_kl(m, v, mu, sigma)
    print('wd: mu ',inf_mu_wd,' sigma:',inf_sigma_wd)
    print('kl: mu ', inf_mu_kl, ' sigma:', inf_sigma_kl)
    xplot = np.linspace(mu-5,mu+5,100)
    yplot_true = pr(xplot,m,v,mu,sigma)
    yplot_wd = norm.pdf(xplot,loc=inf_mu_wd,scale=inf_sigma_wd)
    yplot_kl = norm.pdf(xplot, loc=inf_mu_kl, scale=inf_sigma_kl)
    plt.plot(xplot,yplot_true,label='True')
    plt.plot(xplot, yplot_wd, label='wd')
    plt.plot(xplot, yplot_kl, label='kl')
    plt.legend()
    plt.show()

    xplot = np.linspace(mu - 5, mu + 10, 100)
    yplot_true = np.array([Fr(e, m, v, mu, sigma) for e in xplot])
    yplot_wd = norm.cdf(xplot, loc=inf_mu_wd, scale=inf_sigma_wd)
    yplot_kl = norm.cdf(xplot, loc=inf_mu_kl, scale=inf_sigma_kl)
    plt.plot(xplot, yplot_true, label='True')
    plt.plot(xplot, yplot_wd, label='wd')
    plt.plot(xplot, yplot_kl, label='kl')
    plt.legend()
    plt.show()

