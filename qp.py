import numpy as np
from scipy.stats import norm
from scipy.special import owens_t
import time

def Fr(x,m,v,mu,sigma):
    Z = norm.cdf((mu-m)/v/np.sqrt(1+sigma**2/v**2))
    A = 1/Z
    k = (mu-m)/v/np.sqrt(sigma**2/v**2+1)
    h = (x-mu)/sigma
    rho = 1/np.sqrt(1+v**2/sigma**2)
    # print('Z: {}, \nA: {}, \nk: {}, \nh: {}, \nrho: {}'.format(Z,A,k,h,rho))
    eta = 0 if h*k>0 or (h*k==0 and h+k >= 0) else -0.5
    # print('cdf_h: {}, \ncdf_k: {}, \nowt(h,k): {}, \nowt(k,h): {}\neta: {}'.format(norm.cdf(h),norm.cdf(k),owens_t(h,(k-rho*h)/h/np.sqrt(1-rho**2)),owens_t(k,(h-rho*k)/k/np.sqrt(1-rho**2)),eta))
    assert h != 0, 'h is 0'
    assert k != 0, 'k is 0'
    OT1 = owens_t(h,(k+rho*h)/h/np.sqrt(1-rho**2))
    OT2 = owens_t(k,(h+rho*k)/k/np.sqrt(1-rho**2))
    return A*(0.5*norm.cdf(h)+0.5*norm.cdf(k)-OT1-OT2+eta)

def pr(x,m,v,mu,sigma):

def Fr_MC(x,m,v,mu,sigma):
    Z = norm.cdf((mu - m) / v / np.sqrt(1 + sigma ** 2 / v ** 2))
    samples = np.linspace(-100,x,100000)
    values = norm.cdf((samples-m)/v)*norm.pdf(samples, loc=mu,scale = sigma)
    d = samples[1]-samples[0]
    integral = np.sum(values[:-1]+values[1:])/2*(samples[1]-samples[0])
    return integral/Z