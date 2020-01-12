from scipy.special import roots_hermite,erfinv
from scipy.integrate import  quad
import numpy as np
import time
from scipy.stats import norm
from math import factorial


def main():
    x,w = roots_hermite(20)
    h = lambda x: (x**3+x+1)
    f = lambda x: np.exp(-x*x)*h(x)
    t1 = time.time()
    q1 = quad(f,-np.inf,np.inf)
    t2 = time.time()
    print(q1,t2-t1)
    q2 = h(x)@w
    print(q2,time.time()-t2)


def test_Z(y,mu,sigma):
    # like = lambda x: norm.cdf(y*x)
    like = lambda x: x**y*np.exp(-y)/factorial(y)
    link = lambda x: np.exp(x)
    Z = 0#norm.cdf(mu  / y / np.sqrt(1 + sigma ** 2))
    fZ = lambda x: like(x)*norm.pdf(x,mu,sigma)
    Z1 = quad(lambda x: fZ(link(x)),-np.inf,np.inf,epsrel=0)
    x, w = roots_hermite(20)
    Z2 = w@like(link(np.sqrt(2)*sigma*x+mu))*sigma*np.sqrt(2)
    print('mu, sigma:', mu,sigma,"\nZ analytical, quad, Gauss_Hermite:",Z,Z1,Z2)

def double_quad_variance(y,mu,sigma):
    like = lambda x: norm.cdf(y*x)
    pr = lambda x: norm.pdf(x,mu,sigma**2)
    Z = quad(lambda x: like(x)*pr(x),-np.inf,np.inf)[0]
    print('Z ', Z)
    post = lambda x: like(x)*pr(x)/Z
    F = lambda x: quad(post,-np.inf,x)[0]
    sigmastar = np.sqrt(2)*quad(lambda x: x*erfinv(2*F(x)-1)*post(x),-np.inf,np.inf)[0]
    print(mu,sigma,sigmastar)



if __name__ == '__main__':
    v = 1
    sigma = 2
    for mu in np.linspace(-100,100,100):
        # test_Z(v,mu,sigma)
        double_quad_variance(v,mu,sigma)