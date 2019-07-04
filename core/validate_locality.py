import __init__
import matplotlib
matplotlib.use('Qt5Agg')

import quantile
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import numpy as np
from scipy.special import erfc
from scipy import integrate
import ot

def truth(x,y,v,k,a,tmu,tsigma):

    k_a = k**4-a**2
    tsigma2 = tsigma**2
    expp = -(k_a*(x-tmu)**2-2*a*x*y*tsigma2+k**2*(x**2+y**2)*tsigma2)/2/tsigma2/k_a
    return np.exp(expp)*erfc(-y/np.sqrt(2)/v)/4/np.sqrt(2*k_a)/np.pi/np.sqrt(np.pi)/tsigma

def gauss(x,K,tmus,tsigmas):
    Sigma =np.linalg.inv(np.linalg.inv(K)+np.diag(1/tsigmas))
    # print(Sigma,np.diag(1/tsigmas),tmus)
    mu = Sigma@np.diag(1/tsigmas)@tmus.T
    pr = multivariate_normal(mu.flatten(),np.diag(tsigmas))
    return pr.pdf(x)

def cavity_param(K,tmus,tsigmas):
    Sigma = np.linalg.inv(np.linalg.inv(K) + np.diag(1 / tsigmas))
    mu = Sigma @ np.diag(1 / tsigmas) @ tmus.T
    mu = mu.flatten()
    sigma_ni2 = 1/(1/Sigma[1,1]**2-1/tsigmas[1]**2)
    mu_ni = sigma_ni2*(1/Sigma[1,1]**2*mu[1]-1/tsigmas[1]**2*tmus[1])
    return mu_ni, sigma_ni2

if __name__ == '__main__':
    n = 100
    v = 1
    k = 1
    a = 0
    # ct = integrate.dblquad(lambda x,y: truth(x,y,v,k,a,tl_mu,tl_sigma),-10,10,lambda x:-10,lambda x:10)[0]
    # ys = truth(xdata, ydata, v, k, a, tl_mu, tl_sigma)/ct

    tmus = np.array([0, 0])
    tsigmas = np.array([1, 1])
    K = np.array([[1, 0.5], [0.5, 1]])
    mu_ni, sigma_ni2 = cavity_param(K, tmus, tsigmas)
    tl_mu, tl_sigma = quantile.fit_gauss_kl(0, v, mu_ni, np.sqrt(sigma_ni2))

    for mu in np.linspace(-1,1,11):
        xs = np.array([[i,j] for i in np.linspace(-5+mu,mu+5,n) for j in np.linspace(mu-5,mu+5,n)])
        xdata = xs[:, 0]
        ydata = xs[:, 1]

        ys = truth(xdata, ydata, v, k, a, tl_mu, tl_sigma)
        ys /= np.sum(ys)

        tmus = np.array([mu, tl_mu])
        tsigmas = np.array([1, tl_sigma])
        ys2 = gauss(xs, K, tmus, tsigmas)
        ys2 /= np.sum(ys2)

        M = ot.dist(xs, xs)

        G0 = ot.emd2(ys, ys2, M,numItermax=100000)
        print(mu,G0)
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xdata,ydata,ys)
    # # plt.show()
    #
    # tmu_star, tsigma_star = 0, 1
    # # ys = truth(xdata, ydata, v, k, a, tl_mu, tl_sigma)

    # ys = gauss(xs,K,tmus,tsigmas)
    # # ax = plt.axes(projection='3d')
    # ax.scatter3D(xdata, ydata, ys)
    #
    # plt.show()
    # # for angle in range(0, 360):
    # #     ax.view_init(30, angle)
    # #     plt.draw()
    # #     plt.pause(.001)