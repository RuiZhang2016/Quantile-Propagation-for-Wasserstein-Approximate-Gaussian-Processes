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
import ot.plot
from joblib import Parallel, delayed
# def truth(x,y,v,k,a,tmu,tsigma):
#     k_a = k**4-a**2
#     tsigma2 = tsigma**2
#     expp = -(k_a*(x-tmu)**2-2*a*x*y*tsigma2+k**2*(x**2+y**2)*tsigma2)/2/tsigma2/k_a
#     return np.exp(expp)*erfc(-y/np.sqrt(2)/v)/4/np.sqrt(2*k_a)/np.pi/np.sqrt(np.pi)/tsigma

def truth(x,K,tmus,tsigma2s):
    tsigmas = np.sqrt(tsigma2s)
    mn = multivariate_normal([0,0],K)
    return mn.pdf(x)*norm.pdf(x[:,0],loc=tmus[0],scale=tsigmas[0])*norm.pdf(x[:,1],loc=tmus[1],scale=tsigmas[1])

def gauss(x,K,tmus,tsigma2s):
    Sigma =np.linalg.inv(np.linalg.inv(K)+np.diag(1/tsigma2s))
    # print(Sigma,np.diag(1/tsigmas),tmus)
    mu = Sigma@np.diag(1/tsigma2s)@tmus.T
    pr = multivariate_normal(mu.flatten(),Sigma)
    return pr.pdf(x)

def cavity_param(K,tmus,tsigma2s):
    Sigma = np.linalg.inv(np.linalg.inv(K) + np.diag(1 / tsigma2s))
    mu = Sigma @ np.diag(1 / tsigma2s) @ tmus.T
    mu = mu.flatten()
    sigma_ni2 = 1/(1/Sigma[1,1]-1/tsigma2s[1])
    mu_ni = sigma_ni2*(1/Sigma[1,1]*mu[1]-1/tsigma2s[1]*tmus[1])
    return mu_ni, sigma_ni2

if __name__ == '__main__':
    n = 100
    l = n ** 2
    v = 1
    k = 1
    a = 0
    # ct = integrate.dblquad(lambda x,y: truth(x,y,v,k,a,tl_mu,tl_sigma),-10,10,lambda x:-10,lambda x:10)[0]
    # ys = truth(xdata, ydata, v, k, a, tl_mu, tl_sigma)/ct
    K = np.array([[1, 0.1], [0.1, 1]])

    for tmus0 in [-0.1,-0.05,0,0.05,0.1]:
        tmus = np.array([tmus0, 0])
        tsigma2s = np.array([1, 1])
        mu_ni, sigma_ni2 = cavity_param(K, tmus, tsigma2s)
        tl_mu, tl_sigma = quantile.fit_gauss_wd(0, v, mu_ni, np.sqrt(sigma_ni2))

        mus = [0.15,0.1,0.05,0,-0.05,-0.1,-0.15]
        sigmas = [1.15,1.1,1.05,1,0.95,0.9,0.85]
        def loop(mu,sigma):
            xs = np.array([[i,j] for i in np.linspace(-3+mu,mu+3,n) for j in np.linspace(mu-3,mu+3,n)])

            # ys = truth(xdata, ydata, v, k, a, tl_mu, tl_sigma)
            ys = truth(xs,K,tmus,tsigma2s)
            ys /= np.sum(ys)

            tmus_2 = np.array([mu, tl_mu])
            tsigma2s_2 = np.array([sigma, tl_sigma])
            ys2 = gauss(xs, K, tmus_2, tsigma2s_2)
            ys2 /= np.sum(ys2)


            M2 = np.sum([((xs[:, i]).reshape((l, 1)) - (xs[:, i]).reshape((1, l))) ** 2 for i in range(2)], axis=0)
            G0 = ot.emd2(ys, ys2, M2,numItermax=1000000)
            return G0

        # res = Parallel(n_jobs=2)(delayed(loop)(mu) for mu in mus)
        # print(mus,res)
        res = [[loop(mu,sigma) for sigma in sigmas] for mu in mus]
        plt.imshow(res)
        plt.yticks([i for i in range(len(mus))],mus)
        plt.xticks([i for i in range(len(sigmas))],sigmas)
        for i in range(len(mus)):
            for j in range(len(sigmas)):
                plt.text(j-0.5,i,'{0:.4f}'.format(res[i][j]))
        plt.xlabel('sigma')
        plt.ylabel('mu')
        plt.colorbar()
        plt.savefig('../plots/mu{}_sigma{}_K{}.pdf'.format(tmus[0],tsigma2s[0],K))
        plt.close('all')
        # plt.show()
    #
    # 1-d Gaussian
    # b = 1000
    # x = np.linspace(-5,5,n)
    # # Gaussian distributions
    # a = norm.pdf(x, loc=0, scale=1)  # m= mean, s= std
    # b = norm.pdf(x, loc=2, scale=1)
    # a /= np.sum(a)
    # b /= np.sum(b)
    # # loss matrix
    # M2 = x.reshape((n,1))-x.reshape((1,n))
    # M2 = M2**2
    # print(ot.emd2(a, b, M2,numItermax=100000))

    # 2-d Gaussian
    # n = 100
    # x = np.array([[i, j] for i in np.linspace(-5, 5, n) for j in np.linspace(-5, 5, n)])
    # mn = multivariate_normal(mean=[0,0],cov=[[1,0],[0,1]])
    # ys = np.array([mn.pdf(e) for e in x])
    # mn = multivariate_normal(mean=[0,0.5], cov=[[1,0], [0,1]])
    # ys2 = np.array([mn.pdf(e) for e in x])
    # ys /= np.sum(ys)
    # ys2 /= np.sum(ys2)
    # l = n**2
    # M2 = np.sum([((x[:,i]).reshape((l,1))-(x[:,i]).reshape((1,l)))**2 for i in range(2)],axis=0)
    # print(ot.emd2(ys, ys2, M2, numItermax=10000000))

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