import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from quantile import *
from scipy.special import erfinv
import time
np.random.seed(100)
from joblib import Parallel, delayed

class IS:
    def __init__(self):
        self.J = 20000
        self.samples = np.sort(np.random.normal(size=self.J))
        self._nugget0 = -1 + 1e-14
        self._nugget1 = 1 - 1e-14
        self.sqrt2 = np.sqrt(2)

    def fit_gauss_kl_IS(self, v, mu, sigma):
        samples = self.samples*sigma+mu
        w = norm.cdf(v*samples)
        w /= np.sum(w)
        mean = w@samples
        inf_s = w@(samples**2)-mean**2
        return mean, np.sqrt(inf_s)

    def fit_gauss_wd2_IS(self, v, mu, sigma):
        samples = self.samples*sigma+mu
        w = norm.cdf(v*samples)
        w /= np.sum(w)
        mean = w@samples

        tmp = 2*np.cumsum(w)-1
        tmp[tmp >= self._nugget1] = self._nugget1
        tmp[tmp <= self._nugget0] = self._nugget0
        # inf_s = self.sqrt2*w@(samples*erfinv(tmp))
        sigma_q = w @ (samples ** 2) - mean ** 2
        sigma = sigma_q
        inf_s = (sigma_q**2-w@(samples - mean-self.sqrt2*sigma*erfinv(tmp))**2+sigma**2)/2/sigma
        return mean, inf_s


def main():
    # m,v = fit_gauss_kl_IS(1,1,2)
    # m1,v1 = fit_gauss_kl(1, 1, 2)
    # print(m,v,m1,v1)

    ## show abs error of importance sampling for kl estimate
    myIS = IS()
    inf_mat = []
    m_list = np.linspace(-5, 5, 30)
    s_list = np.linspace(0.3, 3, 30)
    for m in m_list:
        inf_vec = []
        for s in s_list:
            t0 = time.time(); inf_m1,inf_s1 = myIS.fit_gauss_kl_IS(1,m,s)
            t1 = time.time(); inf_m2,inf_s2 = fit_gauss_kl(1,m,s)
            t2 = time.time(); inf_vec += [[m,s,inf_m1,inf_s1,inf_m2,inf_s2,t1-t0,t2-t1]]
        inf_mat += [inf_vec]

    inf_mat = np.array(inf_mat)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    ids = [3,5]
    im = axes.flat[0].imshow(inf_mat[:,:,ids[0]], vmin=0, vmax=2)
    im = axes.flat[1].imshow(inf_mat[:,:,ids[1]],vmin=0, vmax=2)
    error = abs(inf_mat[:,:,ids[0]] - inf_mat[:,:,ids[1]])/abs(inf_mat[:,:,ids[1]])
    im = axes.flat[2].imshow(error,vmin=0, vmax=2)
    fig.colorbar(im, ax=axes.ravel().tolist())
    ticks_pos = [i*5 for i in range(6)]


    # x_label_list = ['{:.2f}'.format(s_list[i]) for i in ticks_pos]
    # y_label_list = ['{:.2f}'.format(m_list[i]) for i in ticks_pos]
    # axes.flat[0].set_xticks(ticks_pos)
    # axes.flat[0].set_yticks(ticks_pos)
    # axes.flat[0].set_xticklabels(x_label_list)
    # axes.flat[0].set_yticklabels(y_label_list)
    # fig,ax = plt.figure(figsize=(12,6))
    # plt.subplot(1,3,1);plt.imshow(inf_m1_mat)
    # plt.subplot(1, 3, 2);plt.imshow(inf_m2_mat)
    # plt.subplot(1, 3, 3);plt.imshow(np.array(inf_m2_mat) - np.array(inf_m1_mat))
    # plt.colorbar(ax=ax)
    plt.show()

def main2():
    ## importance sampling for wd estimation
    myIS = IS()
    m_list = np.linspace(-5, 5, 10)
    s_list = np.linspace(0.3, 3, 10)
    def loop(myIS,m,s_list):
        inf_vec = []
        for s in s_list:
            t0 = time.time(); inf_m1, inf_s1 = myIS.fit_gauss_wd2_IS(1, m, s)
            t1 = time.time(); inf_m2, inf_s2 = fit_gauss_wd2_quad(1, m, s)
            t2 = time.time(); inf_vec += [[m, s, inf_m1, inf_s1, inf_m2, inf_s2, t1 - t0, t2 - t1]]
        return inf_vec
    # inf_mat =[]
    # for m in m_list:
    #     inf_vec = []
    #     for s in s_list:
    #         t0 = time.time(); inf_m1, inf_s1 = myIS.fit_gauss_wd2_IS(1, m, s)
    #         t1 = time.time(); inf_m2, inf_s2 = fit_gauss_wd_sampling(1, m, s)
    #         t2 = time.time(); inf_vec += [[m, s, inf_m1, inf_s1, inf_m2, inf_s2, t1 - t0, t2 - t1]]
    #     inf_mat += [inf_vec]

    inf_mat = Parallel(n_jobs=4)(delayed(loop)(myIS,m,s_list) for m in m_list)
    inf_mat = np.concatenate(inf_mat)
    # print(inf_mat)
    # inf_mat = np.array(inf_mat)
    inf_mat = inf_mat.T.reshape((-1,len(m_list),len(s_list)))
    fig, axes = plt.subplots(nrows=1, ncols=3)
    ids = [3, 5]
    im = axes.flat[0].imshow(inf_mat[ids[0]], vmin=0, vmax=2)
    im = axes.flat[1].imshow(inf_mat[ids[1]], vmin=0, vmax=2)
    error = abs(inf_mat[ids[0]] - inf_mat[ids[1]])
    # im = axes.flat[0].imshow(inf_mat[:,:,ids[0]], vmin=0, vmax=2)
    # im = axes.flat[1].imshow(inf_mat[:,:,ids[1]], vmin=0, vmax=2)
    # error = abs(inf_mat[:,:,ids[0]] - inf_mat[:,:,ids[1]])/abs(inf_mat[:,:,ids[1]])
    im = axes.flat[2].imshow(error, vmin=0, vmax=2)
    fig.colorbar(im, ax=axes.ravel().tolist())
    ticks_pos = [i * 5 for i in range(6)]
    plt.show()

if __name__ == '__main__':
    main2()


