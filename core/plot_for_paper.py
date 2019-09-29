import matplotlib
matplotlib.rcParams['text.usetex'] = True
from . import __init__

from core.quantile import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_components():
    x = np.linspace(-4,9,1024)
    m = 0
    s = 5
    v = 1
    likelihood = norm.cdf(x)
    prior = norm.pdf(x,loc=m, scale=s)
    posterior = pr(x,v,m,s)
    m_ep, s_ep = fit_gauss_kl(v, m, s)
    m_qc, s_qc = fit_gauss_wd_integral(v, m, s)
    q_ep = norm.pdf(x,m_ep,s_ep)
    q_qc = norm.pdf(x, m_qc, s_qc)
    # q_qc = norm.pdf(x, m_ep, s_ep)
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax2 = ax1.twinx()

    l1 = ax2.plot(x,likelihood, label='Likelihood p(y$|$f)')
    l2 = ax1.plot(x, prior, label='Prior p(f)')
    l3 = ax1.plot(x, posterior,'-.',label='Posterior p(f$|$y)')
    l4 = ax1.plot(x, q_ep, '--',label='EP q(f$|$y)')
    l5 = ax1.plot(x, q_qc, ':',label='QC q(f$|$y)')

    lns = l1 + l2 + l3 + l4 + l5
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=2,prop={'size': 20})

    fontsize = 28
    ax1.set_xlabel('f',fontsize=fontsize)
    ax1.set_ylabel('p(f$|$y)',fontsize=fontsize)
    ax2.set_ylabel('p(y$|$f)',fontsize=fontsize)
    ax1.xaxis.set_tick_params(labelsize=fontsize)
    ax1.yaxis.set_tick_params(labelsize=fontsize)
    ax2.yaxis.set_tick_params(labelsize=fontsize)

    # plt.legend()
    fig.tight_layout()
    # plt.savefig('../plots/components.pdf')
    fig.show()

if __name__ == '__main__':
    plot_components()