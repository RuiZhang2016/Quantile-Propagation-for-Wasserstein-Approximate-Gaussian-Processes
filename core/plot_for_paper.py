import matplotlib
matplotlib.rcParams['text.usetex'] = True
from . import __init__

from core.quantile import *
import matplotlib.pyplot as plt
from scipy.stats import norm
from core.util import *
import pyGPs
os.environ['proj'] = "/home/rzhang/PycharmProjects/WGPC/"


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
    plt.savefig('../plots/components.pdf')
    # fig.show()

def calc_lks_is(f1,f2,exp_id):
    # import csv
    # file = os.environ['proj'] + '/data/ionosphere.data'
    #
    # def str2int(s):
    #     return 1 if s is 'g' else -1
    #
    # print('reading Ionosphere')
    #
    # with open(file, 'r') as rf:
    #     reader = csv.reader(rf)
    #     lines = list(reader)
    #     n = len(lines[0])
    #     z = np.array([[float(l[i]) if i < n - 1 else str2int(l[i]) for i in range(n) if i != 1] for l in lines])
    #
    # n_features = len(z[0])-1
    # print("#features: ", m - 1)
    # print("#data: ", n)
    # x_train = z[:200,:-1]
    # x_test = z[200:,:-1]
    # y_train = z[:200,-1]
    # y_test = z[200:, -1]
    # n_test = len(y_test)
    data_id, piece_id = divmod(exp_id, 10)
    datanames = ['ionosphere', 'crabs', 'breast_cancer', 'pima', 'usps', 'sonar']
    dic = load_obj('{}_{}'.format(datanames[data_id], piece_id))
    print('finish loading ', datanames[data_id])
    x_train = dic['x_train']
    x_test = dic['x_test']
    n_features = x_train.shape[1]
    n_test = len(x_test)
    xmean = np.mean(x_train, axis=0)
    xstd = np.std(x_train, axis=0)
    x_train = preproc(x_train, xmean, xstd)
    x_test = preproc(x_test, xmean, xstd)
    y_train = dic['y_train']
    y_test = dic['y_test']

    # define models
    # modelEP = pyGPs.GPC()
    # modelQP = pyGPs.GPC()
    # modelQP.useInference('QP', f1, f2)
    # modelQP.setOptimizer('CG')
    # k = pyGPs.cov.RBFard(log_ell_list=[np.log(n_features)/10] * n_features, log_sigma=1.)
    # modelQP.setPrior(kernel=k)

    # print('EP')
    # modelQP.optimize(x_train, y_train.reshape((-1, 1)), numIterations=40)
    # ymu, ys2, fmu, fs2, lp = modelQP.predict(x_test, ys=np.ones((n_test, 1)))
    # IEP = compute_I(y_test, np.exp(lp.flatten()), y_train)
    # EEP = compute_E(y_test, np.exp(lp.flatten()))
    # print(IEP, EEP)

    # modelQP.useInference('QP', f1, f2)
    logsigma_range = np.linspace(-1,5,21)
    logell_range = np.linspace(-1,5, 21)
    table_ep = []
    table_qp = []
    model = pyGPs.GPC()
    for logsigma in logsigma_range:
        row_ep = []
        row_qp = []
        for logell in logell_range:
            # modelQP.setOptimizer('CG')
            k =  pyGPs.cov.RBFard(log_ell_list=[logell] * n_features, log_sigma=logsigma)
            model.setPrior(kernel=k)
            for rnd in range(2):
                try:
                    model.getPosterior(x_train,y_train.reshape((-1,1)))
                    nlZ = model.nlZ
                    ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=np.ones((n_test, 1)))
                    I = compute_I(y_test, np.exp(lp.flatten()), y_train)
                    E = compute_E(y_test, np.exp(lp.flatten()))
                except Exception as e:
                    print(e)
                    nlZ = -1000
                    I = -1000
                    E = -1000
                if rnd == 0:
                    row_ep += [(nlZ, I, E)]
                    model.useInference('QP', f1, f2)
                else:
                    row_qp+= [(nlZ, I, E)]
                    model.useInference('EP')
            print(logsigma, logell, row_ep[-1],row_qp[-1])
        table_ep += [row_ep]
        table_qp += [row_qp]

    np.save('../res/ll_I_E_EP_restrict_boundary.npy',np.array(table_ep))
    np.save('../res/ll_I_E_QP_restrict_boundary.npy', np.array(table_qp))

def plot_lk_Is(filename):
    table = np.load(filename)
    lks = [[-e[0] for e in row] for row in table]
    for row in lks:
        print(row)
    Is = [[e[1] for e in row] for row in table]
    Es = [[e[2] for e in row] for row in table]
    logsigma_range = np.linspace(-1, 5, 21)
    logell_range = np.linspace(-1, 5, 21)
    lks = interpolate.interp2d(logell_range, logsigma_range, lks, kind='linear')
    Is = interpolate.interp2d(logell_range, logsigma_range, Is, kind='linear')
    X, Y = np.meshgrid(logell_range, logsigma_range)

    plt.figure(figsize=(10, 7))
    logsigma_range = np.linspace(-1, 5, 21*3)
    logell_range = np.linspace(-1, 5, 21*3)
    X, Y = np.meshgrid(logell_range, logsigma_range)
    cp = plt.contour(X, Y, lks(logell_range, logsigma_range), levels=[-200, -150, -120, -100]) # lks(logell_range, logsigma_range)
    plt.clabel(cp, cp.levels, inline=True, inline_spacing=0.5, fontsize=24, fmt='%1.1f')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.show()
    plt.figure(figsize=(10, 7))
    cp = plt.contour(X, Y, Is(logell_range, logsigma_range),levels=[0.2, 0.4, 0.6, 0.8,1.0]) # Is(logell_range, logsigma_range)
    plt.clabel(cp, cp.levels, inline=True, inline_spacing=0.5, fontsize=24, fmt='%1.1f')
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.show()
    # plt.figure()
    # CS = plt.contour(X, Y, Es)
    # plt.show()


if __name__ == '__main__':
    # plot_components()
    # f1,f2=interp_fs()
    # calc_lks_is(f1, f2, 1)
    plot_lk_Is('../res/ll_I_E_EP_restrict_boundary.npy')
    plot_lk_Is('../res/ll_I_E_QP_restrict_boundary.npy')
