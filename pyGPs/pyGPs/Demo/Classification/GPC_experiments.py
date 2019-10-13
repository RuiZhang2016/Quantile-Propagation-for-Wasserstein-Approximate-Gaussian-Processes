from __future__ import print_function
import pickle,sys,os
# from . import __init__
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    sys.path.append('/Users/ruizhang/PycharmProjects/WGPC/pyGPs')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    sys.path.append('/home/rzhang/PycharmProjects/WGPC/pyGPs')
import pyGPs
import numpy as np

# np.random.seed(10230)
# from .read_data import *
from core.generate_table import *
from scipy import interpolate
from mpl_toolkits.mplot3d import axes3d

def preproc(x, m, s):
    return (x - m) / s


def compute_I(ys, ps, ys_train):
    p1 = np.mean([e if e == 1 else 0 for e in ys_train])
    p2 = 1 - p1
    H = -p1 * np.log2(p1) - p2 * np.log2(p2)
    assert ys.shape == ps.shape
    Is = (ys + 1) / 2 * np.log2(ps) + (1 - ys) / 2 * np.log2(1 - ps) + H
    return np.mean(Is)


def compute_E(ys, ps):
    return np.mean([100 if (ps[i] > 0.5) ^ (ys[i] == 1) else 0 for i in range(len(ps))])


def interp_fs():
    table1 = WR_table(os.environ['proj'] + '/res/WD_GPC/sigma_new_1.csv', 'r')
    table2 = WR_table(os.environ['proj'] + '/res/WD_GPC/sigma_new_-1.csv', 'r')
    x = [i * 0.001 - 5 for i in range(10000)]
    y = [0.4 + 0.001 * i for i in range(4601)]
    f1 = interpolate.interp2d(y, x, table1, kind='cubic')
    f2 = interpolate.interp2d(y, x, table2, kind='cubic')
    return f1, f2


def experiments(f1, f2, exp_id):
    data_id, piece_id = divmod(exp_id, 10)
    datanames = ['ionosphere', 'breast_cancer', 'crabs', 'pima', 'usps', 'sonar']
    dic = load_obj('{}_{}'.format(datanames[data_id], piece_id))
    run(dic['x_train'], dic['y_train'], dic['x_test'], dic['y_test'], f1, f2, datanames[data_id], exp_id)


def run(x_train,y_train,x_test,y_test,f1,f2,dataname,expid):

    # print(x_train[1:10, 1:10], np.std(x_train, axis=0), np.std(x_test, axis=0))
    n_features = x_train.shape[1]
    n_test = len(x_test)
    # xmean = np.mean(x_train, axis=0)
    # xstd = np.std(x_train, axis=0)
    # x_train = preproc(x_train, xmean, xstd)
    # x_test = preproc(x_test, xmean, xstd)


    # define models
    modelEP = pyGPs.GPC()
    modelQP = pyGPs.GPC()
    modelEP.useLikelihood('Heaviside')
    modelQP.useLikelihood('Heaviside')
    if not f1 is None and not f2 is None:
        modelQP.useInference('QP', f1, f2)
    k = pyGPs.cov.RBFard(log_ell_list=[0.01] * n_features, log_sigma=1.)  # kernel
    print('kernel params: ', k.hyp)

    #setup plots
    fig = plt.figure()

    # ax = fig.gca(projection='3d')

    # calculations of EP and QP
    models =  [modelEP, modelQP]
    Es = []
    for i in range(2):
        model = models[i]
        model.setPrior(kernel=k)

        print('Inference Method: ',model.inffunc.name)
        print('Likelihood Function: ', model.likfunc)
        # model.getPosterior(x_train, y_train)
        model.optimize(x_train, y_train.reshape((-1,1)), numIterations=40)
        print('kernel params: ', model.covfunc.hyp)
        #model.getPosterior(x_train, y_train)
        print('negative log likelihood: ',model.nlZ)

        K = model.covfunc.getCovMatrix(x=x_train, mode='train')
        # print('K: ',K)
        from scipy.stats import multivariate_normal as norm

        tau_ni0 = 1 / model.inffunc.Sigma[0, 0] - model.inffunc.last_ttau[0]  # first find the cavity distribution ..
        nu_ni0 = model.inffunc.mu[0] / model.inffunc.Sigma[0, 0] - model.inffunc.last_tnu[0]
        tau_ni1 = 1 / model.inffunc.Sigma[1, 1] - model.inffunc.last_ttau[1]  # first find the cavity distribution ..
        nu_ni1 = model.inffunc.mu[1] / model.inffunc.Sigma[1, 1] - modelEP.inffunc.last_tnu[1]
        if isinstance(modelEP.likfunc,pyGPs.lik.Heaviside):
            true_pdf = lambda x: norm.pdf(x,mean=np.zeros(2),cov = K)*\
                             (((x[0]>=0)*2-1)==y_train[0,0])*(((x[1]>=0)*2-1)==y_train[1,0])*4
            cavity_f1 = lambda x: norm.pdf(x, mean=nu_ni0 / tau_ni0, cov=1 / tau_ni0) * (
                        ((x >= 0) * 2 - 1) == y_train[0, 0])
            cavity_f2 = lambda x: norm.pdf(x, mean=nu_ni1 / tau_ni1, cov=1 / tau_ni1) * (
                        ((x >= 0) * 2 - 1) == y_train[1, 0])
        elif isinstance(modelEP.likfunc,pyGPs.lik.Erf):
            true_pdf = lambda x: norm.pdf(x,mean=np.zeros(2),cov = K)*norm.cdf(x[0]*y_train[0,0])*norm.cdf(x[1]*y_train[1,0])
            cavity_f1 = lambda x: norm.pdf(x, mean=nu_ni0 / tau_ni0, cov=1 / tau_ni0) * norm.cdf(x * y_train[0, 0])
            cavity_f2 = lambda x: norm.pdf(x, mean=nu_ni1 / tau_ni1, cov=1 / tau_ni1) * norm.cdf(x * y_train[1, 0])
        # print('mu1,sigma1,mu2,sigma2:',nu_ni0 / tau_ni0,1 / tau_ni0,nu_ni1 / tau_ni1,1 / tau_ni1)
        # ax = fig.add_subplot(1, 2, i+1, projection='3d')
        # ax = fig.add_subplot(1, 2, i + 1)
        # model.plot_f(true_pdf,cavity_f1,cavity_f2,ax)
        # ax.set_title(model.inffunc.name)
    # plt.show(block=True)
    # return

        ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=np.ones((n_test,1)))
    # IQP = 0 # compute_I(y_test, np.exp(lp.flatten()), y_train)
    #     pred = np.exp(lp.flatten())

        # y_test = y_test>0
        # y_score_bin_mean, empirical_prob_pos = model.reliability_curve(y_test>0,pred,bins=10)
        # scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        # line_style = '-' if i == 0 else '-.'
        # plt.plot(y_score_bin_mean[scores_not_nan],
        #          empirical_prob_pos[scores_not_nan],linestyle=line_style,label=model.inffunc.name)
        Es+=[compute_E(y_test, np.exp(lp.flatten()))]
    # plt.plot(np.linspace(0,1,20),np.linspace(0,1,20),'-.')
    # plt.xlabel('Predictive Probability')
    # plt.ylabel('Empirical Probability')
    # plt.legend()
    # plt.show()

    # print results
    print("Negative log marginal liklihood before and after optimization")
    f = open(os.environ['proj'] + "/res/{}_output.txt".format(dataname), "a")
    f.write("Negative log marginal liklihood before and after optimization:\n")
    # f.write('{} I E: EP {} {} QP {} {}\n'.format(id, IEP, EEP, IQP, EQP))
    f.write('{} Es: EP {} QP {}\n'.format(expid, Es[0], Es[1]))
    f.close()


def synthetic(f1, f2):
    print('generating data ...')
    n = 300
    data_n1 = np.random.multivariate_normal([-0.4], [[1]], int(n / 2))
    # data_n1 = np.vstack((np.random.multivariate_normal([-2], [[1]], int(n / 4)),np.random.multivariate_normal([2], [[1]], int(n / 4))))
    data_n1 = np.array([np.append(e, -1) for e in data_n1])
    data_p1 = np.random.multivariate_normal([0.4], [[1]], int(n / 2))
    # data_p1 = np.vstack((np.random.multivariate_normal([-0.4], [[1]], int(n / 4)),np.random.multivariate_normal([0.4], [[1]], int(n / 4))))
    data_p1 = np.array([np.append(e, 1) for e in data_p1])
    data = np.vstack((data_n1, data_p1))
    train_id = np.random.choice(len(data), int(n * 0.5),replace=False)
    train = np.array([data[i] for i in train_id])
    test = np.array([data[i] for i in range(n) if i not in train_id])
    print('done')
    # run(np.array([[-0.4],[0.4]]), np.array([[-1],[1]]), test[:, 0:-1], test[:, -1].reshape((-1,1)), f1, f2)
    print('train,test.shape:',train.shape,test.shape)
    print(train_id,test)
    run(train[:, 0:-1], train[:, -1].reshape((-1, 1)),test[:, 0:-1], test[:, -1].reshape((-1, 1)), f1, f2)

def load_obj(name):
    with open(os.environ['proj'] + '/data/split_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

    f1, f2 = lambda x:x, lambda x:x #interp_fs()
    # synthetic(f1, f2)
    for expid in range(20,30):
        experiments(f1,f2,1)
    # lines = read_output_table('/home/rzhang/PycharmProjects/WGPC/res/sonar_output.txt')
    # for l in lines:
    #     print(l)
    # print('I E: ', np.mean(lines,axis=0))
