from __future__ import print_function
import pickle,sys,os
# from . import __init__
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    sys.path.append('/Users/ruizhang/PycharmProjects/WGPC/pyGPs')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    os.environ['proj'] = '/home/users/u5963436/Work/WGPC'#'/home/rzhang/PycharmProjects/WGPC'
sys.path.append(os.environ['proj']+'/pyGPs')
sys.path.append(os.environ['proj'])
import pyGPs
import numpy as np

# np.random.seed(10230)
# from .read_data import *
from core.generate_table import *
from scipy import interpolate
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import ttest_ind

def preproc(x, m, s):
    return (x - m) / s

def compute_I(ys, ps, ys_train):
    p1 = np.mean([e if e == 1 else 0 for e in ys_train])
    p2 = 1 - p1
    H = -p1 * np.log2(p1) - p2 * np.log2(p2)
    assert ys.shape == ps.shape
    I =np.mean([np.log2(ps[i]) if ys[i]==1 else np.log2(1-ps[i]) for i in range(len(ys))])+H
    return I


def compute_testll(ys, ps):
    p1 = np.mean([e if e == 1 else 0 for e in ys])
    p2 = 1 - p1
    assert ys.shape == ps.shape
    Is = (ys + 1) / 2 * np.log2(ps) + (1 - ys) / 2 * np.log2(1 - ps)
    return np.mean(Is)


def compute_E(ys, ps):
    return np.nanmean([100 if (ps[i] > 0.5) ^ (ys[i] == 1) else 0 for i in range(len(ps))])


def interp_fs():
    table1 = WR_table(os.environ['proj'] + '/res/WD_GPC/sigma_new_1.csv', 'r')
    table2 = WR_table(os.environ['proj'] + '/res/WD_GPC/sigma_new_-1.csv', 'r')
    x = [i * 0.001 - 5 for i in range(10000)]
    y = [0.4 + 0.001 * i for i in range(4601)]
    f1 = interpolate.interp2d(y, x, table1, kind='cubic')
    f2 = interpolate.interp2d(y, x, table2, kind='cubic')
    return f1, f2

datanames = {0:'ionosphere',1:'breast_cancer',2:'crabs',3:'pima',4:'usps35',5:'usps47',6:'usps28',7:'sonar',8:'iris12',
        9:'iris13',10:'iris23', 11:'adult',12:'scaled_wine12',13:'scaled_wine23',14:'scaled_wine13',15:'scaled_car01',16:'scaled_car02',17:'scaled_car13'}
def experiments(f1, f2, exp_id):
    data_id, piece_id = divmod(exp_id, 10)
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
    # modelEP.useLikelihood('Heaviside')
    # modelQP.useLikelihood('Heaviside')
    # modelEP.setOptimizer('BFGS')
    if not f1 is None and not f2 is None:
        modelQP.useInference('QP', f1, f2)
    kEP = pyGPs.cov.RBFard(log_ell_list=[0.1] * n_features, log_sigma=1.)  # kernel
    kQP = pyGPs.cov.RBFard(log_ell_list=[0.1] * n_features, log_sigma=1.)  # kernel
    modelEP.setPrior(kernel=kEP)
    modelQP.setPrior(kernel=kQP)

    # print('kernel params: ', k.hyp)

    #setup plots
    # fig = plt.figure()

    # ax = fig.gca(projection='3d')

    # calculations of EP and QP
    Es = []
    Is = []
    lps = []
    for i in range(2):
        if i == 0:
            model = modelEP
        else:
            del modelEP
            model = modelQP
        try:
            model.optimize(x_train, y_train.reshape((-1,1)), numIterations=40)
        except Exception as e:
            print(e)
            Is += [None]
            Es += [None]
            continue
        # print('kernel params: ', model.covfunc.hyp)
        #model.getPosterior(x_train, y_train)
        # print('negative log likelihood: ',model.nlZ)

        K = model.covfunc.getCovMatrix(x=x_train, mode='train')
        # print('K: ',K)
        from scipy.stats import multivariate_normal as norm

        tau_ni0 = 1 / model.inffunc.Sigma[0, 0] - model.inffunc.last_ttau[0]  # first find the cavity distribution ..
        nu_ni0 = model.inffunc.mu[0] / model.inffunc.Sigma[0, 0] - model.inffunc.last_tnu[0]
        tau_ni1 = 1 / model.inffunc.Sigma[1, 1] - model.inffunc.last_ttau[1]  # first find the cavity distribution ..
        nu_ni1 = model.inffunc.mu[1] / model.inffunc.Sigma[1, 1] - model.inffunc.last_tnu[1]
        c2 = False
        if c2:
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

        ymu, ys2, fmu, fs2, lp = model.predict(x_test, ys=np.ones(y_test.shape))

        lp = lp.flatten()
        y_test = y_test.flatten()
        lp2 = (1+y_test)/2*lp+(1-y_test)/2*(np.log(1-np.exp(lp)))
        # print('lp2:',np.exp(lp2))
        lps += [lp2]
        Is += [np.nansum(lp2)]
        # print('{} Inference Method: '.format(expid),model.inffunc.name,' ','Likelihood Function: ', model.likfunc)
        # print('test ll: ', np.sum(lp),np.exp(lp).flatten())
        # I = compute_I(y_test, np.exp(lp.flatten()), y_train)
        # pred = np.exp(lp.flatten())
        # y_test = y_test>0
        # y_score_bin_mean, empirical_prob_pos = model.reliability_curve(y_test>0,pred,bins=10)
        # scores_not_nan = np.logical_not(np.isnan(empirical_prob_pos))
        # line_style = '-' if i == 0 else '-.'
        # plt.plot(y_score_bin_mean[scores_not_nan],
        #          empirical_prob_pos[scores_not_nan],linestyle=line_style,label=model.inffunc.name)
        Es+=[compute_E(y_test, np.exp(lp))]
    # plt.plot(np.linspace(0,1,20),np.linspace(0,1,20),'-.')
    # plt.xlabel('Predictive Probability')
    # plt.ylabel('Empirical Probability')
    # plt.legend()
    # plt.savefig(os.environ['proj'] + "/res/{}_rely_diag_{}.pdf".format(dataname,expid))
    # plt.show()
    # print results
    # print("Negative log marginal liklihood before and after optimization")
    # np.save(os.environ['proj'] + "/res/lps_{}_2.npy".format(expid),lps)
    # f = open(os.environ['proj'] + "/res/{}_output_2.txt".format(dataname), "a")
    # f.write('{} Es: EP {} QP {}; Is: EP {} QP {} \n'.format(expid, Es[0], Es[1], Is[0],Is[1]))
    # f.close()
    print(expid,'Es: ', Es,'Is: ', Is)

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


def reliability_curve(y_true, y_score, bins=10, normalize=False):
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2

    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin
        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)
        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
    return y_score_bin_mean, empirical_prob_pos

def read_output_table(file_path):
    def str2float(s):
        return None if s == 'None' else float(s)

    with open(file_path,'r') as file:
        lines = file.readlines()
        for re in ['\n',';']:
            lines = np.array([l.replace(re,'') for l in lines])
        lines = np.array([l.split() for l in lines])

        lines = np.array([[str2float(l[3]),str2float(l[5]),str2float(l[8]),str2float(l[-1])] for l in lines if 'Es:' in l])
        return lines


if __name__ == '__main__':

    f1, f2 = lambda x:x, lambda x:x #interp_fs()
    # synthetic(f1, f2)R
    # experiments(f1,f2,1)
    # x =input('delete *_output_2.txt?Y/N')
    # if x == 'Y':
    #     for dataname in datanames:
    #         filename = os.environ['proj'] + "/res/{}_output_2.txt".format(dataname)
    #         if os.path.exists(filename):
    #             os.remove(filename)

    Parallel(n_jobs=2)(delayed(experiments)(f1,f2,expid) for expid in range(80,90))
    for dn_id in range(len(datanames)):
        dataname = datanames[dn_id]
        filename = os.environ['proj'] + "/res/{}_output_2.txt".format(dataname)
        if os.path.exists(filename):
            lines = read_output_table(filename)
            lines = np.array([l for l in lines if None not in l])
            try:
                lines_E = np.array([l for l in lines[:,:2] if None not in l])
                lines_Q = np.array([l for l in lines[:,2:] if None not in l])
                print(dataname,': ',np.mean(lines_E,axis=0),np.mean(lines_Q,axis=0))
                lps = None
                for exp_id in range(dn_id*10,dn_id*10+10):
                    tmp = np.load(os.environ['proj']+'/res/lps_{}_2.npy'.format(exp_id))
                    if lps is None:
                        lps = tmp # np.load(os.environ['proj']+'/res/lps_{}_2.npy'.format(exp_id))
                    elif len(tmp)>1:
                        lps = np.hstack((lps,tmp))

                print('p-value:', ttest_ind(lps[0],lps[1]))
            except Exception as e:
                print(e)
    ## reliability diagram
    # for did in range(6,7):
    #     lps = None
    #     testy = None
    #     for exp_id in range(did*10,(did+1)*10):
    #         data_id, piece_id = divmod(exp_id, 10)
    #         dic = load_obj('{}_{}'.format(datanames[data_id], piece_id))
    #         filename = os.environ['proj'] + "/res/lps_{}_2.npy".format(exp_id)
    #         if os.path.exists(filename):
    #             try:
    #                 tmp = np.load(filename,allow_pickle=True)
    #                 if tmp.shape[0] ==2:
    #                     if lps is None:
    #                         lps = tmp
    #                         testy = dic['y_test']
    #                     else:
    #                         lps = np.hstack((lps,tmp))
    #                         testy = np.hstack((testy,dic['y_test']))
    #             except Exception as e:
    #                 print(e)

    #     from sklearn.calibration import calibration_curve
    #     testy = testy>0.5
    #     lps = np.exp(lps)
    #     print('dataname:',datanames[data_id])
    #     print('percentile: ',np.mean(testy))
    #     # print('log likelihood:',lps)
    #     fop, mpv = calibration_curve(testy, lps[0], n_bins=10, normalize=True)
    #     plt.plot([0, 1], [0, 1], linestyle='--')
    #     plt.plot(mpv, fop, marker='.',label='EP')
    #     fop, mpv = calibration_curve(testy, lps[1], n_bins=10, normalize=True)
    #     plt.plot(mpv, fop, marker='.',label='QP')
    #     plt.legend()
    #     plt.show()




