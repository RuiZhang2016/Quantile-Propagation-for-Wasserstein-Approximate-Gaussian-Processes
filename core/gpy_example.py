import GPy
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from joblib import Parallel,delayed
import os
from __init__ import ROOT_PATH


def poisson_square_data():
    with open('../res/poisson_regression_output_2.txt','r') as f:
        lines = f.readlines()
        lines2= np.array([lines[i].split() for i in range(120) if 'Wrong' not in lines[i]],dtype=np.float)
        print(ttest_ind(lines2[:,1],lines2[:,2]))
        print(np.mean(lines2,axis=0))


def classification_data_err_ll():
    datanames = {0: 'ionosphere', 1: 'breast_cancer', 2: 'crabs', 3: 'pima', 4: 'usps35', 5: 'usps47', 6: 'usps28',
                 7: 'sonar', 8: 'iris12',
                 9: 'iris13', 10: 'iris23', 11: 'adult', 12: 'scaled_wine12', 13: 'scaled_wine23', 14: 'scaled_wine13',
                 15: 'scaled_car01', 16: 'scaled_car02', 17: 'scaled_car13'}

    for i in range(18):
        file = os.environ['proj'] + '/res/{}_vb_ep.npy'.format(datanames[i])
        file2 = os.environ['proj'] + '/res/{}_vb_ep_qp_new_data.npy'.format(datanames[i])
        file3 = os.environ['proj'] + '/res/{}_qp.npy'.format(datanames[i])
        file4 = os.environ['proj'] + '/res/{}_qp_no_constraints.npy'.format(datanames[i])

        if os.path.isfile(file) and os.path.isfile(file2)and os.path.isfile(file3) and os.path.isfile(file4):
            res = np.load(file); res[res == -np.inf] = -1
            res2 = np.load(file2); res2[res2 == -np.inf] = -1
            res3 = np.load(file3); res3[res3 == -np.inf] = -1
            res4 = np.load(file4); res4[res4 == -np.inf] = -1
            print("---------- ", datanames[i], " ----------")
            means = np.mean(~(np.exp(res) >= 0.5), axis=1)
            means2 = np.mean(~(np.exp(res2) >= 0.5), axis=1)
            means3 = np.mean(~(np.exp(res3) >= 0.5), axis=1)
            means4 = np.mean(~(np.exp(res4) >= 0.5), axis=1)
            means_err = np.hstack((means, means2, means3, means4))
            # print('vb  ep  ep2  qp  qp2')
            print('mean: ', np.nanmean(means_err, axis=0))
            print('std: ', np.nanstd(means_err, axis=0))
            # print(ttest_ind(means_err[:, 0], means_err[:, 1]))
            print("-----")
            means =np.mean(res, axis=1) # np.array(means)
            means2= np.mean(res2,axis=1) # np.array(means2)
            means3 = np.mean(res3, axis=1)
            means4 = np.mean(res4, axis=1)
            means_ll = np.hstack((means,means2,means3,means4))
            print('mean: ', np.nanmean(means_ll,axis=0))
            print('std: ', np.nanstd(means_ll,axis=0))
            # print(ttest_ind(means_ll[:,0],means_ll[:,1]))
        else:
            print(file,' not exists!')

def copy_folder(target,source, is_folder=True):
    os.system('scp {} {} {}'.format('-r' if is_folder else '', source,target))

def copy_res_paper():
    copy_folder(ROOT_PATH+'/res/paper','u5963436@dijkstra.cecs.anu.edu.au:/home/users/u5963436/Work/WGPC/res/paper/')


if __name__ == '__main__':
    # copy_res_paper()
    args = sys.argv[1:]

    # GPy.examples.regression.toy_poisson_rbf_1d_laplace()
    # plt.show()
    # m = GPy.examples.classification.toy_linear_1d_classification()

    # Parallel(n_jobs=8)(delayed(GPy.examples.classification.toy_linear_1d_classification)(seed=i,plot=False) for i in range(100))
    # for i in range(10):
    #     GPy.examples.classification.toy_linear_1d_classification(i,plot=True)


    # for i in range(200):
    #     print(i)
    #     GPy.examples.classification.other_data()
    #     GPy.examples.regression.coal_mining_poisson_ep(seed=i,plot=False)
    #     plt.savefig('/home/rzhang/Documents/QP_Summary/figures/poisson_square_{}.pdf'.format(i))
    # classification_data_err_ll()


    # def loop(i):
    #     try:
    #         GPy.examples.classification.other_data(datanames[i])
    #     except Exception as e:
    #         print(e)

    #
    # Parallel(n_jobs=8)(delayed(loop)(i) for i in [9])

    # with open('../res/poisson_regression_output_2.txt','r') as f:
    #     lines = f.readlines()
    #     lines = np.array([l.split() for l in lines])
    #
    # with open('../res/poisson_regression_output_lbfgs.txt','r') as f:
    #     lines2 = f.readlines()
    #     lines2 = np.array([l.split() for l in lines2])
    #
    # with open('../res/poisson_regression_output_vb.txt','r') as f:
    #     lines3 = f.readlines()
    #     lines3 = np.array([l.split() for l in lines3])
    #     lines3 = np.array([l for l in lines3 if 'Wrong' not in l], dtype=np.float)
    #     print(np.mean(lines3[:120],axis=0),np.std(lines3[:120],axis=0))

    # # ind=np.argsort(lines[:,0])
    # # lines = lines[ind]
    # ind = np.argsort(lines2[:, 0])
    # lines2 = lines2[ind]
    # # lines3 = np.hstack((lines,lines2))
    # # lines = np.array([lines3[:,i] for i in [1,2,4]]).transpose()
    # lines2 = np.array([l for l in lines2 if 'Wrong' not in l],dtype=np.float)
    # for i in range(len(lines2)):
    #     ra = range(i+1)
    #     tmp = np.array([lines2[j,1:3] for j in ra])
    #     print(i,np.mean(tmp,axis=0),np.std(tmp,axis=0))
    #     comparison = [ 1 if l[0]<l[1] else 0 for l in tmp]
    #     print(np.mean(comparison))
    #     print(ttest_ind(tmp[:,0],tmp[:,1]))


    # os.environ['proj']+ '/data/split_data/{}_{}.pkl'.format(dataname, id)
    # os.environ['proj']+'/res/{}_vb_ep.npy'.format(dataname)



