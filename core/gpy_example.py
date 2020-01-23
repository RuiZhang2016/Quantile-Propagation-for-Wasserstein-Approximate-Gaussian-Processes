import GPy
try:
    from matplotlib import pyplot as plt
except:
    pass
import numpy as np
from scipy.stats import ttest_ind
from joblib import Parallel,delayed

def poisson_square_data():
    with open('../res/poisson_regression_output_2.txt','r') as f:
        lines = f.readlines()
        lines2= np.array([lines[i].split() for i in range(120) if 'Wrong' not in lines[i]],dtype=np.float)
        print(ttest_ind(lines2[:,1],lines2[:,2]))
        print(np.mean(lines2,axis=0))

if __name__ == '__main__':
    # GPy.examples.regression.toy_poisson_rbf_1d_laplace()
    # plt.show()
    # m = GPy.examples.classification.toy_linear_1d_classification()

    # Parallel(n_jobs=8)(delayed(GPy.examples.classification.toy_linear_1d_classification)(seed=i,plot=False) for i in range(100))
    # for i in range(10):
    #     GPy.examples.classification.toy_linear_1d_classification(i,plot=True)


    # for i in range(200):
    #     print(i)
    #     GPy.examples.regression.coal_mining_poisson_ep(seed=i,plot=True)
    #     plt.savefig('/home/rzhang/Documents/QP_Summary/figures/poisson_square_{}.pdf'.format(i))


    datanames = {0: 'ionosphere', 1: 'breast_cancer', 2: 'crabs', 3: 'pima', 4: 'usps35', 5: 'usps47', 6: 'usps28',
                 7: 'sonar', 8: 'iris12',
                 9: 'iris13', 10: 'iris23', 11: 'adult', 12: 'scaled_wine12', 13: 'scaled_wine23', 14: 'scaled_wine13',
                 15: 'scaled_car01', 16: 'scaled_car02', 17: 'scaled_car13'}
    import os
    for i in range(18):
        file = os.environ['proj'] + '/res/{}_vb_ep.npy'.format(datanames[i])
        file2 = os.environ['proj'] + '/res/{}_qp.npy'.format(datanames[i])
        if os.path.isfile(file) and os.path.isfile(file2):
            res = np.load(file)
            res2 = np.load(file2)
            assert len(res) == len(res2)
            print(datanames[i], ' experiment#: ',len(res))
            means = []
            for e in res:
                e = np.array([l for l in e if -np.inf not in l and np.nan not in l])
                means += [np.mean(e,axis=0)]
            means = np.array(means)
            means = np.hstack((means,np.mean(res2,axis=0).reshape(())))
            print('mean: ', np.mean(means,axis=0))
            print('std: ', np.std(means,axis=0))
            print(ttest_ind(means[:,0],means[:,1]))
            print(ttest_ind(means[:, 1], means[:, 2]))
            print(ttest_ind(means[:, 0], means[:, 2]))
        else:
            print(file,' not exists!')



    # def loop(i):
    #     try:
    #         GPy.examples.classification.other_data(datanames[i])
    #     except Exception as e:
    #         print(e)

    # Parallel(n_jobs=8)(delayed(loop)(i) for i in range(18))


    # with open('../res/poisson_regression_output_2.txt','r') as f:
    #     lines = f.readlines()
    #     lines = np.array([l.split() for l in lines])
    #
    # with open('../res/poisson_regression_output_lbfgs.txt','r') as f:
    #     lines2 = f.readlines()
    #     lines2 = np.array([l.split() for l in lines2])
    #
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



