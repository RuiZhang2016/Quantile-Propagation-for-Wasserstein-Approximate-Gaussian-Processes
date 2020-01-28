import GPy
try:
    from matplotlib import pyplot as plt
except:
    pass
import numpy as np
import pickle
from scipy.stats import ttest_ind
from joblib import Parallel,delayed
import os
from __init__ import ROOT_PATH
from tabulate import tabulate

def poisson_square_data():
    with open('../res/poisson_regression_output_2.txt','r') as f:
        lines = f.readlines()
        lines2= np.array([lines[i].split() for i in range(120) if 'Wrong' not in lines[i]],dtype=np.float)
        print(ttest_ind(lines2[:,1],lines2[:,2]))
        print(np.mean(lines2,axis=0))

datanames = {0: 'ionosphere', 1: 'breast_cancer', 2: 'crabs', 3: 'pima', 4: 'usps35', 5: 'usps47', 6: 'usps28',
             7: 'sonar', 8: 'iris12',
             9: 'iris13', 10: 'iris23', 11: 'adult', 12: 'wine12', 13: 'wine23', 14: 'wine13',
             15: 'scaled_car01', 16: 'scaled_car02', 17: 'scaled_car13'}





head = np.array([['Ionosphere',351,34],
            ['Cancer',683,9],
            ['Pima',732,7],
            ['Crabs',200,7],
            ['Sonar',208,60],
            ['Iris1',100, 4],
            ['Iris2', 100, 4],
            ['Iris3', 100, 4],
            ['Wine1', 130, 13],
            ['Wine2', 107, 13],
            ['Wine3', 119, 13]])




def classification_data_err_ll():
    table = []
    for i in [0,1,2,3,7,8,9,10,12,13,14]:
        print("---------- ", datanames[i], " ----------")
        means_all = []
        for set_id in range(10):
            means_all_tmp = []
            for split_id in range(10):
                file = ROOT_PATH + '/res/paper/{}_{}_{}_vb_ep.npy'.format(datanames[i],set_id,split_id)
                file2 = ROOT_PATH + '/res/paper/{}_{}_{}_qp.npy'.format(datanames[i],set_id,split_id)

                if os.path.isfile(file) and os.path.isfile(file2):
                    res = np.load(file)[0]; res[res == -np.inf] = -1
                    res2 = np.load(file2)[0]; res2[res2 == -np.inf] = -1
                    res = np.hstack((res,res2))
                    means_err = np.mean(~(np.exp(res) >= 0.5), axis=0)
                    means_ll= np.mean(res,axis=0)
                    means_all_tmp += [np.hstack((means_err,means_ll))]
                else:
                    print(file,' not exists!')
            means_all += [np.mean(means_all_tmp,axis=0)]
        means_all = np.array(means_all)
        mean_all_m = np.mean(means_all,axis=0)
        mean_all_std = np.std(means_all, axis=0)
        print('means:', mean_all_m)
        print('stds:', mean_all_std)
        mean_all_m = np.array([ float("{0:.1f}".format(mean_all_m[i]*100)) if i <3 else float("{0:.4g}".format(-mean_all_m[i]*1000))
                        for i in range(6)])
        mean_all_std = np.array([ float("{0:.1f}".format(mean_all_std[i]*100)) if i <3 else float("{0:.4g}".format(mean_all_std[i]*1000))
                        for i in range(6)])
        line = [r"$\bm{{{0}_{{\pm{1}}}}}$".format(mean_all_m[i],mean_all_std[i]) if min(mean_all_m[:3])==mean_all_m[i] else
                r"${0}_{{\pm{1}}}$".format(mean_all_m[i], mean_all_std[i]) for i in [1,2,0]]+ \
               [r"$\bm{{{0}_{{\pm{1}}}}}$".format(mean_all_m[i], mean_all_std[i]) if min(mean_all_m[3:])==mean_all_m[i] else
                r"${0}_{{\pm{1}}}$".format(mean_all_m[i], mean_all_std[i]) for i in [4,5,3]]
        table += [line]
    table = np.hstack((head,table))
    print(tabulate(table, tablefmt="latex_raw"))


def reliability_diagram():
    from sklearn.calibration import calibration_curve
    import pickle
    k = 1
    for i in [0,1,2,3,7,8,9,10,12,13,14]:
        test_ys = []
        test_ps = []
        for set_id in range(10):
            for split_id in range(10):
                file = ROOT_PATH + '/res/paper/{}_{}_{}_vb_ep.npy'.format(datanames[i],set_id,split_id)
                file2 = ROOT_PATH + '/res/paper/{}_{}_{}_qp.npy'.format(datanames[i],set_id,split_id)
                datafile = ROOT_PATH + '/data/split_data_paper/{}_{}_{}.pkl'.format(datanames[i],set_id,split_id)
                with open(datafile, 'rb') as f:
                    data = pickle.load(f)
                if os.path.isfile(file) and os.path.isfile(file2):
                    res = np.load(file)[0]; res[res == -np.inf] = -1
                    res2 = np.load(file2)[0]; res2[res2 == -np.inf] = -1
                    res = np.hstack((res, res2))
                    # print("---------- ", datanames[i] " ----------")
                    test_p = np.exp(res)
                    test_p = np.array([test_p[k] if data['y_test'][k] == 1  else (1 -test_p[k]) for k in range(len(test_p))])
                    test_ys += [data['y_test']]
                    test_ps += [test_p]
                else:
                    print(file,' not exists!')

        test_ys = np.hstack(test_ys)
        test_ps = np.vstack(test_ps).transpose()

        method_names = ['VB','EP','QP']
        linestyles = [':','-.',None]
        for j in [0,2,1]:
            fop, mpv = calibration_curve(test_ys,test_ps[j], n_bins=10, normalize=True)
            plt.subplot(1,2,2-k%2)
            if j == 0:
                plt.plot([0, 1], [0, 1], linestyle='--')
            plt.plot(mpv, fop, marker='.', label=method_names[j],linestyle=linestyles[j])
            # plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('equal')

        if k == 1:
            plt.xlabel('Predictive Probability')
            plt.ylabel('Empirical Probability')
        plt.title(datanames[i])
        plt.legend()
        if k%2 ==0:
            # plt.show()
            plt.savefig('../plots/{}_{}.pdf'.format(dn,datanames[i]))
            plt.close()
        else:
            dn = datanames[i]
        k += 1

    if k % 2 == 0:
        # plt.show()
        plt.savefig('../plots/{}.pdf'.format(dn))
        plt.close()

def copy_folder(target,source, is_folder=True):
    os.system('scp {} {} {}'.format('-r' if is_folder else '', source,target))

def copy_res_from_gadi():
    def tmp_loop(i,j,k):
        fn1 = 'rz6339@gadi.nci.org.au:/home/887/rz6339/Work/WGPC/res/paper/{}_{}_{}_vb_ep.npy'.format(datanames[i], j, k)
        fn2 = 'rz6339@gadi.nci.org.au:/home/887/rz6339/Work/WGPC/res/paper/{}_{}_{}_qp.npy'.format(datanames[i], j, k)
        copy_folder(ROOT_PATH+'/res/paper',fn1,False)
        copy_folder(ROOT_PATH + '/res/paper', fn2, False)
        print('{}_{}_{}'.format(datanames[i], j, k))

    Parallel(n_jobs=4)(delayed(tmp_loop)(i,j,k) for i in [0,1,2,3,7,8,9,10] for j in range(10) for k in range(10))

def copy_data_to_gadi():
    def tmp_loop(i,j,k):
        fn = '../data/split_data_paper/{}_{}_{}.pkl'.format(datanames[i], j, k)
        target='rz6339@gadi.nci.org.au:/home/887/rz6339/Work/WGPC/data/split_data_paper/'
        copy_folder(target,fn,False)
        print(fn)

    Parallel(n_jobs=8)(delayed(tmp_loop)(i,j,k) for i in [0,1,2,3,7,8,9,10,12,13,14] for j in range(100) for k in range(10))

if __name__ == '__main__':
    # copy_res_paper()
    #print(np.load(ROOT_PATH + '/res/paper/ionosphere_0_0_vb_ep.npy'))
    #print(np.load(ROOT_PATH+'/res/paper/ionosphere_0_0_qp.npy'))
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
    # reliability_diagram()
    # copy_res_from_gadi()
    copy_data_to_gadi()

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


    # tmp
    # for i in [12,13,14]:
    #     for j in range(10):
    #         for k in range(10):
    #             fn = '../data/split_data_paper/{}_{}_{}.pkl'.format(datanames[i],j,k)
    #             os.system('scp {} rz6339@gadi.nci.org.au:/home/887/rz6339/Work/WGPC/data/split_data_paper/'.format(fn))
    #             print(fn)


