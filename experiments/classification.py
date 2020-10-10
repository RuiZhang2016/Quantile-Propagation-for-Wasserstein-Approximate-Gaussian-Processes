import pickle, os
from read_data import generate_data
from __init__ import datanames, ROOT_PATH
import sys
sys.path.append(ROOT_PATH)
import GPy
import numpy as np
import matplotlib.pyplot as plt
default_seed = 10000
import time
from joblib import Parallel, delayed
import gzip
from scipy import interpolate


def real_data(input_filename,output_filename,model_file,var_file,qp=False,max_iters=1000,tune=True):
    with open(input_filename, 'rb') as f:
        data = pickle.load(f)
        if qp is None:
            # vb
            kernel = GPy.kern.RBF(data['x_train'].shape[1])
            likelihood = GPy.likelihoods.Bernoulli()
            m = GPy.models.GPVariationalGaussianApproximation(data['x_train'], data['y_train'][:,None], kernel=kernel, likelihood=likelihood)
            m.optimize(max_iters=max_iters)
            l_list = m.log_predictive_density(x_test=data['x_test'],y_test=data['y_test'][:,None])
            mean,var = m.predict(Xnew = data['x_test'],include_likelihood=False)
            np.save(output_filename,l_list)
            np.save(var_file,var)
            return
        # ep or qp
        if tune:
            kernel2 = GPy.kern.RBF(data['x_train'].shape[1])
            likelihood2 = GPy.likelihoods.Bernoulli()
            inference_method2 = GPy.inference.latent_function_inference.EP(ep_mode='nested')
            m2 = GPy.models.GPClassification(data['x_train'], data['y_train'][:,None], kernel=kernel2, likelihood=likelihood2, inference_method=inference_method2)
            # l2_list = m2.log_predictive_density(x_test=data['x_test'],y_test=data['y_test'][:,None])
            m2.optimize()
            m2.save_model(model_file)
        else:
            m2 = GPy.models.GPClassification.load_model(model_file+'.zip')
            m2.likelihood.qp=qp
            if qp is True:
                m2.likelihood.f_p1 = f_p1
                m2.likelihood.f_n1 = f_n1
            m2.inference_method.max_iters = 3
            m2.optimize(max_iters=10)
            l2_list = m2.log_predictive_density(x_test=data['x_test'],y_test=data['y_test'][:,None])
            np.save(output_filename,l2_list)
            mean,var = m2.predict(Xnew = data['x_test'],include_likelihood=False)
            np.save(var_file,var)


def loop(name_i,set_i,split_i,tune=True):
    try:
        input_fn =ROOT_PATH+'/data/split_data/{}_{}_{}.pkl'.format(datanames[name_i],set_i,split_i)
        model_fn =ROOT_PATH+'/res/objs/{}_{}_{}'.format(datanames[name_i],set_i,split_i)
        res_path = ROOT_PATH +'/res/paper_clf'
        if tune:
            if not os.path.exists(model_fn+'.zip'):
                real_data(input_fn,None,model_fn,None,qp=False,tune=True)
        else:
            if os.path.exists(model_fn+'.zip'):
                var_file = res_path+'/{}_{}_{}_ep_var.npy'.format(datanames[name_i],set_i,split_i)
                output_fn = res_path +'/{}_{}_{}_ep.npy'.format(datanames[name_i], set_i, split_i)
                if not os.path.exists(output_fn) or not os.path.exists(var_file):
                    time0 = time.time()
                    real_data(input_fn,output_fn,model_fn,var_file,qp=False,tune=False)
                    print('ep time ',time.time()-time0)
                var_file = res_path+'/{}_{}_{}_qp_var.npy'.format(datanames[name_i],set_i,split_i)
                output_fn = res_path+'/{}_{}_{}_qp.npy'.format(datanames[name_i],set_i,split_i)
                if not os.path.exists(output_fn) or not os.path.exists(var_file):
                    time0 = time.time()
                    real_data(input_fn,output_fn,model_fn,var_file,qp=True,tune=False)
                    print('qp time ', time.time() - time0)
                var_file = res_path+'/{}_{}_{}_vb_var.npy'.format(datanames[name_i],set_i,split_i)
                output_fn = res_path+'/{}_{}_{}_vb.npy'.format(datanames[name_i],set_i,split_i)
                if not os.path.exists(output_fn) or not os.path.exists(var_file):
                    time0 = time.time()
                    real_data(input_fn,output_fn,None,var_file,qp=None,tune=False)
                    print('vb time ', time.time() - time0)
            else:
                print(model_fn+'.zip'+' not exists.')
    except Exception as e:
        print(e)


def classification_data_err_ll(i):
    res_path = ROOT_PATH + '/res/paper_clf'
    print("---------- ", datanames[i], " ----------")
    means_all = []

    for set_id in range(100):
        means_all_tmp = []
        for split_id in range(10):
            file = res_path+'/{}_{}_{}_ep.npy'.format(datanames[i],set_id,split_id)
            file2 = res_path+'/{}_{}_{}_qp.npy'.format(datanames[i],set_id,split_id)
            file3 = res_path+'/{}_{}_{}_vb.npy'.format(datanames[i],set_id,split_id)
            if os.path.isfile(file) and os.path.isfile(file2) and os.path.isfile(file3):
                try:
                    res = np.load(file,allow_pickle=True); res[res == -np.inf] = -1
                    res2 = np.load(file2,allow_pickle=True); res2[res2 == -np.inf] = -1
                    res3 = np.load(file3, allow_pickle=True); res3[res3 == -np.inf] = -1
                    res = np.hstack((res,res2,res3))
                    # print(res)
                    means_err = np.nanmean(~(np.exp(res) >= 0.5), axis=0)
                    means_ll= np.nanmean(res,axis=0)
                    means_all_tmp += [np.hstack((means_err,means_ll))]
                except Exception as e:
                    print(e)
                    break
            else:
                # print(file,' not exists!')
                pass
        if len(means_all_tmp)>0:
            tmp = np.nanmean(means_all_tmp,axis=0)
            means_all += [tmp]

    means_all = np.array(means_all)
    mean_all_m = np.nanmean(means_all,axis=0)
    mean_all_std = np.nanstd(means_all, axis=0)
    print('TE and TLL means (EP, QP, VB):', mean_all_m)
    print('TE and TLL stds (EP, QP, VB):', mean_all_std)


def lookup_table():
    f = gzip.open(ROOT_PATH + '/data/lookup_table_1.npy.gz', 'rb')
    table = np.load(f)
    f.close()
    mu_range = np.linspace(-10, 10, 20001)
    sigma_range = np.log10(np.logspace(-1, 1, 2001))
    f_p1 = interpolate.interp2d(mu_range, sigma_range, table, kind='linear')
    f = gzip.open(ROOT_PATH + '/data/lookup_table_-1.npy.gz', 'rb')
    table = np.load(f)
    f.close()
    f_n1 = interpolate.interp2d(mu_range, sigma_range, table, kind='linear')
    return f_p1, f_n1

f_p1, f_n1 = lookup_table()

if __name__ == '__main__':
    generate_data()
    def loop2(i,j,k):
        loop(i, j, k, tune=True)
        loop(i, j, k, tune=False)

    Parallel(n_jobs=2)(delayed(loop2)(6,j,k)  for j in range(100) for k in range(10))
    classification_data_err_ll(6)

    Parallel(n_jobs=2)(delayed(loop2)(7,j,k)  for j in range(100) for k in range(10))
    classification_data_err_ll(7)

    Parallel(n_jobs=2)(delayed(loop2)(8,j,k)  for j in range(100) for k in range(10))
    classification_data_err_ll(8)
