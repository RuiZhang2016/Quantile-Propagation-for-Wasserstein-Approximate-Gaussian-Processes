import numpy as np
from __init__ import ROOT_PATH
import math,os,sys
sys.path.append(ROOT_PATH)
import GPy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def coal_mining_poisson_ep(seed):
    X = np.array([1851+i for i in range(112)])[:,None]
    Y = np.array([4, 5, 4, 1, 0, 4, 3, 4,0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4,
              2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
              2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 0, 1])[:,None]

    np.random.seed(seed)
    train_ids = np.array([np.random.rand()>0.5 for i in range(len(X)) ])
    train_X = X[train_ids == 1]
    train_Y = Y[train_ids == 1]
    test_X = X[train_ids != 1]
    test_Y = Y[train_ids != 1]

    mf = GPy.core.Mapping(1, 1)
    mf.f = lambda x: np.ones_like(x)
    mf.update_gradients = lambda a, b: None
    # create simple GP Model
    poisson_lik = GPy.likelihoods.Poisson()
    kern = GPy.kern.RBF(1)
    ep_inf = GPy.inference.latent_function_inference.EP(ep_mode='nested')
    res = ROOT_PATH+'/res/paper_poi'
    try:
        m = GPy.core.GP(train_X, train_Y, kernel=kern, likelihood=poisson_lik, inference_method=ep_inf,mean_function=mf)
        m.optimize()
        m.optimize()
        l1 = m.log_predictive_density(x_test=test_X, y_test=test_Y)
        mean,var = m.predict_noiseless(Xnew=test_X)
        np.save(res+'/ep_ll_{}.npy'.format(seed),l1)
        np.save(res+'/ep_mean_{}.npy'.format(seed),mean)
        np.save(res+'/ep_var_{}.npy'.format(seed),var)
        print('done {} ep'.format(seed))
    except Exception as e:
        print('{} ep wrong'.format(seed))
        print(e)

    kern = GPy.kern.RBF(1)
    poisson_lik = GPy.likelihoods.Poisson()
    ep_inf = GPy.inference.latent_function_inference.EP(ep_mode='nested')
    try:
        m = GPy.core.GP(train_X, train_Y, kernel=kern, likelihood=poisson_lik, inference_method=ep_inf, mean_function=mf)
        m.optimize()
        m.likelihood.qp=True
        # m.inference_method.max_iter
        m.optimize()
        l2 = m.log_predictive_density(x_test=test_X, y_test=test_Y)
        mean,var = m.predict_noiseless(Xnew=test_X)
        np.save(res+'/qp_ll_{}.npy'.format(seed),l2)
        np.save(res+'/qp_mean_{}.npy'.format(seed),mean)
        np.save(res+'/qp_var_{}.npy'.format(seed),var)
        print('done {} qp'.format(seed))
    except Exception as e:
        print('{} qp wrong'.format(seed))
        print(e)

    kern = GPy.kern.RBF(1)
    poisson_lik = GPy.likelihoods.Poisson()
    try:
        m = GPy.models.GPVariationalGaussianApproximation(train_X, train_Y, kernel=kern, likelihood=poisson_lik)
        m.optimize()
        l3 = m.log_predictive_density(x_test=test_X, y_test=test_Y)
        mean,var = m.predict_noiseless(Xnew=test_X)
        np.save(res+'/vb_ll_{}.npy'.format(seed),l3)
        np.save(res+'/vb_mean_{}.npy'.format(seed),mean)
        np.save(res+'/vb_var_{}.npy'.format(seed),var)
        print('done {} vb'.format(seed))
    except Exception as e:
        print('{} vb wrong'.format(seed))
        print(e)


def poisson_pred_error():
    X = np.array([1851 + i for i in range(112)])[:, None]
    Y = np.array(
        [4, 5, 4, 1, 0, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4,
         2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
         2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0,
         1, 0, 0, 1, 0, 1])[:, None]

    def poi_pred(m2,s2):
        k = (m2 + s2) ** 2 / 2 / s2 / (2 * m2 + s2)
        c = 1/k*(m2+s2)
        pred_y= math.floor(c*(k-1)) if k > 1 else 0
        return pred_y

    res_ll = []
    res_e = []
    var1 = []
    var2 = []
    var3 = []
    res_path = ROOT_PATH + '/res/paper_poi'
    for seed in range(200):
        f1 = res_path+"/ep_ll_{}.npy".format(seed)
        f2 = res_path+"/qp_ll_{}.npy".format(seed)
        f3 = res_path+"/vb_ll_{}.npy".format(seed)

        if os.path.exists(f1) and os.path.exists(f2) and os.path.exists(f3):
            ll1, ll2, ll3 = np.load(f1),np.load(f2),np.load(f3)
            ll1, ll2, ll3 = np.mean(ll1,axis=0)[0], np.mean(ll2,axis=0)[0], np.mean(ll3,axis=0)[0]
            res_ll += [[ll1,ll2,ll3]]

            # tmp = np.array(res_ll)
            # print('log-like', np.mean(tmp, axis=0), np.std(tmp, axis=0), np.mean(tmp[:, 1] > tmp[:, 0]), len(tmp))
        else:
            print('log-like {} missing'.format(seed))


        f11 = res_path+"/ep_mean_{}.npy".format(seed)
        f12 = res_path+"/ep_var_{}.npy".format(seed)
        f21 = res_path+"/qp_mean_{}.npy".format(seed)
        f22 = res_path+"/qp_var_{}.npy".format(seed)
        f31 = res_path+"/vb_mean_{}.npy".format(seed)
        f32 = res_path+"/vb_var_{}.npy".format(seed)

        if os.path.exists(f11) and os.path.exists(f12) and \
           os.path.exists(f21) and os.path.exists(f22) and \
           os.path.exists(f31) and os.path.exists(f32):
            m21, s21 = np.load(f11)**2, np.load(f12)
            m22, s22 = np.load(f21)**2, np.load(f22)
            m23, s23 = np.load(f31)**2, np.load(f32)
            var1 += [s21.flatten()]
            var2 += [s22.flatten()]
            var3 += [s23.flatten()]

            np.random.seed(seed)
            train_ids = np.array([np.random.rand() > 0.5 for i in range(len(X))])
            test_Y = Y[train_ids != 1].flatten()
            pred1 = np.array([poi_pred(m21[i,0],s21[i,0]) for i in range(len(m21))])
            pred2 = np.array([poi_pred(m22[i, 0], s22[i, 0]) for i in range(len(m21))])
            pred3 = np.array([poi_pred(m23[i, 0], s23[i, 0]) for i in range(len(m21))])

            res_e += [[np.mean(abs(test_Y - pred1)), np.mean(abs(test_Y - pred2)),np.mean(abs(test_Y - pred3))]]
        else:
            print('error {} missing'.format(seed))

    res_ll = np.array(res_ll)
    print('ep, qp, vb (ll): ', np.mean(res_ll,axis=0),np.std(res_ll,axis=0))
    res_e = np.array(res_e)
    print('ep, qp, vb (e):', np.mean(res_e,axis=0),np.std(res_e,axis=0))
    var1 = np.hstack(var1)
    var2 = np.hstack(var2)

    plt.figure(figsize=(6, 5))
    plt.scatter(var1, var2, c='blue', s=2)
    x = np.linspace(0, max(max(var2),max(var1)), 10)
    plt.plot(x, x, '-.', color='red', linewidth=3)
    plt.xlabel('EP', fontsize=18)
    plt.ylabel('QP', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    Parallel(n_jobs=4)(delayed(coal_mining_poisson_ep)(i) for i in range(200))
    # for i in range(200):
    #     coal_mining_poisson_ep(i)

    poisson_pred_error()