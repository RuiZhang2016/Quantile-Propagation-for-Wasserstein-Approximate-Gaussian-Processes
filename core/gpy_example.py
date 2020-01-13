import GPy
try:
    from matplotlib import pyplot as plt
except:
    pass
import numpy as np

if __name__ == '__main__':
    # GPy.examples.regression.toy_poisson_rbf_1d_laplace()
    # plt.show()
    # m = GPy.examples.classification.toy_linear_1d_classification()
    m = GPy.examples.regression.toy_poisson_rbf_1d_laplace()
    print(m.log_likelihood())
    plt.show()

    # optimizer = 'scg'
    # plot = True
    # # x_len = 100
    # # X = np.linspace(0, 10, x_len)[:, None]
    # # f_true = np.random.multivariate_normal(np.zeros(x_len), GPy.kern.RBF(1).K(X))
    # # Y = np.array([np.random.poisson(np.exp(f)) for f in f_true])[:, None]
    # X = np.array([1851+i for i in range(112)])[:, None]
    # Y = np.array([4, 5, 4, 1, 0, 4, 3, 4,0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4,
    #           2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
    #           2, 1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    #          1, 0, 0, 1, 0, 1])[:, None]
    #
    # kern = GPy.kern.RBF(1)
    # poisson_lik = GPy.likelihoods.Poisson(gp_link=GPy.likelihoods.link_functions.Log_ex_1())
    # laplace_inf = GPy.inference.latent_function_inference.EP()
    #
    # # create simple GP Model
    # m = GPy.core.GP(X, Y, kernel=kern, likelihood=poisson_lik, inference_method=laplace_inf)
    # optimize = True
    # if optimize:
    #     m.optimize(optimizer)
    # if plot:
    #     m.plot()
    #     # plot the real underlying rate function
    #     # plt.plot(X, np.exp(f_true), '--k', linewidth=2)
    # plt.show()

    # def oil(num_inducing=50, max_iters=100, kernel=None, optimize=True, plot=True):
    #     """
    #     Run a Gaussian process classification on the three phase oil data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.
    #     """
    #     try:
    #         import pods
    #     except ImportError:
    #         raise ImportWarning('Need pods for example datasets. See https://github.com/sods/ods, or pip install pods.')
    #     data = pods.datasets.oil()
    #     X = data['X']
    #     Xtest = data['Xtest']
    #     Y = data['Y'][:, 0:1]
    #     Ytest = data['Ytest'][:, 0:1]
    #     Y[Y.flatten() == -1] = 0
    #     Ytest[Ytest.flatten() == -1] = 0
    #
    #     # Create GP model
    #     m = GPy.models.SparseGPClassification(X, Y, kernel=kernel, num_inducing=num_inducing)
    #     m.Ytest = Ytest
    #
    #     # Contrain all parameters to be positive
    #     # m.tie_params('.*len')
    #     m['.*len'] = 10.
    #
    #     # Optimize
    #     if optimize:
    #         m.optimize(messages=1)
    #     print(m)
    #
    #     # Test
    #     probs = m.predict(Xtest)[0]
    #     GPy.util.classification.conf_matrix(probs, Ytest)
    #     return m
