# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""
Gaussian Processes classification examples
"""
import GPy
import pickle
import os
from sklearn.model_selection import train_test_split
import numpy as np
default_seed = 10000
from matplotlib import pyplot as plt



def oil(num_inducing=50, max_iters=100, kernel=None, optimize=True, plot=True):
    """
    Run a Gaussian process classification on the three phase oil data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    """
    try:import pods
    except ImportError:raise ImportWarning('Need pods for example datasets. See https://github.com/sods/ods, or pip install pods.')
    data = pods.datasets.oil()
    X = data['X']
    Xtest = data['Xtest']
    Y = data['Y'][:, 0:1]
    Ytest = data['Ytest'][:, 0:1]
    Y[Y.flatten()==-1] = 0
    Ytest[Ytest.flatten()==-1] = 0

    # Create GP model
    m = GPy.models.SparseGPClassification(X, Y, kernel=kernel, num_inducing=num_inducing)
    m.Ytest = Ytest

    # Contrain all parameters to be positive
    #m.tie_params('.*len')
    m['.*len'] = 10.

    # Optimize
    if optimize:
        m.optimize(messages=1)
    print(m)

    #Test
    probs = m.predict(Xtest)[0]
    GPy.util.classification.conf_matrix(probs, Ytest)
    return m

def toy_linear_1d_classification(seed=default_seed, optimize=True, plot=True):
    """
    Simple 1D classification example using EP approximation

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """
    try:import pods
    except ImportError:raise ImportWarning('Need pods for example datasets. See https://github.com/sods/ods, or pip install pods.')
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0

    X_train, X_test, Y_train, Y_test = train_test_split(data['X'], Y, test_size = 0.2,random_state=seed)
    kernel = GPy.kern.RBF(data['X'].shape[1])
    likelihood = GPy.likelihoods.Bernoulli()
    # inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(ep_mode='alternated')

    # Model definition
    # m = GPy.models.GPClassification(X_train, Y_train,kernel=kernel,likelihood=likelihood,inference_method=inference_method)
    # m = GPy.core.GP(X_train, Y_train,kernel=kernel,likelihood=likelihood,inference_method=inference_method)
    m = GPy.models.GPVariationalGaussianApproximation(X_train,Y_train,kernel=kernel,likelihood=likelihood)

    # Optimize
    if optimize:
        m.optimize()
    l1 = np.mean(m.log_predictive_density(X_test, Y_test))

    if plot:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])
    plt.show()

    kernel = GPy.kern.RBF(data['X'].shape[1])
    likelihood = GPy.likelihoods.Bernoulli()
    inference_method = GPy.inference.latent_function_inference.EP(ep_mode='nested')
    # Model definition
    m = GPy.models.GPClassification(X_train, Y_train, kernel=kernel, likelihood=likelihood,
                                    inference_method=inference_method)
    if optimize:
        m.optimize()
    l2 = np.mean(m.log_predictive_density(X_test, Y_test))

    # Plot
    if plot:
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])
    # f = open("/home/rzhang/PycharmProjects/WGPC/res/probit_1d_toy_output.txt", "a")
    # f.write('{} {} {} \n'.format(seed, l1, l2))
    # f.close()
    plt.show()
    print(l1,l2)

def toy_linear_1d_classification_laplace(seed=default_seed, optimize=True, plot=True):
    """
    Simple 1D classification example using Laplace approximation

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:import pods
    except ImportError:print('pods unavailable, see https://github.com/sods/ods for example datasets')
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0


    likelihood = GPy.likelihoods.Bernoulli()
    laplace_inf = GPy.inference.latent_function_inference.Laplace()
    kernel = GPy.kern.RBF(1)

    # Model definition
    m = GPy.core.GP(data['X'], Y, kernel=kernel, likelihood=likelihood, inference_method=laplace_inf)

    # Optimize
    if optimize:
        try:
            m.optimize('scg', messages=1)
        except Exception as e:
            return m

    # Plot
    if plot:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m

def sparse_toy_linear_1d_classification(num_inducing=10, seed=default_seed, optimize=True, plot=True):
    """
    Sparse 1D classification example

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:import pods
    except ImportError:print('pods unavailable, see https://github.com/sods/ods for example datasets')
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0

    # Model definition
    m = GPy.models.SparseGPClassification(data['X'], Y, num_inducing=num_inducing)
    m['.*len'] = 4.

    # Optimize
    if optimize:
        m.optimize()

    # Plot
    if plot:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m

def sparse_toy_linear_1d_classification_uncertain_input(num_inducing=10, seed=default_seed, optimize=True, plot=True):
    """
    Sparse 1D classification example

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:import pods
    except ImportError:print('pods unavailable, see https://github.com/sods/ods for example datasets')
    import numpy as np
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0
    X = data['X']
    X_var = np.random.uniform(0.3,0.5,X.shape)

    # Model definition
    m = GPy.models.SparseGPClassificationUncertainInput(X, X_var, Y, num_inducing=num_inducing)
    m['.*len'] = 4.

    # Optimize
    if optimize:
        m.optimize()

    # Plot
    if plot:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m

def toy_heaviside(seed=default_seed, max_iters=100, optimize=True, plot=True):
    """
    Simple 1D classification example using a heavy side gp transformation

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:import pods
    except ImportError:print('pods unavailable, see https://github.com/sods/ods for example datasets')
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0

    # Model definition
    kernel = GPy.kern.RBF(1)
    likelihood = GPy.likelihoods.Bernoulli(gp_link=GPy.likelihoods.link_functions.Heaviside())
    ep = GPy.inference.latent_function_inference.expectation_propagation.EP()
    m = GPy.core.GP(X=data['X'], Y=Y, kernel=kernel, likelihood=likelihood, inference_method=ep, name='gp_classification_heaviside')
    #m = GPy.models.GPClassification(data['X'], likelihood=likelihood)

    # Optimize
    if optimize:
        # Parameters optimization:
        for _ in range(5):
            m.optimize(max_iters=int(max_iters/5))
        print(m)

    # Plot
    if plot:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m

def crescent_data(model_type='Full', num_inducing=10, seed=default_seed, kernel=None, optimize=True, plot=True):
    """
    Run a Gaussian process classification on the crescent data. The demonstration calls the basic GP classification model and uses EP to approximate the likelihood.

    :param model_type: type of model to fit ['Full', 'FITC', 'DTC'].
    :param inducing: number of inducing variables (only used for 'FITC' or 'DTC').
    :type inducing: int
    :param seed: seed value for data generation.
    :type seed: int
    :param kernel: kernel to use in the model
    :type kernel: a GPy kernel
    """
    try:import pods
    except ImportError:print('pods unavailable, see https://github.com/sods/ods for example datasets')
    data = pods.datasets.crescent_data(seed=seed)
    Y = data['Y']
    Y[Y.flatten()==-1] = 0

    if model_type == 'Full':
        m = GPy.models.GPClassification(data['X'], Y, kernel=kernel)

    elif model_type == 'DTC':
        m = GPy.models.SparseGPClassification(data['X'], Y, kernel=kernel, num_inducing=num_inducing)
        m['.*len'] = 10.

    elif model_type == 'FITC':
        m = GPy.models.FITCClassification(data['X'], Y, kernel=kernel, num_inducing=num_inducing)
        m['.*len'] = 3.
    if optimize:
        m.optimize(messages=1)

    if plot:
        m.plot()

    print(m)
    return m


def other_data(input_filename,output_filename, seed=default_seed, kernel=None, optimize=True, plot=True, qp=False):
    res = []
    with open(input_filename, 'rb') as f:
        data = pickle.load(f)
        if not qp:
            # Model 1 definition
            kernel = GPy.kern.RBF(data['x_train'].shape[1])
            likelihood = GPy.likelihoods.Bernoulli()
            m = GPy.models.GPVariationalGaussianApproximation(data['x_train'], data['y_train'][:,None], kernel=kernel, likelihood=likelihood)
            m.optimize()
            l1_list = m.log_predictive_density(x_test=data['x_test'],y_test=data['y_test'][:,None])
            # Model 2 definition
            kernel2 = GPy.kern.RBF(data['x_train'].shape[1])
            likelihood2 = GPy.likelihoods.Bernoulli()
            inference_method2 = GPy.inference.latent_function_inference.EP(ep_mode='nested')
            m2 = GPy.models.GPClassification(data['x_train'], data['y_train'][:,None], kernel=kernel2, likelihood=likelihood2, inference_method=inference_method2)
            m2.optimize()
            l2_list = m2.log_predictive_density(x_test=data['x_test'],y_test=data['y_test'][:,None])
            l_list = np.hstack((l1_list,l2_list))
            res +=[l_list]
        else:
            # Model 3 definition
            kernel2 = GPy.kern.RBF(data['x_train'].shape[1])
            likelihood2 = GPy.likelihoods.Bernoulli()
            inference_method2 = GPy.inference.latent_function_inference.EP(ep_mode='nested')
            m2 = GPy.models.GPClassification(data['x_train'], data['y_train'][:,None], kernel=kernel2, likelihood=likelihood2, inference_method=inference_method2)
            m2.optimize()
            likelihood2.qp=True
            #inference_method2.max_iters = 10
            m2.optimize()
            l3_list = m2.log_predictive_density(x_test=data['x_test'],y_test=data['y_test'][:,None])
            res += [l3_list]
    np.save(output_filename,res)
    #print(np.mean(np.mean(res,axis=1),axis=0))


if __name__ == '__main__':
    pass
