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
    # GPy.examples.classification.toy_linear_1d_classification(1,plot=True)

    GPy.examples.regression.coal_mining_poisson_ep()
    # plt.savefig('/home/rzhang/Documents/QP_Summary/figures/poisson_square.pdf')
    #
    # poisson_square_data()