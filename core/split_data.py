from read_data import *
import numpy as np
import pickle, sys, os
np.random.seed(0)

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    os.environ['proj'] = '/home/rzhang/PycharmProjects/WGPC'
sys.path.append(os.environ['proj']+'/pyGPs')
sys.path.append(os.environ['proj'])

def split_data(data,dataname):
    # data = read_ionosphere()
    np.random.shuffle(data)
    n = data.shape[0]
    nfolds = 10
    l = int(n / nfolds)

    def loop(i, data, l):
        # split data
        test = data[i * l:i * l + l]
        train = np.vstack((data[:i * l], data[i * l + l:]))
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]
        dic = {'x_train':x_train, 'y_train':y_train, 'x_test':x_test,'y_test':y_test}
        save_obj(dic,'{}_{}'.format(dataname,i))
    for i in range(nfolds):
        loop(i,data,l)

def save_obj(obj, name ):
    with open(os.environ['proj']+'/data/split_data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    # split_data(read_ionosphere(),'ionosphere')
    # split_data(read_breast_cancer(),'breast_cancer')
    # split_data(read_crabs(),'crabs')
    # split_data(read_pima(),'pima')
    # split_data(read_sonar(),'sonar')
    # split_data(read_usps(),'usps28')
    # split_data(read_iris(), 'iris23')
    # split_data(read_adult(), 'adult')
    split_data(read_wine(), 'wine23')