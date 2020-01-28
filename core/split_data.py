from read_data import *
import numpy as np
import pickle, sys, os
from sklearn.preprocessing import StandardScaler

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('TkAgg')
    os.environ['proj'] = '/Users/ruizhang/PycharmProjects/WGPC'
else:
    os.environ['proj'] = '/home/rzhang/PycharmProjects/WGPC'
sys.path.append(os.environ['proj']+'/pyGPs')
sys.path.append(os.environ['proj'])

def split_data(data,dataname,seed,normalize=False):
    # data = read_ionosphere()
    np.random.seed(seed)
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
        if normalize:
            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        dic = {'x_train':x_train, 'y_train':y_train, 'x_test':x_test,'y_test':y_test}
        save_obj(dic,'{}_{}'.format(dataname,i))
    for i in range(nfolds):
        loop(i,data,l)


def save_obj(obj, name):
    with open(os.environ['proj']+'/data/split_data_paper/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for i in range(100):
        seed = i*10
        split_data(read_ionosphere(),'ionosphere_{}'.format(i),seed,True)
        split_data(read_breast_cancer(),'breast_cancer_{}'.format(i),seed,True)
        split_data(read_crabs(),'crabs_{}'.format(i),seed,True)
        split_data(read_pima(),'pima_{}'.format(i),seed,True)
        split_data(read_sonar(),'sonar_{}'.format(i),seed,True)

        split_data(read_iris(1, 2),'iris12_{}'.format(i),seed,True)
        split_data(read_iris(1, 3), 'iris13_{}'.format(i), seed,True)
        split_data(read_iris(2, 3), 'iris23_{}'.format(i), seed,True)
        split_data(read_wine(1,3), 'wine13_{}'.format(i), seed,True)
        split_data(read_wine(2, 3), 'wine23_{}'.format(i), seed,True)
        split_data(read_wine(1,2), 'wine12_{}'.format(i), seed,True)
        # split_data(read_breast_cancer(),'breast_cancer_{}'.format(seed),seed)
        # split_data(read_crabs(),'crabs_{}'.format(seed),seed)
        # split_data(read_pima(),'pima_{}'.format(seed),seed)
        # split_data(read_sonar(),'sonar_{}'.format(seed),seed)


    # split_data(read_usps(),'usps28')
    # split_data(read_iris(), 'iris23')
    # split_data(read_adult(), 'adult')
    # split_data(read_wine(1,3), 'scaled_wine13')
    # split_data(read_wine(2, 3), 'scaled_wine23')
    # split_data(read_wine(1,2), 'scaled_wine12')
    # split_data(read_car(0,1), 'scaled_car01')
    # split_data(read_car(1,3), 'scaled_car13')

