import csv
import numpy as np
np.random.seed(0)
import pickle,os
from __init__ import ROOT_PATH,datanames
from sklearn.preprocessing import StandardScaler
import collections

def read_ionosphere():
    file = ROOT_PATH+'/data/ionosphere.data'
    def str2int(s):
        return 1 if s is 'g' else -1
    # print('reading Ionosphere')

    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i]) for i in range(n) if i != 1] for l in lines]
    return np.array(z)

def read_breast_cancer():
    file = ROOT_PATH+'/data/breast-cancer-wisconsin.data'
    def str2int(s):
        return 1 if s is '4' else -1

    # print('reading Cancer')
    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i]) for i in range(1,n)] for l in lines if '?' not in l]
    return np.array(z)


def read_glass():
    file = ROOT_PATH+'/data/glass.data'

    # print('reading glass')
    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i <n-1 else float(l[i])>4 for i in range(n)] for l in lines if '?' not in l]
    return np.array(z)


def read_sonar():
    file = ROOT_PATH+'/data/sonar.all-data'
    def str2int(s):
        return 1 if s is 'R' else -1
    # print('reading Sonar')
    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i])  for i in range(n)] for l in lines]
    n = len(z)
    m = len(z[0])
    return np.array(z)


def read_crabs():

    file = ROOT_PATH+'/data/crabs.dat'

    str2int = {'M':1,'F':-1,'B':1,'O':-1}
    # print('reading crabs')
    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        lines = [l[0].split() for l in lines[1:]]
        n = len(lines[0])
        z = [[str2int[l[i]] if i <= 1 else float(l[i])  for i in range(n)] for l in lines]
    m = len(z[0])
    z = np.array(z)
    x = np.array([z[:,i] for i in range(m) if i!= 1]).T
    y = z[:,1]
    z = np.hstack((x,y.reshape((-1,1))))
    return z


def read_pima():
    str2int = {'Yes':1,'No':-1}

    def read_and_remove_na(fileid):
        file = ROOT_PATH+'/data/{}'.format(fileid)
        # print('reading {}'.format(fileid))
        with open(file, 'r') as rf:
            reader = csv.reader(rf)
            lines = list(reader)
            lines = [l[0].split() for l in lines[1:]]
            lines = [l for l in lines if 'NA' not in l]
            n = len(lines[0])
            z = [[str2int[l[i]] if i == n - 1 else float(l[i]) for i in range(n)] for l in lines]
        return z

    z_tr =  read_and_remove_na('pima.tr')
    z_tr2 = read_and_remove_na('pima.tr2')
    z_te = read_and_remove_na('pima.te')

    z = np.vstack((z_tr,z_tr2))
    z = np.vstack((z,z_te))
    return z


def read_wine(l1,l2):
    file = ROOT_PATH + '/data/wine.data'
    # print('reading wine')
    with open(file, 'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        lines = np.array(lines,dtype=float)
        n = len(lines[0])
        col_ids = np.array([i+1 if i != n-1 else 0 for i in range(n)])

        lines = lines[:,col_ids]

        label1, label2 = l1,l2
        lines = np.array([l for l in lines if l[-1] == label1 or l[-1] == label2])
        lines[:, -1] -= (label1 + label2) / 2
        lines[:, -1] /= abs(label1 - label2) / 2
        return lines


def split_data(data,dataname,seed,normalize=False):
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
    with open(ROOT_PATH+'/data/split_data/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        # print(f)

def generate_data():
    for i in range(100):
        seed = i * 10
        split_data(read_wine(1,3), 'wine13_{}'.format(i), seed,True)
        split_data(read_wine(2, 3), 'wine23_{}'.format(i), seed,True)
        split_data(read_wine(1,2), 'wine12_{}'.format(i), seed,True)

    # download lookup table

if __name__ == '__main__':
    generate_data()
