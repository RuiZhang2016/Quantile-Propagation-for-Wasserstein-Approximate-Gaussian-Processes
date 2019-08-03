import csv
import numpy as np
np.random.seed(0)
import copy
import h5py


def read_ionosphere():
    file = '/home/rzhang/PycharmProjects/WGPC/data/ionosphere.data'
    def str2int(s):
        return 1 if s is 'g' else -1

    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i])  for i in range(n)] for l in lines]
    return np.array(z)

def read_breast_cancer():
    file = '/Users/ruizhang/PycharmProjects/WGPC/data/breast-cancer-wisconsin.data'
    def str2int(s):
        return 1 if s is '4' else -1

    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i])  for i in range(1,n)] for l in lines if '?' not in l]
    return np.array(z)

def read_sonar():
    file = '/Users/ruizhang/PycharmProjects/WGPC/data/sonar.all-data'
    def str2int(s):
        return 1 if s is 'R' else -1

    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i])  for i in range(n)] for l in lines]
    return np.array(z)

def read_usps():
    file = '/home/rzhang/PycharmProjects/WGPC/data/usps.h5'

    with h5py.File(file, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
        return np.vstack((X_tr,X_te)),np.hstack((y_tr,y_te))

def read_crabs():
    file = '/home/rzhang/PycharmProjects/WGPC/data/crabs.dat'

    str2int = {'M':1,'F':-1,'B':1,'O':-1}

    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        lines = [l[0].split() for l in lines[1:]]
        n = len(lines[0])
        z = [[str2int[l[i]] if i <= 1 else float(l[i])  for i in range(n)] for l in lines]
    return np.array(z)

def read_pima():
    file = '/home/rzhang/PycharmProjects/WGPC/data/pima.te'

    str2int = {'M':1,'F':-1,'B':1,'O':-1}

    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        lines = [l[0].split() for l in lines[1:]]
        n = len(lines[0])
        z = [[str2int[l[i]] if i <= 1 else float(l[i])  for i in range(n)] for l in lines]
    return np.array(z)

if __name__ == '__main__':
    data = read_crabs()
    print(data)
    np.random.shuffle(data)
    n = len(data)
    l = int(n/10)
    for i in range(10):
        tmp = copy.copy(data)
        test = data[i*l:i*l+l]
        train = np.vstack((data[:i*l],data[i*l+l:]))
