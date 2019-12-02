import csv
import numpy as np
np.random.seed(0)
import h5py
import os
import __init__

def read_ionosphere():
    file = os.environ['proj']+'/data/ionosphere.data'
    def str2int(s):
        return 1 if s is 'g' else -1
    print('reading Ionosphere')

    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i]) for i in range(n) if i != 1] for l in lines]
    n = len(z)
    m = len(z[0])
    print("#features: ", m - 1)
    print("#data: ", n)
    return np.array(z)

def read_breast_cancer():
    file = os.environ['proj']+'/data/breast-cancer-wisconsin.data'
    def str2int(s):
        return 1 if s is '4' else -1

    print('reading Cancer')
    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i]) for i in range(1,n)] for l in lines if '?' not in l]
    n = len(z)
    m = len(z[0])
    print("#features: ", m - 1)
    print("#data: ", n)
    return np.array(z)

def read_sonar():
    file = os.environ['proj']+'/data/sonar.all-data'
    def str2int(s):
        return 1 if s is 'R' else -1
    print('reading Sonar')
    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        z = [[float(l[i]) if i < n-1 else str2int(l[i])  for i in range(n)] for l in lines]
    n = len(z)
    m = len(z[0])
    print("#features: ", m - 1)
    print("#data: ", n)
    return np.array(z)

def read_usps():
    file = os.environ['proj']+'/data/usps.h5'

    print('reading USPS')
    with h5py.File(file, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]
    a,b = np.vstack((X_tr, X_te)), np.hstack((y_tr, y_te))
    ids = [i for i in range(len(b)) if b[i] == 2 or b[i] == 8] # choose which two numbers
    b = np.array([[-1 if b[i] == 2 else 1] for i in ids]) # one is labeled -1 and the other 1
    a = np.array([a[i] for i in ids])
    n = len(a)
    m = len(a[0])
    print("#features: ",m)
    print("#data: ", n)
    return np.hstack((a,b))

def read_crabs():

    file = os.environ['proj']+'/data/crabs.dat'

    str2int = {'M':1,'F':-1,'B':1,'O':-1}
    print('reading crabs')
    with open(file,'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        lines = [l[0].split() for l in lines[1:]]
        n = len(lines[0])
        z = [[str2int[l[i]] if i <= 1 else float(l[i])  for i in range(n)] for l in lines]
    n = len(z)
    m = len(z[0])
    print("#features: ", m - 1)
    print("#data: ", n)
    z = np.array(z)
    x = np.array([z[:,i] for i in range(m) if i!= 1]).T
    y = z[:,1]
    z = np.hstack((x,y.reshape((-1,1))))
    return z

def read_pima():
    str2int = {'Yes':1,'No':-1}

    def read_and_remove_na(fileid):
        file = os.environ['proj']+'/data/{}'.format(fileid)
        print('reading {}'.format(fileid))
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
    n = len(z)
    m = len(z[0])
    print("#features: ", m - 1)
    print("#data: ", n)
    return z

def read_iris():
    file = os.environ['proj'] + '/data/iris.data'
    str2int = {'Iris-versicolor': 1, 'Iris-virginica': -1 }
    with open(file, 'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        n = len(lines[0])
        lines = [[str2int[l[i]] if i == n - 1 else float(l[i]) for i in range(n)] for l in lines if len(l)==5 and l[-1] in str2int.keys()]
        return np.array(lines)

def read_adult():
    from sklearn import preprocessing
    from sklearn.preprocessing import StandardScaler
    file = os.environ['proj'] + '/data/adult.data'
    with open(file, 'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        lines1 = np.array(lines[:-1])
    file = os.environ['proj'] + '/data/adult.test'
    with open(file, 'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        lines2 = np.array(lines[1:-1])
    lines = np.vstack((lines1,lines2))
    for i in [1,3,5,6,7,8,9,13,14]:
        le = preprocessing.LabelEncoder()
        le.fit(lines[:,i])
        # print(lines[:10, i])
        lines[:,i] = le.transform(lines[:,i])
    lines = np.array(lines,dtype=float)
    scaler = StandardScaler()
    scaler.fit(lines)
    lines = scaler.transform(lines)
    lines[:,-1] = (lines[:,-1]>0)*2-1
    return lines

def read_wine():
    file = os.environ['proj'] + '/data/wine.data'
    with open(file, 'r') as rf:
        reader = csv.reader(rf)
        lines = list(reader)
        lines = np.array(lines,dtype=float)
        n = len(lines[0])
        col_ids = np.array([i+1 if i != n-1 else 0 for i in range(n)])
        print(col_ids)
        lines = lines[:,col_ids]
        lines = np.array([l for l in lines if l[-1] == 1 or l[-1] == 2])
        lines[:,-1] -= 1
        return lines

if __name__ == '__main__':
    # z = read_usps()
    # print(np.std(z,axis=0))
    # read_iris()
    read_wine()