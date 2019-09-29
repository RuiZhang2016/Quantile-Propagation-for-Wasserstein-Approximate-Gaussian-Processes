import pickle, os
from core.generate_table import WR_table
from scipy import interpolate
import numpy as np

def load_obj(name):
    with open(os.environ['proj']+'/data/split_data/'+ name + '.pkl', 'rb') as f:
        return pickle.load(f)

def preproc(x, m, s):
    return (x-m)/s

def compute_I(ys, ps, ys_train):
    p1 = np.mean([e if e == 1 else 0 for e in ys_train])
    p2 = 1-p1
    H = -p1*np.log2(p1)-p2*np.log2(p2)
    assert ys.shape == ps.shape
    I = np.mean([np.log2(ps[i]) if ys[i] == 1 else np.log2(1-ps[i]) for i in range(len(ys))])+H
    return I

def compute_E(ys,ps):
    return np.mean([100 if (ps[i] > 0.5)^(ys[i] == 1) else 0 for i in range(len(ps))])

def interp_fs():
    table1 = WR_table(os.environ['proj']+'/res/WD_GPC/sigma_new_1.csv', 'r')
    table2 = WR_table(os.environ['proj']+'/res/WD_GPC/sigma_new_-1.csv', 'r')
    x = [i * 0.001 - 5 for i in range(10000)]
    y = [0.4 + 0.001 * i for i in range(4601)]
    f1 = interpolate.interp2d(y, x, table1, kind='cubic')
    f2 = interpolate.interp2d(y, x, table2, kind='cubic')
    return f1, f2