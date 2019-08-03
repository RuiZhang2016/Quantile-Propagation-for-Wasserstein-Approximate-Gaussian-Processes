import csv
import numpy as np
np.random.seed(0)
import copy

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

if __name__ == '__main__':
    data = read_ionosphere()
    np.random.shuffle(data)
    n = len(data)
    l = int(n/10)
    for i in range(10):
        tmp = copy.copy(data)
        test = data[i*l:i*l+l]
        train = np.vstack((data[:i*l],data[i*l+l:]))
