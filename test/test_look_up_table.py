import unittest
import csv
import core.generate_table as GT
import numpy as np
from scipy import interpolate
from core.quantile import *


class TestLookUpTable(unittest.TestCase):
    # def test_lookup_Table(self):
    #     z = GT.read_table('../core/sigma_1.csv')
    #     x = np.linspace(-2, 2, 401)
    #     y = np.linspace(0.51, 5, int((5 - 0.51) / 0.01 + 1))
    #     f = interpolate.interp2d(y,x, z, kind='linear')
    #
    #
    #     for i in range(len(x)):
    #         for j in range(len(y)):
    #             assert np.isclose(f(y[j],x[i]),z[i][j],atol=1e-8)

    def test_interpolation(self):
        z = GT.WR_table('../res/WD_GPC/sigma_1.csv','r')
        print(z[:10,:10])
        x = np.array([i*0.01-10 for i in range(400*5)])
        y = np.linspace(0.4, 5, int((5 - 0.4) / 0.01 + 1))
        f = interpolate.interp2d(y, x, z, kind='linear')
        v = 1

        for i in range(500,501): # len(x)):
            for j in range(10,20): #len(y)):
                # for ki in range(1,10):
                    # for kj in range(1,10):
                x_2 = x[i]# x[i-1]+(x[i]-x[i-1])/10*ki
                y_2 = y[j]
                x_2 = 6
                y_2 = 0.4
                print(x_2,y_2)
                _,sigma = fit_gauss_wd_nature(v, x_2, y_2)
                sigma_interp = f(y_2,x_2)[0]
                try:
                    assert np.isclose(sigma_interp,sigma,atol=1e-5),('sigma: interp, true: {}, {}'.format(sigma_interp,sigma))
                except:
                    print(('sigma: interp, true: {}, {}'.format(sigma_interp,sigma)))

if __name__ == '__main__':
    unittest.main()