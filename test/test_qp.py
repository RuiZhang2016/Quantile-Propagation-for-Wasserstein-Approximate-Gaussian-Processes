from unittest import TestCase
import qp
import time
import numpy as np

class TestQp(TestCase):

    def test_Fr(self):
        for x in np.linspace(1,100,50):
            for m in np.linspace(1,2,5):
                t = time.time()
                res1 = qp.Fr(x, m, 1, 0, 1)
                print(time.time() - t, ' seconds')
                t = time.time()
                res2 = qp.Fr_MC(x, m, 1, 0, 1)
                print(time.time() - t, ' seconds')
                assert np.isclose(res1,res2),(res1,res2)
