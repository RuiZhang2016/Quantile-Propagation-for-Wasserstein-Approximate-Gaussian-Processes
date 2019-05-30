from unittest import TestCase
import qp
import time
import numpy as np

class TestQp(TestCase):

    def test_Fr(self):
        ms = [-2,-1,1,2]
        vs = [-2,-1,1,2]
        for x in np.linspace(-10,10,500):
            for m in ms:
                for v in vs:
                    # t = time.time()
                    res1 = qp.Fr(x, m, v, 0, 1)
                    # print(time.time() - t, ' seconds')
                    # t = time.time()
                    res2 = qp.Fr_MC(x, m, v, 0, 1)
                    # print(time.time() - t, ' seconds')
                    assert np.isclose(res1,res2),(x,' Fr, Fr_MC',res1,res2)
                    print(res1,res2)