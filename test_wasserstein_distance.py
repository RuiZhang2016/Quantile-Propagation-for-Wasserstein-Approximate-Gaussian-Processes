import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize

class OP:

    def __init__(self,a=None,b=None,C=None):
        if not(a is None and b is None and C is None):
            self.a = a
            self.b = b
            self.C = C
            self.onem = np.ones(a.shape)
            self.onen = np.ones(b.shape)
            self.Cs =C*(a@self.onen.T)

    def objfunc(self,f):
        # assert len(f.shape) == 2, "f should be 2d"
        # assert f.shape[1] == 1, "f's second dimension has 1 element"
        f = f.reshape((-1,1))
        # A = (np.sqrt((self.Cs-self.a @ f.T)**2)+self.Cs-self.a @ f.T)*0.5
        A = self.Cs-self.a @ f.T
        minA = np.min(A,axis=1)
        objf = np.sum(minA - np.abs(minA))*0.5
        return -(objf +np.sum(self.b* f))

    def optimization(self):
        _minimize_method = 'L-BFGS-B'
        _minimize_options = dict(maxiter=100, disp=True, ftol=0, maxcor=20)
        dfn = grad(self.objfunc)
        bounds = [(-100,100)]*len(self.b)
        paramslin = np.ones(len(self.b))*20

        res = minimize(self.objfunc, paramslin, jac=dfn, method=_minimize_method,bounds=bounds,options=_minimize_options)

    def test(self):
        C = np.array([[1, 2, 3], [3, 4, 5]])
        a = np.array([[0.2], [0.8]])
        b = np.array([[0.1],[0.2],[0.7]])
        self.__init__(a, b, C)
        f = np.array([0.2,0.3,0.4])
        self.optimization()

if __name__ == '__main__':
    op = OP()
    op.test()