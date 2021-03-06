# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util.univariate_Gaussian import std_norm_pdf, std_norm_cdf, derivLogCdfNormal, logCdfNormal
from . import link_functions
from .likelihood import Likelihood
from scipy.special import  erfinv,owens_t
from scipy.stats import norm
from scipy.integrate import quad
import gzip
from scipy import interpolate

class Bernoulli(Likelihood):
    """
    Bernoulli likelihood

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\lambda(f_{i})^{y_{i}}(1-f_{i})^{1-y_{i}}

    .. Note::
        Y takes values in either {-1, 1} or {0, 1}.
        link function should have the domain [0, 1], e.g. probit (default) or Heaviside

    .. See also::
        likelihood.py, for the parent class
    """
    def __init__(self, gp_link=None, qp = False, table_folder=None):
        if gp_link is None:
            gp_link = link_functions.Probit() # Heaviside()
        self.qp = qp
        super(Bernoulli, self).__init__(gp_link, 'Bernoulli')

        self.f_p1 = None
        self.f_n1 = None

        if isinstance(gp_link , (link_functions.Heaviside, link_functions.Probit)):
            self.log_concave = True

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(Bernoulli, self)._save_to_input_dict()
        input_dict["class"] = "GPy.likelihoods.Bernoulli"
        return input_dict

    def _preprocess_values(self, Y):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.

        ..Note:: Binary classification algorithm works better with classes {-1, 1}
        """
        Y_prep = Y.copy()
        Y1 = Y[Y.flatten()==1].size
        Y2 = Y[Y.flatten()==0].size
        assert Y1 + Y2 == Y.size, 'Bernoulli likelihood is meant to be used only with outputs in {0, 1}.'
        Y_prep[Y.flatten() == 0] = -1
        return Y_prep

    def moments_match_ep(self, Y_i, tau_i, v_i, Y_metadata_i=None):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        if Y_i == 1:
            sign = 1.
        elif Y_i == 0 or Y_i == -1:
            sign = -1
        else:
            raise ValueError("bad value for Bernoulli observation (0, 1)")
        if isinstance(self.gp_link, link_functions.Probit):
            z = sign*v_i*np.exp(-np.log(tau_i)/2-np.log(1 + tau_i)/2)
            phi_div_Phi = derivLogCdfNormal(z)
            log_Z_hat = logCdfNormal(z)
            Z = np.exp(log_Z_hat)
            mu_hat = v_i/tau_i + sign*phi_div_Phi/np.sqrt(tau_i**2 + tau_i)
            # sigma2_hat = 1./tau_i - (phi_div_Phi/(tau_i**2+tau_i))*(z+phi_div_Phi)
            if not self.qp:
                sigma2_hat = 1./tau_i - (phi_div_Phi/(tau_i**2+tau_i))*(z+phi_div_Phi)
            else:
                mu_ni = v_i/tau_i
                sigma_ni = 1./np.sqrt(tau_i)
                if abs(mu_ni)>=10 or (sigma_ni<0.1 and sigma_ni > 10):
                    sigma2_hat = (1. / tau_i - (phi_div_Phi / (tau_i ** 2 + tau_i)) * (z + phi_div_Phi))
                else:
                    sigma2_hat = self.f_n1(mu_ni,np.log10(sigma_ni)) if sign < 0 else self.f_p1(mu_ni,np.log10(sigma_ni))
                    sigma2_hat *= sigma2_hat*2*np.pi

        elif isinstance(self.gp_link, link_functions.Heaviside):
            # z = sign*v_i/np.sqrt(tau_i)
            # phi_div_Phi = derivLogCdfNormal(z)
            # log_Z_hat = logCdfNormal(z)
            # mu_hat = v_i/tau_i + sign*phi_div_Phi/np.sqrt(tau_i)
            mu = v_i/tau_i
            sigma = 1/np.sqrt(tau_i)
            cdf = norm.cdf(mu, scale=sigma)
            pdf = norm.pdf(mu,scale=sigma)
            Z= (1 - sign) / 2 + sign *cdf
            dlZ = sign*pdf/Z
            mu_hat = dlZ/tau_i+mu

            if not self.qp:
                # sigma2_hat = (1. - a * phi_div_Phi - np.square(phi_div_Phi)) / tau_i
                d2lZ = -sign * norm.pdf(mu, scale=sigma) *v_i / Z - dlZ * dlZ
                sigma2_hat = d2lZ/tau_i/tau_i+1/tau_i
            else:
                Fc = F_comp_probit(sign, mu, sigma)
                f = lambda t: t * erfinv(2 * Fc.F(t) - 1) * Fc.pr(t)
                sigma2_hat = np.sqrt(2) * quad(f, -np.inf, np.inf)[0]
                sigma2_hat *= sigma2_hat
        else:
            #TODO: do we want to revert to numerical quadrature here?
            raise ValueError("Exact moment matching not available for link {}".format(self.gp_link.__name__))

        # TODO: Output log_Z_hat instead of Z_hat (needs to be change in all others likelihoods)
        return Z, mu_hat, sigma2_hat


    def variational_expectations(self, Y, m, v, gh_points=None, Y_metadata=None):
        if isinstance(self.gp_link, link_functions.Probit):

            if gh_points is None:
                gh_x, gh_w = self._gh_points()
            else:
                gh_x, gh_w = gh_points
            gh_w = gh_w / np.sqrt(np.pi)
            shape = m.shape
            m,v,Y = m.flatten(), v.flatten(), Y.flatten()
            Ysign = np.where(Y==1,1,-1)
            X = gh_x[None,:]*np.sqrt(2.*v[:,None]) + (m*Ysign)[:,None]
            p = std_norm_cdf(X)
            p = np.clip(p, 1e-9, 1.-1e-9) # for numerical stability
            N = std_norm_pdf(X)
            F = np.log(p).dot(gh_w)
            NoverP = N/p
            dF_dm = (NoverP*Ysign[:,None]).dot(gh_w)
            dF_dv = -0.5*(NoverP**2 + NoverP*X).dot(gh_w)
            return F.reshape(*shape), dF_dm.reshape(*shape), dF_dv.reshape(*shape), None
        else:
            raise NotImplementedError


    def predictive_mean(self, mu, variance, Y_metadata=None):

        if isinstance(self.gp_link, link_functions.Probit):
            return std_norm_cdf(mu/np.sqrt(1+variance))

        elif isinstance(self.gp_link, link_functions.Heaviside):
            return std_norm_cdf(mu/np.sqrt(variance))

        else:
            raise NotImplementedError

    def predictive_variance(self, mu, variance, pred_mean, Y_metadata=None):

        if isinstance(self.gp_link, link_functions.Heaviside):
            return 0.
        else:
            return np.nan

    def pdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Likelihood function given inverse link of f.

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\lambda(f_{i})^{y_{i}}(1-f_{i})^{1-y_{i}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: likelihood evaluated for this point
        :rtype: float

        .. Note:
            Each y_i must be in {0, 1}
        """
        #objective = (inv_link_f**y) * ((1.-inv_link_f)**(1.-y))
        return np.where(y==1, inv_link_f, 1.-inv_link_f)

    def logpdf_link(self, inv_link_f, y, Y_metadata=None):
        """
        Log Likelihood function given inverse link of f.

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = y_{i}\\log\\lambda(f_{i}) + (1-y_{i})\\log (1-f_{i})

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: log likelihood evaluated at points inverse link of f.
        :rtype: float
        """
        #objective = y*np.log(inv_link_f) + (1.-y)*np.log(inv_link_f)
        p = np.where(y==1, inv_link_f, 1.-inv_link_f)
        return np.log(np.clip(p, 1e-9 ,np.inf))

    def dlogpdf_dlink(self, inv_link_f, y, Y_metadata=None):
        """
        Gradient of the pdf at y, given inverse link of f w.r.t inverse link of f.

        .. math::
            \\frac{d\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)} = \\frac{y_{i}}{\\lambda(f_{i})} - \\frac{(1 - y_{i})}{(1 - \\lambda(f_{i}))}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: gradient of log likelihood evaluated at points inverse link of f.
        :rtype: Nx1 array
        """
        #grad = (y/inv_link_f) - (1.-y)/(1-inv_link_f)
        #grad = np.where(y, 1./inv_link_f, -1./(1-inv_link_f))
        ff = np.clip(inv_link_f, 1e-9, 1-1e-9)
        denom = np.where(y==1, ff, -(1-ff))
        return 1./denom

    def d2logpdf_dlink2(self, inv_link_f, y, Y_metadata=None):
        """
        Hessian at y, given inv_link_f, w.r.t inv_link_f the hessian will be 0 unless i == j
        i.e. second derivative logpdf at y given inverse link of f_i and inverse link of f_j  w.r.t inverse link of f_i and inverse link of f_j.


        .. math::
            \\frac{d^{2}\\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)^{2}} = \\frac{-y_{i}}{\\lambda(f)^{2}} - \\frac{(1-y_{i})}{(1-\\lambda(f))^{2}}

        :param inv_link_f: latent variables inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points inverse link of f.
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on inverse link of f_i not on inverse link of f_(j!=i)
        """
        #d2logpdf_dlink2 = -y/(inv_link_f**2) - (1-y)/((1-inv_link_f)**2)
        #d2logpdf_dlink2 = np.where(y, -1./np.square(inv_link_f), -1./np.square(1.-inv_link_f))
        arg = np.where(y==1, inv_link_f, 1.-inv_link_f)
        ret =  -1./np.square(np.clip(arg, 1e-9, 1e9))
        if np.any(np.isinf(ret)):
            stop
        return ret

    def d3logpdf_dlink3(self, inv_link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given inverse link of f w.r.t inverse link of f

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{3}\\lambda(f)} = \\frac{2y_{i}}{\\lambda(f)^{3}} - \\frac{2(1-y_{i}}{(1-\\lambda(f))^{3}}

        :param inv_link_f: latent variables passed through inverse link of f.
        :type inv_link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in bernoulli
        :returns: third derivative of log likelihood evaluated at points inverse_link(f)
        :rtype: Nx1 array
        """
        assert np.atleast_1d(inv_link_f).shape == np.atleast_1d(y).shape
        #d3logpdf_dlink3 = 2*(y/(inv_link_f**3) - (1-y)/((1-inv_link_f)**3))
        state = np.seterr(divide='ignore')
        # TODO check y \in {0, 1} or {-1, 1}
        d3logpdf_dlink3 = np.where(y==1, 2./(inv_link_f**3), -2./((1.-inv_link_f)**3))
        np.seterr(**state)
        return d3logpdf_dlink3

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None):
        """
        Get the "quantiles" of the binary labels (Bernoulli draws). all the
        quantiles must be either 0 or 1, since those are the only values the
        draw can take!
        """
        p = self.predictive_mean(mu, var)
        return [np.asarray(p>(q/100.), dtype=np.int32) for q in quantiles]

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        ns = np.ones_like(gp, dtype=int)
        Ysim = np.random.binomial(ns, self.gp_link.transf(gp))
        return Ysim.reshape(orig_shape)

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
        return np.zeros(self.size)


    # def _Fr(self,x,v,mu,sigma):
    #     sigma2 = sigma ** 2
    #     sqrtsigma = np.sqrt(sigma2 + 1)
    #
    #     k = mu / sqrtsigma
    #     Z = norm.cdf(k / v)  # Z = norm.cdf((mu - m) / v / np.sqrt(1 + sigma2 / v2))
    #
    #     A = 1 / Z
    #     h = (x - mu) / sigma
    #     rho = sigma / sqrtsigma  # rho = 1 / np.sqrt(1 + v2 / sigma2)
    #     cdfk = norm.cdf(k)
    #     res = [0] * len(x) if np.ndim(x) else [0]
    #     hs = h if np.ndim(h) > 0 else [h]
    #     for i in range(len(hs)):
    #         h = hs[i]
    #         eta = 0 if h * k > 0 or (h * k == 0 and h + k >= 0) else -0.5
    #         if k == 0 and h == 0:
    #             res[i] = A * (0.25 + 1 / np.sin(-rho))
    #         OT1 = self._my_owens_t(h, k, rho)
    #         OT2 = self._my_owens_t(k, h, rho)
    #         res[i] = A * (0.5 * norm.cdf(h) + 0.5 * v * cdfk - v * OT1 - v * OT2 + v * eta)
    #     output = res[0]
    #     if output > 1 - 1e-14:
    #         output = 1 - 1e-14
    #     if output < 1e-14:
    #         output = 1e-14
    #     return output


    # def _pr(self,x,v,mu,sigma):
    #     Z = norm.cdf(mu / v / np.sqrt(1 + sigma ** 2))
    #     return norm.cdf(x / v) * norm.pdf(x, loc=mu, scale=sigma) / Z



class F_comp_probit:

    def __init__(self,v,mu,sigma):
        self.mu = mu
        self.sigma = sigma
        self.v = v
        sigma2 = sigma*sigma
        self.sqrtsigma2 = np.sqrt(sigma2 + 1)
        self.k = mu / self.sqrtsigma2
        self.Z = norm.cdf(self.k / v)
        rho = sigma / self.sqrtsigma2
        self.cdfk = norm.cdf(self.k)
        self.db0 = (0.25 + 1 / np.sin(-rho))/self.Z

    def F(self,x):
        h = (x - self.mu) / self.sigma
        if self.k == 0 and h == 0:
            return self.db0
        eta = 0 if h * self.k > 0 or (h * self.k == 0 and h + self.k >= 0) else -0.5
        OT1 = self._my_owens_t(h, self.k)
        OT2 = self._my_owens_t(self.k, h)
        res = (0.5 * norm.cdf(h) + 0.5 * self.v * self.cdfk - self.v * OT1 - self.v * OT2 + self.v * eta)/self.Z
        return np.clip(res,1e-14,1-1e-14)

    def pr(self,x):
        return norm.cdf(x / self.v) * norm.pdf(x, loc=self.mu, scale=self.sigma) / self.Z

    def _my_owens_t(self,x1,x2):
        if x1 == 0 and x2 > 0:
            return 0.25
        elif x1 == 0 and x2 < 0:
            return -0.25
        else:
            # return owens_t(x1, (x2 / x1 + self.rho) / np.sqrt(1 - self.rho ** 2))
            return owens_t(x1, x2 / x1*self.sqrtsigma2 + self.sigma)

class F_comp_heaviside:

    def __init__(self,v,mu,sigma):
        self.mu = mu
        self.sigma = sigma
        self.v = v
        self.Z = (1-v)/2+v*norm.cdf(mu,scale=sigma)

    def F(self,x):
        res=((1 - self.v) / 2 / self.Z* norm.cdf(x, loc=mu, scale=sigma)) if x < 0 else \
            ((1 + self.v) / 2 / self.Z * norm.cdf(x, loc=mu, scale=sigma)-self.v / self.Z * norm.cdf(0, loc=self.mu, scale=self.sigma))
        return np.clip(res,1e-14,1-1e-14)

    def pr(self,x):
        return ((1+self.v)/2 if x>=0 else (1-self.v)/2)*norm.pdf(x, loc=self.mu, scale=self.sigma)/self.Z
