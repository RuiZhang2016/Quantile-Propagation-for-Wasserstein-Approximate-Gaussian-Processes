from __future__ import division
# Copyright (c) 2012-2014 Ricardo Andrade, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats,special
import scipy as sp
from . import link_functions
from .likelihood import Likelihood
from math import factorial
from scipy.special import gamma,hyp1f1,loggamma,gammaincc,logsumexp, erfinv
from scipy.integrate import quad


class Poisson(Likelihood):
    """
    Poisson likelihood

    .. math::
        p(y_{i}|\\lambda(f_{i})) = \\frac{\\lambda(f_{i})^{y_{i}}}{y_{i}!}e^{-\\lambda(f_{i})}

    .. Note::
        Y is expected to take values in {0,1,2,...}
    """
    def __init__(self, gp_link=None, qp=False):
        if gp_link is None:
            gp_link = link_functions.Square()
        self.qp=qp
        super(Poisson, self).__init__(gp_link, name='Poisson')

    def _conditional_mean(self, f):
        """
        the expected value of y given a value of f
        """
        return self.gp_link.transf(f)

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        .. math::
            p(y_{i}|\\lambda(f_{i})) = \\frac{\\lambda(f_{i})^{y_{i}}}{y_{i}!}e^{-\\lambda(f_{i})}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        assert np.atleast_1d(link_f).shape == np.atleast_1d(y).shape
        return np.exp(self.logpdf_link(link_f, y, Y_metadata))
        # return np.prod(stats.poisson.pmf(y,link_f))

    def logpdf_link(self, link_f, y, Y_metadata=None):
        """
        Log Likelihood Function given link(f)

        .. math::
            \\ln p(y_{i}|\lambda(f_{i})) = -\\lambda(f_{i}) + y_{i}\\log \\lambda(f_{i}) - \\log y_{i}!

        :param link_f: latent variables (link(f))
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: likelihood evaluated for this point
        :rtype: float

        """
        return -link_f + y*np.log(link_f) - special.gammaln(y+1)

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log likelihood function at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|\lambda(f_{i}))}{d\\lambda(f)} = \\frac{y_{i}}{\\lambda(f_{i})} - 1

        :param link_f: latent variables (f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: gradient of likelihood evaluated at points
        :rtype: Nx1 array

        """
        return y/link_f - 1

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """
        Hessian at y, given link(f), w.r.t link(f)
        i.e. second derivative logpdf at y given link(f_i) and link(f_j)  w.r.t link(f_i) and link(f_j)
        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\lambda(f_{i}))}{d^{2}\\lambda(f)} = \\frac{-y_{i}}{\\lambda(f_{i})^{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: Diagonal of hessian matrix (second derivative of likelihood evaluated at points f)
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        return -y/(link_f**2)

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\lambda(f_{i}))}{d^{3}\\lambda(f)} = \\frac{2y_{i}}{\\lambda(f_{i})^{3}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata which is not used in poisson distribution
        :returns: third derivative of likelihood evaluated at points f
        :rtype: Nx1 array
        """
        d3lik_dlink3 = 2*y/(link_f)**3
        return d3lik_dlink3

    def conditional_mean(self,gp):
        """
        The mean of the random variable conditioned on one value of the GP
        """
        return self.gp_link.transf(gp)

    def conditional_variance(self,gp):
        """
        The variance of the random variable conditioned on one value of the GP
        """
        return self.gp_link.transf(gp)

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        orig_shape = gp.shape
        gp = gp.flatten()
        # Ysim = np.random.poisson(self.gp_link.transf(gp), [samples, gp.size]).T
        # return Ysim.reshape(orig_shape+(samples,))
        Ysim = np.random.poisson(self.gp_link.transf(gp))
        return Ysim.reshape(orig_shape)

    def moments_match_ep(self, Y_i, tau_i, v_i, Y_metadata_i=None):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        Y_i = Y_i[0]
        if isinstance(self.gp_link, link_functions.Square):
            mu = v_i/tau_i
            sigma2 = 1/tau_i
            alpha = 2 * sigma2 / (1 + 2 * sigma2)
            y = int(Y_i)
            h = mu * mu / (1 + 2 * sigma2)
            z_F = -h / 2 / sigma2
            lZ_1 = (y + 0.5) * np.log(alpha) - np.log(2 * np.pi * sigma2) / 2 - np.sum(
                np.log(range(1, int(y) + 1))) - h + loggamma(y + 0.5)
            lZ = lZ_1 + np.log(hyp1f1(-y, 0.5, z_F))
            dlZ_1 = y / sigma2 * hyp1f1(1 - y, 1.5, z_F) / hyp1f1(-y, 0.5, z_F) - 1
            dlZ = dlZ_1 * 2 * mu / (1 + 2 * sigma2)
            mean_thm = dlZ * sigma2 + mu
            if not self.qp:
                d2lZ_1 = dlZ_1 * 2 / (1 + 2 * sigma2)
                d2lZ_2 = hyp1f1(2 - y, 2.5, z_F) / hyp1f1(-y, 0.5, z_F) * 2 * (1 - y) / 3
                # print(hyp1f1(-y, 0.5, z_F),hyp1f1(-y, 0.5, z_F))
                d2lZ_2 += hyp1f1(1 - y, 1.5, z_F) ** 2 / hyp1f1(-y, 0.5, z_F) ** 2 * 2 * y
                d2lZ_2 *= 2 * h * y / sigma2 ** 2 / (1 + 2 * sigma2)
                d2lZ = d2lZ_1 - d2lZ_2
                v_thm = sigma2 ** 2 * d2lZ + sigma2
            else:
                beta = mu / (1 + 2 * sigma2)
                lA = -(y + 0.5) * np.log(alpha) - loggamma(y + 0.5) - np.log(hyp1f1(-y, 0.5, z_F))
                norm_Z = -np.log(2 * np.pi * sigma2) / 2
                log_fac_y = np.sum(np.log(range(1, y + 1)))
                self.log_comb_list = np.array([self.log_comb(2 * y, k) for k in range(2*y+1)])
                k2_list = (np.arange(2*y+1)+1)/2
                self.logalpha_loggamma = k2_list * np.log(alpha) + loggamma(k2_list)

                def F_thm(x):
                    res_dict = {1: [], -1: []}
                    for k in range(1 + 2 * y):
                        value, sign = self.log_term_k(k, x, y, alpha, beta)
                        # if np.random.rand()<0.001: print(k,x,y,alpha,beta,sign,value)
                        if sign != 0:
                            res_dict[sign] += [value]
                    p = 0 if len(res_dict[1]) == 0 else np.exp(logsumexp(res_dict[1]))
                    n = 0 if len(res_dict[-1]) == 0 else np.exp(logsumexp(res_dict[-1]))
                    res = (p - n) * np.exp(lA) / 2
                    res = np.clip(res,1e-14,1-1e-14)
                    return res

                def logp_thm(x):
                    #loglike = lambda f: 2*y*np.log(abs(f)) - f**2 - np.sum(np.log(range(1, y + 1)))
                    #logpr = lambda x: -np.log(2*np.pi*sigma2)/2-(x-mu)*(x-mu)/2/sigma2
                    #return loglike(x)+logpr(x)-lZ
                    loglike = 2 * y * np.log(abs(x)) -  x ** 2 -log_fac_y
                    logpr = norm_Z - (x - mu) * (x - mu) / 2 / sigma2
                    return loglike + logpr - lZ

                v_thm = np.sqrt(2)*quad(lambda x: x*erfinv(2*F_thm(x)-1)*np.exp(logp_thm(x)),-np.inf,np.inf)[0]
                v_thm *= v_thm

        elif isinstance(self.gp_link, link_functions.Rectified_linear):
            z = sign*v_i/np.sqrt(tau_i)
            phi_div_Phi = derivLogCdfNormal(z)
            log_Z_hat = logCdfNormal(z)
            mu_hat = v_i/tau_i + sign*phi_div_Phi/np.sqrt(tau_i)
            sigma2_hat = (1. - a*phi_div_Phi - np.square(phi_div_Phi))/tau_i
        else:
            #TODO: do we want to revert to numerical quadrature here?
            raise ValueError("Exact moment matching not available for link {}".format(self.gp_link.__name__))

        # TODO: Output log_Z_hat instead of Z_hat (needs to be change in all others likelihoods)
        # print(np.exp(lZ), mean_thm, v_thm)
        assert not (np.isnan(lZ) or np.isnan(mean_thm) or np.isnan(v_thm)), '{} {} {}'.format(lZ, mean_thm, v_thm)
        return np.exp(lZ), mean_thm, v_thm


    # def log_term_k(self, k, x, y, alpha, beta):
    #   def log_comb(n, k):
    #         return np.sum(np.log(range(n - k + 1, n + 1))) - np.sum(np.log(range(1, k + 1)))
    #    k2 = (k + 1) / 2
    #    a = (-1) ** k + np.sign(x - beta) ** (k + 1) * (1 - gammaincc(k2, (x - beta) ** 2 / alpha))
    #    sign = np.sign(a) * (((2 * y - k) % 2 == 0) * 2 - 1 if np.sign(beta) < 0 else np.sign(beta))
    #    if sign == 0.:
    #        return 0, 0
    #    res = log_comb(2 * y, k) + (2 * y - k) * np.log(abs(beta)) + k2 * np.log(alpha) + loggamma(k2) + np.log(abs(a))
    #    return res, sign

    def log_term_k(self, k, x, y, alpha, beta):

        k2 = (k + 1) / 2
        sign_x_beta = np.sign(x - beta)
        a = (k%2 == 0)*2-1 +  (sign_x_beta if sign_x_beta >= 0 else (1-(k%2 == 0)*2))*(1 - gammaincc(k2, (x - beta) ** 2 / alpha))
        sign = np.sign(a) * ((((2 * y - k) % 2 == 0) * 2 - 1) if np.sign(beta) < 0 else np.sign(beta))
        if sign == 0.:
            return 0, 0
        res = self.log_comb_list[k] + self.safe_alogb(2 * y - k,abs(beta)) + self.logalpha_loggamma[k] + np.log(abs(a))
        return res, sign

    def log_comb(self, n, k):
        return np.sum(np.log(range(n - k + 1, n + 1))) - np.sum(np.log(range(1, k + 1)))

    def safe_alogb(self,a,b):
        if a == 0 and b == 0:
            return 0
        else:
            return a*np.log(b)
