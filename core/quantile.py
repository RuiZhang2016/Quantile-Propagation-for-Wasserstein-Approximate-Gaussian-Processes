import numpy as np
from scipy.stats import norm
from scipy.special import owens_t
import scipy.integrate as integrate
import time
from scipy.special import erfinv, roots_hermite
from pynverse import inversefunc
import matplotlib.pyplot as plt
from scipy.integrate import quad,quadrature # useless

def Fr(x, v, mu, sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)

    k = mu / sqrtsigma
    Z = norm.cdf(k / v)  # Z = norm.cdf((mu - m) / v / np.sqrt(1 + sigma2 / v2))

    A = 1 / Z
    h = (x - mu) / sigma
    rho = sigma / sqrtsigma  # rho = 1 / np.sqrt(1 + v2 / sigma2)
    cdfk = norm.cdf(k)
    res = [0] * len(x) if np.ndim(x) else [0]
    hs = h if np.ndim(h) > 0 else [h]
    for i in range(len(hs)):
        h = hs[i]
        eta = 0 if h * k > 0 or (h * k == 0 and h + k >= 0) else -0.5
        if k == 0 and h == 0:
            res[i] = A * (0.25 + 1 / np.sin(-rho))
        # OT1 = owens_t(h,(k+rho*h)/h/np.sqrt(1-rho**2))
        # OT2 = owens_t(k,(h+rho*k)/k/np.sqrt(1-rho**2))
        OT1 = my_owens_t(h, k, rho)
        OT2 = my_owens_t(k, h, rho)
        res[i] = A * (0.5 * norm.cdf(h) + 0.5 * v * cdfk - v * OT1 - v * OT2 + v * eta)
    output = res[0]
    if output>1-1e-14:
        output = 1-1e-14
    if output<1e-14:
        output = 1e-14
    return output


def pr(x,v,mu,sigma):
    Z = norm.cdf(mu  / v / np.sqrt(1 + sigma ** 2))
    return norm.cdf(x/v)*norm.pdf(x,loc=mu,scale=sigma)/Z

def my_owens_t(x1,x2,rho):
    if x1 == 0 and x2>0:
        return 0.25
    elif x1 == 0 and x2<0:
        return -0.25
    else:
        return owens_t(x1,(x2/x1+rho)/np.sqrt(1-rho**2))

def Fr_MC(x,m,v,mu,sigma):
    Z = norm.cdf((mu - m) / v / np.sqrt(1 + sigma ** 2 / v ** 2))
    samples = np.linspace(-50,x,100000)
    values = norm.cdf((samples-m)/v)*norm.pdf(samples, loc=mu,scale = sigma)
    d = samples[1]-samples[0]
    integral = np.sum(values[:-1]+values[1:])/2*d
    return integral/Z


def cal_C2(v,mu,sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)
    z = mu / v / sqrtsigma
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / sqrtsigma
    inf_sigma2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)
    inf_sigma = np.sqrt(inf_sigma2)
    xs_Fr = np.linspace(inf_mu-4*inf_sigma,inf_mu+4*inf_sigma,256)
    d = xs_Fr[1]-xs_Fr[0]
    ys = np.array([Fr(x,v,mu,sigma) for x in xs_Fr])
    # print(ys)
    # print(erfinv(2*ys-1))
    xs_erf = erfinv(2*ys-1)
    prod = np.array(xs_erf)*np.array(xs_Fr)
    prod = prod[~np.isnan(prod)]
    return np.sum(np.sqrt(2)*0.5*(prod[:-1]+prod[1:]))*d


xs_norm = np.random.normal(size=3000)
def fit_gauss_wd2_sampling(v, mu, sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)
    z = mu / v / sqrtsigma  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    Z = norm.cdf(z)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf / v / sqrtsigma  # inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    #inf_sigma2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)
    #inf_sigma = np.sqrt(inf_sigma2)

    xs_Fr = xs_norm*sigma+mu# xs_disc*3*inf_sigma+inf_mu# np.linspace(inf_mu-4*inf_sigma,inf_mu+4*inf_sigma,8200) # xs_norm*sigma+mu

    ys = np.array([Fr(x, v, mu, sigma) for x in xs_Fr])
    # ys = np.array(Fr(xs_Fr,v,mu,sigma))
    ys_2 = 2 * ys - 1
    _nugget0 = -1 + 1e-14;_nugget1 = 1 - 1e-14
    ys_2[ys_2 >= _nugget1] = _nugget1; ys_2[ys_2 <= _nugget0] = _nugget0

    xs_erf = erfinv(ys_2)
    prod = xs_Fr*xs_erf*norm.cdf(xs_Fr*v)

    C2 = np.sqrt(2) * np.mean(prod)/Z
    return inf_mu, C2

samples = np.linspace(-1,1,1024)
def fit_gauss_wd2_nature(v, mu, sigma):
    print('mu,sigma: ',mu,sigma)
    sigma2 = sigma ** 2 # v2 = 1
    z = mu / v / np.sqrt(1 + sigma2)  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf/v/np.sqrt(1+sigma2)# inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    inf_sigma2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)# inf_sigma2 = sigma2 - sigma2 ** 2 * norm.pdf(z) / (v2 + sigma2) / norm.cdf(z) * (z + norm.pdf(z) / norm.cdf(z))
    inf_sigma = np.sqrt(inf_sigma2)
    xs_Fr = samples*5*inf_sigma + inf_mu # np.linspace(inf_mu - 5 * inf_sigma, inf_mu + 5 * inf_sigma, 512)

    ys = np.array([Fr(x, v, mu, sigma) for x in xs_Fr]) #ys = np.array(Fr(xs_Fr, v, mu, sigma))
    _nugget0 = -1+1e-14; _nugget1 = 1 - 1e-14
    dys = ys[1:] - ys[:-1]
    ys = 2 * ys - 1
    ys[ys >= _nugget1] = _nugget1; ys[ys <= _nugget0] = _nugget0
    xs_erf = erfinv(ys)
    prod = xs_Fr * xs_erf
    #
    inf_sigma = np.sqrt(2) * np.nansum((prod[:-1] + prod[1:]) * dys) * 0.5
    return inf_mu, inf_sigma

def fit_gauss_wd2_quad(v, mu, sigma):
    sigma2 = sigma ** 2 # v2 = 1
    z = mu / v / np.sqrt(1 + sigma2)  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf/v/np.sqrt(1+sigma2)# inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    sqrt2 = np.sqrt(2)
    # f = lambda t: (sqrt2*sigma*t+mu)*erfinv(2*Fr(t,v,mu,sigma)-1)*norm.cdf(v*(sqrt2*sigma*t+mu))/norm.cdf(z)*2*sigma
    # x,w = roots_hermite(20)
    f = lambda t: t*erfinv(2*Fr(t,v,mu,sigma)-1)*pr(t,v,mu,sigma)
    inf_sigma = sqrt2*quad(f,-np.inf,np.inf,epsrel=0)[0]
    return inf_mu,inf_sigma


def fit_gauss_wd2_gh(v, mu, sigma):
    print('mu,sigma: ',mu,sigma)
    sigma2 = sigma ** 2 # v2 = 1
    z = mu / v / np.sqrt(1 + sigma2)  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf/v/np.sqrt(1+sigma2)# inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    sqrt2 = np.sqrt(2)
    f = lambda t: (sqrt2*sigma*t+mu)*erfinv(2*Fr(sqrt2*sigma*t+mu,v,mu,sigma)-1)*norm.cdf(v*(sqrt2*sigma*t+mu))/norm.cdf(z)*2*sigma
    x,w = roots_hermite(100)
    return inf_mu,f(x),w,np.sum(w)


def fit_gauss_wd2_by_another_wd(v, mu, sigma):
    print('mu,sigma: ',mu,sigma)
    sigma2 = sigma ** 2
    # v2 = 1
    z = mu / v / np.sqrt(1 + sigma2)  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf/v/np.sqrt(1+sigma2)# inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)

    # inverse_Fr = lambda y: inversefunc(lambda x: self._Fr(x, v, mu, sigma), y_values=y, accuracy=6)
    # inf_sigma =  self.sqrt2* integrate.quad(lambda x: inverse_Fr(x)*erfinv(2 * x - 1), 0, 1)[0]

    inf_sigma2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)# inf_sigma2 = sigma2 - sigma2 ** 2 * norm.pdf(z) / (v2 + sigma2) / norm.cdf(z) * (z + norm.pdf(z) / norm.cdf(z))
    inf_sigma = np.sqrt(inf_sigma2)
    xs_Fr = samples*5*inf_sigma + inf_mu # np.linspace(inf_mu - 5 * inf_sigma, inf_mu + 5 * inf_sigma, 512)

    ys = np.array([Fr(x, v, mu, sigma) for x in xs_Fr])
    #ys = np.array(Fr(xs_Fr, v, mu, sigma))
    _nugget0 = -1+1e-14
    _nugget1 = 1 - 1e-14
    dys = ys[1:] - ys[:-1]
    ys = 2 * ys - 1
    ys[ys >= _nugget1] = _nugget1
    ys[ys <= _nugget0] = _nugget0
    xs_erf = erfinv(ys)
    # prod = (xs_Fr -inf_mu-np.sqrt(2)*xs_erf)**2
    prod = (xs_Fr - inf_mu - np.sqrt(2) /2* xs_erf) ** 2
    w22 = np.nansum((prod[:-1] + prod[1:]) * dys) * 0.5
    inf_sigma = inf_sigma2+0.25-w22 #(inf_sigma2+1-w22)/2
    return inf_mu, inf_sigma

def fit_gauss_wd2_minus_wd(v, mu, sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)
    z = mu / v / sqrtsigma  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    Z = norm.cdf(z)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf / v / sqrtsigma  # inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    sigma_q2 = sigma2 - sigma2 ** 2 * pdfdivcdf / (1 + sigma2) * (z + pdfdivcdf)
    inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y, accuracy=6)
    wd2 = integrate.quad(lambda x: (inverse_Fr(x)-inf_mu-np.sqrt(2)*erfinv(2*x-1)) ** 2, 0, 1)[0]
    inf_sigma = (sigma_q2+1-wd2)/2
    return inf_mu, inf_sigma

def fit_gauss_wd2_integral(v, mu, sigma):
    sigma2 = sigma ** 2
    sqrtsigma = np.sqrt(sigma2 + 1)
    z = mu / v / sqrtsigma  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    inf_mu = mu + sigma2 * pdfdivcdf / v / sqrtsigma  # inf_mu = mu + sigma ** 2 * norm.pdf(z) / norm.cdf(z) / v / np.sqrt(1 + sigma2 / v2)
    inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y, accuracy=6)
    inf_sigma = np.sqrt(2)*integrate.quad(lambda x: inverse_Fr(x)*erfinv(2*x-1), 0, 1)[0]

    return inf_mu, inf_sigma


class fit_gauss_wdp():
    def __init__(self,mu_q,sigma_q, Z, q, Fq, lik, samples):
        self.mu_q = mu_q
        self.sigma_q = sigma_q
        self.Z = Z
        self.q = q
        self.Fq = Fq
        self.lik = lik
        self.SQRT2 = np.sqrt(2)
        self.samples = samples
        self.erf_F_q = None
    # sigma2 = sigma ** 2
    # sqrtsigma = np.sqrt(sigma2 + 1)
    # z = mu / v / sqrtsigma  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
    # pdfdivcdf = norm.pdf(z) / norm.cdf(z)
    # mu_q = mu + sigma2 * pdfdivcdf / v / sqrtsigma
    # sigma2_q = sigma2 - sigma2 ** 2 * norm.pdf(z) / (1 + sigma2) / norm.cdf(z) * (z + norm.pdf(z) / norm.cdf(z))
    # sigma_q = np.sqrt(sigma2_q)
    def der_gauss(self, mu,sigma,p):
        # sigma2 = sigma ** 2
        # sqrtsigma = np.sqrt(sigma2 + 1)
        # z = mu / v / sqrtsigma  # z = (mu - m) / v / np.sqrt(1 + sigma2 / v2)
        # Z = norm.cdf(z)

        samples = self.samples
        N = len(samples)
        if self.erf_F_q is None:
            F_q = self.Fq(samples)
            F_q = 2*F_q-1
            _nugget0 = -1 + 1e-14
            _nugget1 = 1 - 1e-14
            F_q[F_q >= _nugget1] = _nugget1
            F_q[F_q <= _nugget0] = _nugget0
            self.erf_F_q = erfinv(F_q)
        tmp = np.abs(samples - mu -sigma*self.SQRT2*self.erf_F_q)**(p-1)*\
              np.sign(samples - mu -sigma*self.SQRT2*self.erf_F_q)*self.lik(samples)
        der_mu = -np.sum(tmp)/self.Z/N*p
        tmp = self.SQRT2*tmp*self.erf_F_q
        der_sigma = -np.sum(tmp) / self.Z / N * p
        return der_mu, der_sigma

    def der_gauss2(self, mu,sigma,p):
        xs = np.linspace(mu-10*sigma,mu+10*sigma,10000)
        N = len(xs)
        F_q = 2*self.Fq(xs)-1
        _nugget0 = -1 + 1e-14
        _nugget1 = 1 - 1e-14
        F_q[F_q >= _nugget1] = _nugget1
        F_q[F_q <= _nugget0] = _nugget0
        xs_2 = erfinv(F_q)
        tmp = abs(xs-mu-sigma*self.SQRT2*xs_2)**(p-1)\
              * np.sign(xs- mu - sigma * self.SQRT2 * xs_2)
        der_mu = -np.sum(tmp)/N*p
        tmp = self.SQRT2*tmp*xs_2
        der_sigma = -np.sum(tmp) / N * p
        return der_mu, der_sigma

    def inf(self,p,method=1):
        a1 = 0.1
        a2 = 0.1
        inf_mu = self.mu_q
        inf_sigma = self.sigma_q
        i = 0
        while True:
            old_inf_mu = inf_mu
            old_inf_sigma = inf_sigma
            if method == 1:
                der_mu, der_sigma = self.der_gauss(inf_mu,inf_sigma,p)
            else:
                der_mu, der_sigma = self.der_gauss2(inf_mu, inf_sigma, p)
            inf_mu -= a1*der_mu
            inf_sigma -= a2*der_sigma
            i+=1
            if abs(old_inf_mu - inf_mu)/old_inf_mu < 1e-6 or abs(old_inf_sigma - inf_sigma)/old_inf_sigma < 1e-6:
                break
            elif i>1000:
                break
            else:
                if i%4== 0:
                    print(i,inf_mu,inf_sigma)
        return inf_mu,inf_sigma



def fit_gauss_kl(v,mu,sigma):
    sigma2 = sigma**2
    z = mu/v/np.sqrt(1+sigma2)
    inf_mu = mu+sigma2*norm.pdf(z)/norm.cdf(z)/v/np.sqrt(1+sigma2)
    inf_sigma2 = sigma2-sigma2**2*norm.pdf(z)/(1+sigma2)/norm.cdf(z)*(z+norm.pdf(z)/norm.cdf(z))
    return inf_mu,np.sqrt(inf_sigma2)

def reliability_curve(y_true, y_score, bins=10, normalize=False):
    if normalize:  # Normalize scores into bin [0, 1]
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())

    bin_width = 1.0 / bins
    bin_centers = np.linspace(0, 1.0 - bin_width, bins) + bin_width / 2
    y_score_bin_mean = np.empty(bins)
    empirical_prob_pos = np.empty(bins)
    for i, threshold in enumerate(bin_centers):
        # determine all samples where y_score falls into the i-th bin

        bin_idx = np.logical_and(threshold - bin_width / 2 < y_score,
                                 y_score <= threshold + bin_width / 2)

        # Store mean y_score and mean empirical probability of positive class
        y_score_bin_mean[i] = y_score[bin_idx].mean()
        empirical_prob_pos[i] = y_true[bin_idx].mean()
    return y_score_bin_mean, empirical_prob_pos

def compare_different_implementation():
    v,mu,sigma = 1,1,10
    t1 = time.time()
    inf_mu_wd,inf_sigma_wd = fit_gauss_wd2_nature(v,mu,sigma)
    t2 = time.time()
    inf_mu_kl, inf_sigma_kl = fit_gauss_kl(v, mu, sigma)
    t3 = time.time()
    print('wd time, kl time: ',t2-t1,t3-t2)
    print('wd: mu ',inf_mu_wd,' sigma:',inf_sigma_wd)
    print('kl: mu ', inf_mu_kl, ' sigma:', inf_sigma_kl)
    xplot = np.linspace(mu-5,mu+5,100)
    yplot_true = pr(xplot,v,mu,sigma)
    wd_pdf = lambda x: norm.pdf(x,loc=inf_mu_wd,scale=inf_sigma_wd)
    kl_pdf = lambda x: norm.pdf(x, loc=inf_mu_kl, scale=inf_sigma_kl)
    inverse_Fr = lambda y: inversefunc(lambda x: Fr(x, v, mu, sigma), y_values=y,accuracy=6)
    L2_wd = integrate.quad(lambda x: (inverse_Fr(x)-inf_mu_wd-inf_sigma_wd*np.sqrt(2)*erfinv(2*x-1)) ** 2, 0, 1)[0]
    L2_kl = integrate.quad(lambda x: (inverse_Fr(x)-inf_mu_kl-inf_sigma_kl*np.sqrt(2)*erfinv(2*x-1)) ** 2, 0, 1)[0]
    print('wd quantile L2, kl quantile L2: ', L2_wd, L2_kl)
    print('sigma_q^2-sigma^*2: ', inf_sigma_kl**2-inf_sigma_wd**2)

    t1 = time.time()
    inf_mu_wd, inf_sigma_wd_minus_wd = fit_gauss_wd2_by_another_wd(v, mu, sigma)
    t2 = time.time()
    inf_mu_wd, inf_sigma_wd_integral = fit_gauss_wd2_integral(v, mu, sigma)
    t3 = time.time()
    print('wd minus wd time, wd integral time: ', t2 - t1, t3 - t2)
    print('sigma: minus wd ', inf_sigma_wd_minus_wd, ' integral: ', inf_sigma_wd_integral)

    L2_wd_minus_wd = \
    integrate.quad(lambda x: (inverse_Fr(x) - inf_mu_wd - inf_sigma_wd_minus_wd * np.sqrt(2) * erfinv(2 * x - 1)) ** 2, 0,
                   1)[0]
    L2_wd_integral = \
    integrate.quad(lambda x: (inverse_Fr(x) - inf_mu_kl - inf_sigma_wd_integral * np.sqrt(2) * erfinv(2 * x - 1)) ** 2, 0,
                   1)[0]
    print('wd quantile L2: minus wd, integral: ', L2_wd_minus_wd, L2_wd_integral)
    print('sigma_q^2-sigma^*2: minus wd, integral: ', inf_sigma_kl ** 2 - inf_sigma_wd_minus_wd ** 2,
          inf_sigma_kl ** 2 - inf_sigma_wd_integral ** 2)

def main2():
    v = 1
    sigma = 5
    mus = [i for i in np.linspace(5,10,30)]
    inf_mus = []
    inf_sigmas = []
    inf_sigmas2 = []
    for mu in mus:
        try:
            tmp_mu, tmp_sigma = fit_gauss_wd_nature(v, mu, sigma)
            tmp_mu, tmp_sigma2 = fit_gauss_kl(v, mu, sigma)
        except Exception as e:
            print(e)
            tmp_mu, tmp_sigma = 0, 0
            tmp_mu, tmp_sigma2 = 0, 0
        inf_mus += [tmp_mu]
        inf_sigmas += [tmp_sigma]
        inf_sigmas2 += [tmp_sigma2]
    # plt.plot(mus,inf_mus)
    # plt.show()

    plt.plot(mus,inf_sigmas,'*r')
    plt.plot(mus, inf_sigmas2, '*b')
    plt.show()

if __name__ == '__main__':
    # compare_different_implementation()
    t1 = time.time()
    print('quad: ',fit_gauss_wd2_quad(1,1,1))
    t2 = time.time()
    print(t2-t1)
    print('sampling:,',fit_gauss_wd2_sampling(1, 1, 1))
    print(time.time()-t2)
    print('ep: ',fit_gauss_kl(1,1,1))
    print('gh: ', fit_gauss_wd2_gh(1, 1, 1))