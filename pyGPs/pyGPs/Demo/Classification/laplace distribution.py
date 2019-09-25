import numpy as np
import matplotlib.pyplot as plt

def wd_fit(mu,b):
    coef = np.sqrt(2)*0.981344
    return mu, coef*b

def plot(mu,b):
    pdf_ld = lambda x: 1/2/b*np.exp(-abs(x-mu)/b)
    x = np.linspace(-10,10,300)
    y = pdf_ld(x)
    plt.plot(x,y,'.r')

    mu_2, sigma_2 = wd_fit(mu,b)
    pdf_gd = lambda x,mu,sigma: 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/sigma**2/2)
    y = pdf_gd(x,mu_2,sigma_2*0.8)
    plt.plot(x,y,'.b')

    y = pdf_gd(x, mu_2, np.sqrt(2)*b)
    plt.plot(x, y, '.g')
    plt.show()


if __name__ == '__main__':
    plot(0,4)

