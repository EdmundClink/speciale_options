import numpy as np

#
# Zero-Coupon Bond Valuation Formula
#

def gamma(kappa_r, sigma_r):
    ''' Help Function. '''
    return np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)


def b1(alpha):
    ''' Help Function. '''
    r0, kappa_r, theta_r, sigma_r, t, T = alpha
    g = gamma(kappa_r, sigma_r)
    return (((2 * g * np.exp((kappa_r + g) * (T - t) / 2)) /
             (2 * g + (kappa_r + g) * (np.exp(g * (T - t)) - 1))) **
            (2 * kappa_r * theta_r / sigma_r ** 2))


def b2(alpha):
    ''' Help Function. '''
    r0, kappa_r, theta_r, sigma_r, t, T = alpha
    g = gamma(kappa_r, sigma_r)
    return ((2 * (np.exp(g * (T - t)) - 1)) /
            (2 * g + (kappa_r + g) * (np.exp(g * (T - t)) - 1)))


def B(alpha):
    ''' Function to value unit zero-coupon bonds in CIR85 Model.

    Parameters
    ==========
    r0: float
        initial short rate
    kappa_r: float
        mean-reversion factor
    theta_r: float
        long-run mean of short rate
    sigma_r: float
        volatility of short rate
    t: float
        valuation date
    T: float
        time horizon/interval

    Returns
    =======
    zcb_value: float
        value of zero-coupon bond
    '''
    b_1 = b1(alpha)
    b_2 = b2(alpha)
    r0, kappa_r, theta_r, sigma_r, t, T = alpha
    # expected value of r_t
    E_rt = theta_r + np.exp(-kappa_r * t) * (r0 - theta_r)
    zcb_value = b_1 * np.exp(-b_2 * E_rt)
    return zcb_value