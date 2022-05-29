import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate as sci
from scipy.optimize import fmin
import numpy as np
import pandas as pd
import os
import os.path

from CIR_zcb_valuation import B

os.chdir("path_working_directory")

mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(suppress=True,
                    formatter={'all': lambda x: '%7.6f' % x})

rate_data = pd.read_csv("path/out_of_sample/zcbdata_jan8.csv", sep=",", parse_dates=['date'], infer_datetime_format =True )

rate_data.rate = (rate_data.rate)/100

list_of_dates  = rate_data['date'].unique()
list_of_dates.sort()

initial_date = list_of_dates[0]
      
t_list = rate_data[rate_data['date']==initial_date]['days']/365
r_list = rate_data[rate_data['date']==initial_date]['rate']

zero_rates = r_list
r0 = r_list.iloc[0] # 0.0  # set to zero

tck = sci.splrep(t_list, zero_rates, k=3)  # cubic splines
tn_list = np.linspace(0.0, 10, 3*len(r_list))
ts_list = sci.splev(tn_list, tck, der=0)
de_list = sci.splev(tn_list, tck, der=1)

f = ts_list + de_list * tn_list

def plot_term_structure():
    plt.figure(figsize=(8, 5))
    plt.plot(t_list, r_list, 'ro', label='rates')
    # cubic splines
    plt.plot(tn_list, ts_list, 'b', label='interpolation', lw=1.5)
    # first derivative
    plt.plot(tn_list, de_list, 'g--', label='1st derivative', lw=1.5)
    plt.legend(loc=0)
    plt.xlabel('time horizon in years')
    plt.ylabel('rate')


def CIR_forward_rate(opt):
    ''' Function for forward rates in CIR85 model.

    Parameters
    ==========
    kappa_r: float
        mean-reversion factor
    theta_r: float
        long-run mean
    sigma_r: float
        volatility factor

    Returns
    =======
    forward_rate: float
        forward rate
    '''
    kappa_r, theta_r, sigma_r = opt
    t = tn_list
    g = np.sqrt(kappa_r ** 2 + 2 * sigma_r ** 2)
    sum1 = ((kappa_r * theta_r * (np.exp(g * t) - 1)) /
            (2 * g + (kappa_r + g) * (np.exp(g * t) - 1)))
    sum2 = r0 * ((4 * g ** 2 * np.exp(g * t)) /
                 (2 * g + (kappa_r + g) * (np.exp(g * t) - 1)) ** 2)
    forward_rate = sum1 + sum2
    return forward_rate


def CIR_error_function(opt):
    ''' Error function for CIR85 model calibration. '''
    kappa_r, theta_r, sigma_r = opt
    if 2 * kappa_r * theta_r < sigma_r ** 2:
        return 100
    if kappa_r < 0 or theta_r < 0 or sigma_r < 0.001:
        return 100
    forward_rates = CIR_forward_rate(opt)
    MSE = np.sum((f - forward_rates) ** 2) / len(f)
    # print opt, MSE
    return MSE

def CIR_calibration():
    opt = fmin(CIR_error_function, [0.3, 0.04, 0.1],
               xtol=0.00001, ftol=0.00001,
               maxiter=500, maxfun=1000)
    return opt

opt = CIR_calibration()

def plot_calibrated_frc(opt):
    ''' Plots market and calibrated forward rate curves. '''
    forward_rates = CIR_forward_rate(opt)
    plt.figure(figsize=(8, 7))
    plt.subplot(211)
    plt.ylabel('Forward rate, $f(0,T)$')
    plt.plot(tn_list, f, 'b', label='Market')
    plt.plot(tn_list, forward_rates, 'r.', label='Model')
    plt.legend(loc=0)
    plt.axis([min(tn_list) - 0.05, max(tn_list) + 0.05,
              min(f) - 0.005, max(f) * 1.1])
    plt.subplot(212)
    wi = 0.02
    plt.bar(tn_list - wi / 2, forward_rates - f, width=wi)
    plt.xlabel('Time horizon in years')
    plt.ylabel('Difference (Model rate - Market rate)')
    plt.axis([min(tn_list) - 0.05, max(tn_list) + 0.05,
              min(forward_rates - f) * 1.1, max(forward_rates - f) * 1.1])
    plt.tight_layout()

def plot_zcb_values(p0, T):
    ''' Plots unit zero-coupon bond values (discount factors). '''
    t_list = np.linspace(0.0, T, 20)
    r_list = B([r0, p0[0], p0[1], p0[2], t_list, T])
    plt.figure(figsize=(8, 5))
    plt.plot(t_list, r_list, 'b')
    plt.plot(t_list, r_list, 'ro')
    plt.xlabel('time horizon in years')
    plt.ylabel('unit zero-coupon bond value')

print("OPTIMAL PARAMETERS")
print(opt)
print("R0")
print(r0)
MSE_CIR = CIR_error_function(opt)
print("MSE")
print(MSE_CIR)