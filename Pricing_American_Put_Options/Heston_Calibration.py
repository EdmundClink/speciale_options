#
# Calibration of Bakshi, Cao and Chen (1997)
# Stoch Vol Jump Model to EURO STOXX Option Quotes
# Data Source: www.eurexchange.com
# via Numerical Integration
# 11_cal/BCC97_calibration_2.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#
import sys
import math
import numpy as np
import pandas as pd
import matplotlib
from scipy.optimize import brute, fmin
import os
import os.path
import random

os.chdir("path_working_directory")
from Heston_pricing import *

from CIR_calibration import CIR_calibration, r_list
from CIR_zcb_valuation import B as B1

mpl.rcParams['axes.formatter.useoffset'] = False
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(suppress=True,
                    formatter={'all': lambda x: '%5.3f' % x})

#tol = 0.2
#dividend rate
q=0

os.chdir("path_working_directory")

## Read in data

options = pd.read_csv("spyputs_march_03_22.csv", sep=',')
options = options.rename(columns={'date': 'Date', 'exp_date':'Maturity', 'Volume':'volume'})
options.Date = pd.to_datetime(options.Date, format='%Y-%m-%d')
options.Maturity = pd.to_datetime(options.Maturity, format='%Y-%m-%d')
options = options.assign(days_maturity=((options.Maturity - options.Date).dt.days).values)
options['mid_price'] = (options.best_bid + options.best_offer)/2
options['strike_price'] = options['strike_price']
#options = options.assign(days_maturity=((options.Maturity - options.Date).dt.days).values)
#options['days_maturity'] = options['maturity']

options = options[['Date', 'Maturity', 'days_maturity', 'strike_price', 'mid_price', 'volume', 'S0']]
options['T'] = options.days_maturity/365

# Choose dates used for calibration. Limiting to one because of computational expense
final_dates = ['2017-12-29']

options = options[options['Date'].isin(final_dates)]
options = options.reset_index(drop=True)

options.groupby('days_maturity').agg({'days_maturity':'count'})['days_maturity']

#JANUARY03
#options = options[options['days_maturity'].isin([17,73,164])]

#JUNE30
#options = options[options['days_maturity'].isin([14,77,168])]

#DECEMBER29
#options = options[options['days_maturity'].isin([14,33,49,77,112,168])]

#JANUARY03RANGE
#options = options[options['strike_price'].isin(range(205,251))]

#JUNE30RANGE
#options = options[options['strike_price'].isin(range(220,261))]

#DECEMBER29RANGE
#options = options[options['strike_price'].isin(range(220,273))]

#print(len(options.strike_price))

# Calibration Functions
#
i = 0
min_MSE = 500

def H93_error_function(p0):
    ''' Error function for parameter calibration in BCC97 model via
    Lewis (2001) Fourier approach.

    Parameters
    ==========
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial, instantaneous variance

    Returns
    =======
    MSE: float
        mean squared error
    '''
    global i, min_MSE
    kappa_v, theta_v, sigma_v, rho, v0 = p0
    if kappa_v < 0.0 or theta_v < 0.005 or sigma_v < 0.0 or \
            rho < -1.0 or rho > 1.0 or v0 < 0.0:
        return 500.0
    if 2 * kappa_v * theta_v < sigma_v ** 2:
        return 500.0
    se = []
    for row, option in options.iterrows():
        model_value = H93_put_value_mcs(option['S0'], option['strike_price'], option['T'],
                                        kappa_v, theta_v, sigma_v, rho, v0,q)
        se.append((model_value - option['mid_price']) ** 2)
    MSE = sum(se) / len(se)
    min_MSE = min(min_MSE, MSE)
    if i % 1 == 0:
        print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (MSE, min_MSE))
    i += 1
    return MSE


def H93_calibration_full():
    ''' Calibrates H93 stochastic volatility model to market quotes. '''
    # first run with brute force
    # (scan sensible regions)
   # p0 = brute(H93_error_function,
  #             ((2.5, 30.6, 2.5),  # kappa_v
  #              (0.01, 0.031, 0.01),  # theta_v
  #              (0.05, 1.01, 0.1),  # sigma_v
  #              (-0.99, 0.01, 0.25),  # rho
  #              (0.01, 0.021, 0.01)),  # v0
  #              finish=None)
    #p0 = np.array([12.831,  0.034,  0.896 , -0.643,  0.002])
#     second run with local, convex minimization
#     (dig deeper where promising)
    opt = fmin(H93_error_function, p0,
               xtol=0.001, ftol=0.001,
               maxiter=500, maxfun=700)
#    np.save('pathfile', np.array(opt))
    return opt


def H93_calculate_model_values(p0):
    ''' Calculates all model values given parameter vector p0. '''
    kappa_v, theta_v, sigma_v, rho, v0 = p0
    values = []
    for row, option in options.iterrows():
        model_value = H93_put_value_mcs(option['S0'], option['strike_price'], option['T'],
                                     kappa_v, theta_v, sigma_v,
                                     rho, v0, q)
 #       print(row, option.strike_price,option.days_maturity, option.S0, option.mid_price, model_value, option.mid_price - model_value , (option.mid_price - model_value)/option.mid_price)
        values.append(model_value)
    return np.array(values)

#print("Starting Calibration")
#p0 = H93_calibration_full()

#JANUARY03PARAMETER
#p0 = [11.264, 0.035, 0.835, -0.914, 0.002]

#JUNE30PARAMETER
#p0 = [10.926, 0.025, 0.742, -0.996, 0.002]

#DECEMBER29PARAMETER
p0 = [7.721, 0.023, 0.586, -0.999, 0.002]