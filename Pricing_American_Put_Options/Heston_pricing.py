#
# Script for American Put Option Valuation by MCS/LSM
# in H93 and CIR85 model
#
# Examples from Medvedev & Scaillet (2010):
# "Pricing American Options Under Stochastic Volatility
# and Stochastic Interest Rates."
#
# 10_mcs/SVSI_american_mcs.py
#
# (c) Dr. Yves J. Hilpisch
# Derivatives Analytics with Python
#

import os
os.chdir("path_working_directory")

import gc
import math
import sys
import numpy as np
import pandas as pd
#import itertools as it
#from datetime import datetime
from time import time
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from scipy.integrate import quad
import xgboost

#from CIR_zcb_valuation_gen import B
from CIR_calibration import *
# r0 is imported
kappa_r, theta_r, sigma_r= CIR_calibration()


def SRD_generate_paths(x_disc, x0, kappa, theta, sigma, T, M, I, rand, row, cho_matrix):
    ''' Function to simulate Square-Root Difussion (SRD/CIR) process.

    Parameters
    ==========
    x0: float
        initial value
    kappa: float
        mean-reversion factor
    theta: float
        long-run mean
    sigma: float
        volatility factor
    T: float
        final date/time horizon
    M: int
        number of time steps
    I: int
        number of paths
    row: int
        row number for random numbers
    cho_matrix: NumPy array
        cholesky matrix

    Returns
    =======
    x: NumPy array
        simulated variance paths
    '''
    dt = T / M
    x = np.zeros((M + 1, I), dtype=np.float)
    x[0] = x0
    xh = np.zeros_like(x)
    xh[0] = x0
    sdt = math.sqrt(dt)
    for t in range(1, M + 1):
        ran = np.dot(cho_matrix, rand[:, t])
        if x_disc == 'Full Truncation':
            xh[t] = (xh[t - 1] + kappa * (theta -
                                          np.maximum(0, xh[t - 1])) * dt +
                     np.sqrt(np.maximum(0, xh[t - 1])) *
                     sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        elif x_disc == 'Partial Truncation':
            xh[t] = (xh[t - 1] + kappa * (theta - xh[t - 1]) * dt +
                     np.sqrt(np.maximum(0, xh[t - 1])) *
                     sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        elif x_disc == 'Truncation':
            x[t] = np.maximum(0, x[t - 1] +
                              kappa * (theta - x[t - 1]) * dt +
                              np.sqrt(x[t - 1]) * sigma * ran[row] * sdt)
        elif x_disc == 'Reflection':
            xh[t] = (xh[t - 1] +
                     kappa * (theta - abs(xh[t - 1])) * dt +
                     np.sqrt(abs(xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = abs(xh[t])
        elif x_disc == 'Higham-Mao':
            xh[t] = (xh[t - 1] + kappa * (theta - xh[t - 1]) * dt +
                     np.sqrt(abs(xh[t - 1])) * sigma * ran[row] * sdt)
            x[t] = abs(xh[t])
        elif x_disc == 'Simple Reflection':
            x[t] = abs(x[t - 1] + kappa * (theta - x[t - 1]) * dt +
                       np.sqrt(x[t - 1]) * sigma * ran[row] * sdt)
        elif x_disc == 'Absorption':
            xh[t] = (np.maximum(0, xh[t - 1]) +
                     kappa * (theta - np.maximum(0, xh[t - 1])) * dt +
                     np.sqrt(np.maximum(0, xh[t - 1])) *
                     sigma * ran[row] * sdt)
            x[t] = np.maximum(0, xh[t])
        else:
            print(x_disc)
            print("Not valid.")
            sys.exit(0)
    return x
    
def H93_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):

    char_func_value = H93_char_func(u - 1j * 0.5, T, r, kappa_v,
                                    theta_v, sigma_v, rho, v0)
    int_func_value = 1 / (u ** 2 + 0.25) \
        * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    return int_func_value

def H93_call_value(S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    '''

    Parameters
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time-to-maturity (for t=0)
    r: float
        constant risk-free short rate
    kappa_v: float
        mean-reversion factor
    theta_v: float
        long-run mean of variance
    sigma_v: float
        volatility of variance
    rho: float
        correlation between variance and stock/index level
    v0: float
        initial level of variance

    Returns
    =======
    call_value: float
        present value of European call option

    '''
    int_value = quad(lambda u: H93_int_func(u, S0, K, T, r, kappa_v,
                                            theta_v, sigma_v, rho, v0),
                     0, np.inf, limit=500)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) /
                     np.pi * int_value)
    return call_value

def H93_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
   
    c1 = kappa_v * theta_v
    c2 = -np.sqrt((rho * sigma_v * u * 1j - kappa_v) ** 2 -
                  sigma_v ** 2 * (-u * 1j - u ** 2))
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) \
        / (kappa_v - rho * sigma_v * u * 1j - c2)
    H1 = (r * u * 1j * T + (c1 / sigma_v ** 2) *
          ((kappa_v - rho * sigma_v * u * 1j + c2) * T -
           2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))))
    H2 = ((kappa_v - rho * sigma_v * u * 1j + c2) / sigma_v ** 2 *
          ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T))))
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value

def H93_index_paths(S0, r, v, row, cho_matrix, rand, T, q):
    ''' Simulation of the Heston (1993) index process.

    Parameters
    ==========
    S0: float
        initial value
    r: NumPy array
        simulated short rate paths
    v: NumPy array
        simulated variance paths
    row: int
        row/matrix of random number array to use
    cho_matrix: NumPy array
        Cholesky matrix

    Returns
    =======
    S: NumPy array
        simulated index level paths
    '''
    dt = T / M
    sdt = math.sqrt(dt)
    S = np.zeros((M + 1, I), dtype=np.float)
    S[0] = math.log(S0)
    for t in range(1, M + 1, 1):
        ran = np.dot(cho_matrix, rand[:, t])
        S[t] += S[t - 1]
        S[t] += ((r[t] + r[t - 1]) / 2 - q - v[t] / 2) * dt
        S[t] += np.sqrt(v[t]) * ran[row] * sdt
        if momatch is True:
            S[t] -= np.mean(np.sqrt(v[t]) * ran[row] * sdt)
    return np.exp(S)


def random_number_generator(M, I):
    ''' Function to generate pseudo-random numbers.

    Parameters
    ==========
    M: int
        time steps
    I: int
        number of simulation paths

    Returns
    =======
    rand: NumPy array
        random number array
    '''
    if antipath:
        rand = np.random.standard_normal((3, M + 1, int(I / 2)))
        rand = np.concatenate((rand, -rand), 2)
    else:
        rand = np.random.standard_normal((3, M + 1, I))
    if momatch:
        rand = rand / np.std(rand)
        rand = rand - np.mean(rand)
    return rand


D = 10  # number of basis functions
M = 100  # number of time intervals
I = 35000  # number of paths per valuation

x_disc = 'Full Truncation'
convar = True
antipath = True
momatch = True

np.random.seed(250000)  # set RNG seed value
 
def H93_put_value_mcs(S0, K, T, kappa_v, theta_v, sigma_v, rho, v0, q=0): 
    correlation_matrix = np.zeros((3, 3), dtype=np.float)
    correlation_matrix[0] = [1.0, rho, 0.0]
    correlation_matrix[1] = [rho, 1.0, 0.0]
    correlation_matrix[2] = [0.0, 0.0, 1.0]
    cho_matrix = np.linalg.cholesky(correlation_matrix)
    
    S, r, v, h, V, matrix = 0, 0, 0, 0, 0, 0
    gc.collect()

    B0T = B([r0, kappa_r, theta_r, sigma_r, 0.0, T])
    # average constant short rate/yield
    ra = -math.log(B0T) / T
    # time interval in years
    dt = T / M
    # pseudo-random numbers
    rand = random_number_generator(M, I)
    # short rate process paths
    r = SRD_generate_paths(x_disc, r0, kappa_r, theta_r,
                            sigma_r, T, M, I, rand, 0, cho_matrix)
    # volatility process paths
    v = SRD_generate_paths(x_disc, v0, kappa_v, theta_v,
                            sigma_v, T, M, I, rand, 2, cho_matrix)
    # index level process paths
    S = H93_index_paths(S0, r, v, 1, cho_matrix, rand, T, q)
    
    h = np.maximum(K - S, 0)
    # value/cash flow matrix
    V = np.maximum(K - S, 0)
    for t in range(M - 1, 0, -1):
        df = np.exp(-(r[t] + r[t + 1]) / 2 * dt)
        # select only ITM paths
        itm = np.greater(h[t], 0)
        relevant = np.nonzero(itm)
        rel_S = np.compress(itm, S[t])
        no_itm = len(rel_S)
        if no_itm <= 1:
            cv = np.zeros((I), dtype=np.float)
        else:
#            print(t)
            rel_v = np.compress(itm, v[t])
            rel_r = np.compress(itm, r[t])
            rel_V = (np.compress(itm, V[t + 1])
                       * np.compress(itm, df))
            
            matrix = np.zeros((D + 1, no_itm), dtype=np.float)
            matrix[10] = rel_S * rel_v * rel_r
            matrix[9] = rel_S * rel_v
            matrix[8] = rel_S * rel_r
            matrix[7] = rel_v * rel_r
            matrix[6] = rel_S ** 2
            matrix[5] = rel_v ** 2
            matrix[4] = rel_r ** 2
            matrix[3] = rel_S
            matrix[2] = rel_v
            matrix[1] = rel_r
            matrix[0] = 1
            reg = np.linalg.lstsq(matrix.transpose(), rel_V, rcond=None)
            cv = np.dot(reg[0], matrix)
  
        erg = np.zeros((I), dtype=np.float)
        np.put(erg, relevant, cv)
        V[t] = np.where(h[t] > erg, h[t], V[t + 1] * df)
    
    # final discounting step
    df = np.exp(-(r[0] + r[1]) / 2 * dt)
    
    ## European Option Values
    C0 = H93_call_value(S0, K, T, ra, kappa_v,
                        theta_v, sigma_v, rho, v0)
    P0 = C0 + K * B0T - S0

    y = V[1] * df

    ## Control Variate Correction
    if convar is True:
        # statistical correlation
#            x = B0T * h[-1]
#            b = (np.sum((x - np.mean(x)) * (y - np.mean(y)))
#             / np.sum((x - np.mean(x)) ** 2))
        # correction
        y_cv = y - 1.0 * (B0T * h[-1] - P0)
          # set b instead of 1.0
          # to use stat. correlation       
    else:
        y_cv = y
    V0 = max(np.sum(y_cv) / I, h[0, 0])
    return V0
