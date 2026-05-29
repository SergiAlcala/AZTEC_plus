#!/usr/bin/python
import sys

import os
import numpy as np
import math
from helpers import *
import time


#### TRIPLE EXPONENTIAL SMOOTHING
def make_forecast(raw_values, myrequest, mytenant, Tdec):
    ## compute lambda_hat   
    if len(raw_values) < 2:
        lambda_hat = myrequest
        sigma_hat = 0
        return lambda_hat, sigma_hat

    
    window_samples = Tdec
    # window_samples = mon_samples
    #window_samples = 10
    
    n_intervals = int(len(raw_values)/window_samples) # number of history seasons
    
    # get sample windows (stats over each decision interval)
    offset = 0
    series = []
    for k in range(n_intervals):
        series.append(np.max(raw_values[offset:offset+window_samples])) 
        offset = offset + window_samples    
        
    if n_intervals < 2:
        lambda_hat = myrequest
    else:
        slen = 1 ### to check..... ############################
        # slen = int(len(series)/2) ### to check..... ############################
        # slen = int(24*60*2/Tdec) # season num of samples  ### to check..... ############################

        alpha = 0.9
        beta  = 0.2
        gamma = 0.1
        n_preds= 1     ### to check..... ############################

        result = []
        seasonals = initial_seasonal_components(series, slen)
        for i in range(len(series)+n_preds):
            if i == 0: # initial values
                smooth = series[0]
                trend  = initial_trend(series, slen)
                result.append(series[0])
                continue
            if i >= len(series): # we are forecasting
                m = i - len(series) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])
                
        lambda_hat = result[-1]

    ## compute sigma_hat
    sigma_hat = myrequest - lambda_hat # maximum uncertainty
    if n_intervals > 1:
        samples2 = []
        for k in range(len(series)):
            samples2.append(max(0, series[k]-lambda_hat))            
        sigma_hat = max(1, np.average(samples2))
    
        print(sigma_hat)
    return lambda_hat, sigma_hat
	
	
def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # print("\n\n\n\nsdfsdfsdfsdfsdfsdfsfs\n\n",n_seasons,"\n\n\n\nsdfsdfsdfsdfsdfsdfsfs\n\n")
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals
	
	
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

#### NEURAL NETWORK 
# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')