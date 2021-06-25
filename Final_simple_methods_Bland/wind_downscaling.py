"""
This method, in a nutshell:

Windspeed of the 24 hours today will depend on the mean windspeed today, as well as, to a lesser extent, the mean windspeed yesterday and tomorrow.

Fit a polynomial to the daily means, giving greatest weight to today's daily mean, and reduced weight to days in the past and future, to get the slopes close enough while mostly preserving the daily mean.

    For the target date, t:
    
        Take the daily max wind speed at t-2, t-1, t, t+1, t+2
        
        Fit a cubic polynomial, minimising RMSE with weights [0.1, 0.4, 1.0, 0.4, 0.1]
        - assuming each daily mean value is the value at 12h
        
        Assign the hourly wind speeds on day t from 0h - 23h according to this polynomial
        
For endpoints of the dataset, use linear or quadratic polynomials, depending on number of days available.

J Bland, 25/6/21 
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import shapely.geometry
import pandas as pd
import datetime
import iris
import iris.coord_categorisation
from eval_case_study_function2 import RMS_based_eval

iananmean = iris.analysis.Aggregator('nanmean', np.nanmean)


def downscaling(wind_daily_mean, nd):
    
    downscaled_days = np.zeros([nd,24])

    obs_mean = wind_daily_mean
    # extract permitted information: daily mean

    # This method requires 2 days either side, so won't work so well on shorter periods
    # Linear on endpoints
    # Quadratic one away from endpoints
    # Cubic in the middle

    downscaled_days[0, :12] = obs_mean[0]*np.ones(12)

    p = np.polyfit([-12, 12, 36], obs_mean[:3], 2)

    for h in np.arange(12,24):

        downscaled_days[0, h] = (h-24)**2*p[0] + (h-24)*p[1] + p[2]

    for h in np.arange(24):

        downscaled_days[1, h] = (h)**2*p[0] + (h)*p[1] + p[2]

    for i in np.arange(2,nd-2):

        p = np.polyfit([-36, -12, 12, 36, 60], obs_mean[i-2:i+3], 3, w = [.1, .4, 1, .4, .1])

        for h in np.arange(24):

            downscaled_days[i, h] = (h**3)*p[0] + (h**2)*p[1] + h*p[2] + p[3]

    p = np.polyfit([-12, 12, 36], obs_mean[-3:], 2)

    for h in np.arange(24):

        downscaled_days[-2, h] = (h)**2*p[0] + (h)*p[1] + p[2]

    for h in np.arange(13):

        downscaled_days[-1, h] = (h+24)**2*p[0] + (h+24)*p[1] + p[2]

    downscaled_days[-1, 13:] = obs_mean[-1]*np.ones(11)

    return np.reshape(downscaled_days, [nd*24])
    
    
def test_case(case_no, nd):

    field = 'speed10m'
    obs = np.loadtxt('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_' + field + '.dat')
    
    obs_days = np.reshape(np.array(obs), [nd, 24])
              
    obs_daily_mean = np.mean(obs_days, axis = 1)
    
    downscale_ts = downscaling(obs_daily_mean, nd)
    
    return downscale_ts
    
    
def main():
    
    downscale_ts = test_case('1', 7)
    RMS_based_eval(downscale_ts,'speed10m','1','polyfit')
    
    downscale_ts = test_case('2', 9)
    RMS_based_eval(downscale_ts,'speed10m','2','polyfit')
    
    downscale_ts = test_case('3', 28)
    RMS_based_eval(downscale_ts,'speed10m','3','polyfit')