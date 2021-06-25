"""
This method, in a nutshell:

Finds the mean diurnal cycle (mdc(d)) over each day (d) of the calendar year for 40 years, with mean 1

For each target date (t), we know the daily solar mean, and so can take solar(t) = solar_mean(t) * mdc(d(t)), 
where d(t) is the calendar day of date t

For full explanation of method see "Explanation of simple downscaling methods"

The hourly training data 'UK_mean_timeseries_solar_1979_2016.nc' can be generated using functions in the file "iris_national_average.py" and hourly gridded ERA5 fields in a domain containing the UK.

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


def training():
    
    solar_all = iris.load_cube('UK_mean_timeseries_solar_1979_2016.nc')
    # load historical solar data

    daycoord = solar_all[::24].coord('time')
    hourcoord = iris.coords.DimCoord(np.arange(24), long_name='hour_of_day', units='hours', circular=True)
    solar_reshape = iris.cube.Cube(solar_all.data.reshape([len(daycoord.points), len(hourcoord.points)]), 
                             long_name=solar_all.name(), units=solar_all.units, dim_coords_and_dims=[(daycoord, 0), (hourcoord, 1)])
    # Reshape data into days * hours
    
    iris.coord_categorisation.add_day_of_year(solar_reshape, 'time')
    
    solar_mean = solar_reshape.collapsed('hour_of_day', iananmean)
    solar_normalised = solar_reshape/solar_mean
    # each day in solar_normalised has mean 1
    
    solar_seasonal = solar_normalised.aggregated_by(['day_of_year'], iananmean)
    # this method works least well for 29th February, with 1/4 the data - would be better to aggregate by week?
    
    return solar_seasonal



def downscaling(solar_seasonal, solar_daily_mean, nd, obs_dates):
    
    monthoffset = [0, 1, -1, 0, 0, 1, 1, 2, 3, 3, 4, 4]
    # (monthno-1)*30 + monthoffset + date = day

    downscaled_days = np.zeros([nd, 24])
    for i in np.arange(nd):

        Smean = solar_daily_mean[i]
        # extract permitted information: daily mean

        month = int(str(obs_dates[i, 0])[5:7])
        date = int(str(obs_dates[i, 0])[8:10])
        day = (month-1)*30 + monthoffset[month-1] + date

        downscaled_days[i, :] = Smean * solar_seasonal.data[day, :]
        
    return np.reshape(downscaled_days, [nd*24])



def test_case(case_no, nd, solar_seasonal):

    field = 'rsds'
    obs = np.loadtxt('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_' + field + '.dat')
    obs_date = np.load('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_date.npy')
    
    obs_days = np.reshape(np.array(obs), [nd, 24])
    obs_dates = np.reshape(obs_date, [nd, 24])
              
    obs_daily_mean = np.mean(obs_days, axis = 1)
    
    downscale_ts = downscaling(solar_seasonal, obs_daily_mean, nd, obs_dates)
    
    return downscale_ts
    
    
def main():
    
    solar_seasonal = training()
    
    downscale_ts =test_case('1', 7, solar_seasonal)
    RMS_based_eval(downscale_ts,'rsds','1','normalised_mean')
    
    downscale_ts =test_case('2', 9, solar_seasonal)
    RMS_based_eval(downscale_ts,'rsds','2','normalised_mean')
    
    downscale_ts =test_case('3', 28, solar_seasonal)
    RMS_based_eval(downscale_ts,'rsds','3','normalised_mean')
    
    