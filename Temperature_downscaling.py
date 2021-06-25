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
import eval_case_study_function2

iananmean = iris.analysis.Aggregator('nanmean', np.nanmean)


def training():
    
    t2m_all = iris.load_cube('UK_mean_timeseries_t2m_1979_2016.nc')

    daycoord = t2m_all[::24].coord('time')
    hourcoord = iris.coords.DimCoord(np.arange(24), long_name='hour_of_day', units='hours', circular=True)
    
    t2m_reshape = iris.cube.Cube(t2m_all.data.reshape([len(daycoord.points), len(hourcoord.points)]), 
                             long_name=t2m_all.name(), units=t2m_all.units, dim_coords_and_dims=[(daycoord, 0), (hourcoord, 1)])
    
    iris.coord_categorisation.add_day_of_year(t2m_reshape, 'time')
    iris.coord_categorisation.add_year(t2m_reshape, 'time')
    
    t2m_mean = t2m_reshape.collapsed('hour_of_day', iananmean)
    t2m_max = t2m_reshape.collapsed('hour_of_day', iris.analysis.MAX)
    t2m_min = t2m_reshape.collapsed('hour_of_day', iris.analysis.MIN)
    t2m_range = t2m_max - t2m_min

    t2m_normalised = (t2m_reshape - t2m_mean)/t2m_range
    #each day in t2m_normalised has mean zero, and range 1
    
    t2m_seasonal = t2m_normalised.aggregated_by(['day_of_year'], iananmean)
    
    return t2m_seasonal



def downscaling_basic(t2m_seasonal, t2m_daily_mean, t2m_daily_min, t2m_daily_max, nd, obs_dates):
    
    monthoffset = [0, 1, -1, 0, 0, 1, 1, 2, 3, 3, 4, 4]
    # (monthno-1)*30 + monthoffset + date = day

    downscaled_days = np.zeros([nd, 24])
    for i in np.arange(nd):

        Tmean = t2m_daily_mean[i]
        Tmax = t2m_daily_min[i]
        Tmin = t2m_daily_max[i]
        Trange = Tmax - Tmin
        # extract permitted information: daily mean, daily max, daily min

        month = int(str(obs_dates[i, 0])[5:7])
        date = int(str(obs_dates[i, 0])[8:10])
        day = (month-1)*30 + monthoffset[month-1] + date

        downscaled_days[i, :] = Tmean + Trange*t2m_seasonal.data[day, :]
        # t2m_seasonal is the mean diurnal cycle over each day of the calendar year for 40 years, with mean zero, and range 1
    
    return downscaled_days
        
    
    
    
def consecutive_day_matching(downscaled_days, t2m_daily_mean, t2m_daily_min, nd):
    
    # Now we extend this to use the minimum temperatures from the days before (morning, M) and after (night, N)
    # but not for the endpoints of the obs days, because I'm too lazy to go get the data from a broader period
    for i in np.arange(nd):

        if i != 0:

            TminM = t2m_daily_min[i-1]
            # information from climate model

            TmaxD = np.max(downscaled_days[i, :])
            argmax = np.argmax(downscaled_days[i, :])
            # information from our first guess

            if downscaled_days[i, 0] < TminM:
                # if we've said it's colder at midnight00 than yesterday's minumum
                downscaled_days[i, :argmax+1] = TmaxD + ( (downscaled_days[i, :argmax+1] - TmaxD) * 
                                                         ( ( TmaxD - TminM ) / (TmaxD - downscaled_days[i, 0] )) )
                # then re-scale the first half of the sinusoid such that now the midnight00 value is yesterday's minimum, 
                # keeping today's maximum the same

        if i != nd-1:

            TminN = t2m_daily_min[i+1]
            # information from climate model

            TmaxD = np.max(downscaled_days[i, :])
            argmax = np.argmax(downscaled_days[i, :])
            # information from our first guess

            if downscaled_days[i, -1] < TminN:
                # if we've said it's colder at midnight23 than tomorrow's minumum
                downscaled_days[i, argmax:] = TmaxD + ( (downscaled_days[i, argmax:] - TmaxD) * 
                                                         ( ( TmaxD - TminN ) / (TmaxD - downscaled_days[i, -1] )) )
                # then re-scale the second half of the sinusoid such that now the midnight23 value is tomorrow's minimum, 
                # keeping today's maximum the same
                
    # Now smooth out large disagreement between consecutive days

    for i in np.arange(1, nd):

        if (downscaled_days[i, 0] - downscaled_days[i-1, -1]) > 1.5*np.max([np.abs(downscaled_days[i, 1] - downscaled_days[i, 0]), 
                                                                                  np.abs(downscaled_days[i-1, -2] - downscaled_days[i-1, -1])]):
            # if the difference in one hour between two cases is larger than the difference expected from an nour and a half, it's too large

            # assume the day before is fine (meritless assumption)

            TminD = np.min(downscaled_days[i, :argmax])
            argmin = np.argmin(downscaled_days[i, :argmax])
            # information from our first guess

            downscaled_days[i, :argmin+1] = TminD + ( (downscaled_days[i, :argmin+1] - TminD) * 
                                                         ( ( TminD - downscaled_days[i-1, -1] ) / (TminD - downscaled_days[i, 0] )) )
            # re-scale the morning

        Tmean = t2m_daily_mean[i]
        # use the known daily mean for i-1 again

        downscaled_days[i, :] = downscaled_days[i, :] * (Tmean / np.mean(downscaled_days[i, :]))
        # bring daily mean back to known value if the adjustment has changed this
        
    return downscaled_days



def test_case(case_no, nd, t2m_seasonal):

    field = 'T2m'
    obs = np.loadtxt('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_' + field + '.dat')
    obs_date = np.load('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_date.npy')
    
    obs_days = np.reshape(np.array(obs), [nd, 24])
    obs_dates = np.reshape(obs_date, [nd, 24])
              
    obs_daily_mean = np.mean(obs_days, axis = 1)
    obs_daily_max = np.max(obs_days, axis = 1)
    obs_daily_min = np.min(obs_days, axis = 1)
    
    downscale_ts = downscaling_basic(t2m_seasonal, obs_daily_mean, obs_daily_min, obs_daily_max, nd, obs_dates)
    downscale_ts_improved = consecutive_day_matching(downscaled_days, obs_daily_mean, obs_daily_min, nd)
    
    return np.reshape(downscale_ts, [nd*24]), np.reshape(downscale_ts_improved, [nd*24])



def main():
    
    t2m_seasonal = training()
    
    downscale_ts, downscale_ts_improved = test_case('1', 7, t2m_seasonal)
    RMS_based_eval(downscale_ts,'T2m','1','normalised_mean')
    RMS_based_eval(downscale_ts_improved,'T2m','1','normalised_mean')
    
    downscale_ts, downscale_ts_improved = test_case('2', 9, t2m_seasonal)
    RMS_based_eval(downscale_ts,'T2m','2','normalised_mean')
    RMS_based_eval(downscale_ts_improved,'T2m','2','normalised_mean')
    
    downscale_ts, downscale_ts_improved = test_case('3', 28, t2m_seasonal)
    RMS_based_eval(downscale_ts,'T2m','3','normalised_mean')
    RMS_based_eval(downscale_ts_improved,'T2m','3','normalised_mean')