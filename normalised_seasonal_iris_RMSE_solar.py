#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import shapely.geometry
import pandas as pd
import datetime
import iris
import iris.coord_categorisation

iananmean = iris.analysis.Aggregator('nanmean', np.nanmean)


# In[9]:


solar_all = iris.load_cube('UK_mean_timeseries_solar_1979_2016.nc')

daycoord = solar_all[::24].coord('time')
hourcoord = iris.coords.DimCoord(np.arange(24), long_name='hour_of_day', units='hours', circular=True)


# In[10]:


solar_reshape = iris.cube.Cube(solar_all.data.reshape([len(daycoord.points), len(hourcoord.points)]), 
                             long_name=solar_all.name(), units=solar_all.units, dim_coords_and_dims=[(daycoord, 0), (hourcoord, 1)])


# In[11]:


iris.coord_categorisation.add_day_of_year(solar_reshape, 'time')
iris.coord_categorisation.add_year(solar_reshape, 'time')


# In[12]:


print(solar_reshape)


# In[13]:


solar_mean = solar_reshape.collapsed('hour_of_day', iananmean)

solar_normalised = solar_reshape/solar_mean

#solar_normalised has mean1


# In[14]:


solar_seasonal = solar_normalised.aggregated_by(['day_of_year'], iananmean)


# In[15]:


print(solar_seasonal)
import iris.quickplot as qplt
for i in np.arange(1, 360, 55):
    qplt.plot(solar_seasonal[i])
plt.show()


# In[16]:


case_no = '1'
field = 'rsds'
obs = np.loadtxt('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_' + field + '.dat')
obs_date = np.load('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_date.npy')


# In[17]:


# Case 1: 7 days

obs_days = np.reshape(np.array(obs), [7, 24])
obs_dates = np.reshape(obs_date, [7, 24])

monthoffset = [0, 1, -1, 0, 0, 1, 1, 2, 3, 3, 4, 4]
# (monthno-1)*30 + monthoffset + date = day

downscaled_days = np.zeros_like(obs_days)
for i in np.arange(7):
    
    Smean = np.mean(obs_days[i, :])
    # extract permitted information: daily mean
    
    month = int(str(obs_dates[i, 0])[5:7])
    date = int(str(obs_dates[i, 0])[8:10])
    day = (month-1)*30 + monthoffset[month-1] + date
    
    downscaled_days[i, :] = Smean * solar_seasonal.data[day, :]
    # solar_seasonal is the mean diurnal cycle over each day of the calendar year for 40 years, with mean 1


# In[18]:


#from eval_case_study_function2 import RMS_based_eval
#from eval_case_study_function import MAE_based_eval


# In[19]:


downscale_ts1 = np.reshape(downscaled_days, [168])
#RMS_based_eval(downscale_ts,'rsds','1','normalised_mean')


# In[20]:


case_no = '2'
field = 'rsds'
obs = np.loadtxt('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_' + field + '.dat')
obs_date = np.load('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_date.npy')


# In[21]:


# Case 2: 9 days
nd = 9

obs_days = np.reshape(np.array(obs), [nd, 24])
obs_dates = np.reshape(obs_date, [nd, 24])

monthoffset = [0, 1, -1, 0, 0, 1, 1, 2, 3, 3, 4, 4]
# (monthno-1)*30 + monthoffset + date = day

downscaled_days2 = np.zeros_like(obs_days)
for i in np.arange(nd):
    
    Smean = np.mean(obs_days[i, :])
    # extract permitted information: daily mean
    
    month = int(str(obs_dates[i, 0])[5:7])
    date = int(str(obs_dates[i, 0])[8:10])
    day = (month-1)*30 + monthoffset[month-1] + date
    
    downscaled_days2[i, :] = Smean * solar_seasonal.data[day, :]
    # solar_seasonal is the mean diurnal cycle over each day of the calendar year for 40 years, with mean 1


# In[22]:


downscale_ts2 = np.reshape(downscaled_days2, [216])
#RMS_based_eval(downscale_ts,'rsds','2','normalised_mean')
#print(MAE)
#print(RMS)


# In[23]:


case_no = '3'
field = 'rsds'
obs = np.loadtxt('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_' + field + '.dat')
obs_date = np.load('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/case_studies/Case_' + case_no + '_date.npy')


# In[24]:


# Case 3: 28 days
nd = 28

obs_days = np.reshape(np.array(obs), [nd, 24])
obs_dates = np.reshape(obs_date, [nd, 24])

monthoffset = [0, 1, -1, 0, 0, 1, 1, 2, 3, 3, 4, 4]
# (monthno-1)*30 + monthoffset + date = day

downscaled_days3 = np.zeros_like(obs_days)
for i in np.arange(nd):
    
    Smean = np.mean(obs_days[i, :])
    # extract permitted information: daily mean
    
    month = int(str(obs_dates[i, 0])[5:7])
    date = int(str(obs_dates[i, 0])[8:10])
    day = (month-1)*30 + monthoffset[month-1] + date
    
    downscaled_days3[i, :] = Smean * solar_seasonal.data[day, :]
    # solar_seasonal is the mean diurnal cycle over each day of the calendar year for 40 years, with mean 1


# In[25]:


downscale_ts3 = np.reshape(downscaled_days3, [672])
#MAE, RMS = 
#RMS_based_eval(downscale_ts,'rsds','3','normalised_mean')
#print(MAE)
#print(RMS)


# In[26]:


def return_ssrd_jake():
    return(downscale_ts1, downscale_ts2, downscale_ts3)


# In[ ]:




