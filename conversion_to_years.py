def conv2years_1979_2019_hourly(data_time_series):
    import numpy as np
    n = 8760  # no of days in a year
    k = 24
    years_array = np.zeros([41,8784])

# do the leap year properly! 366 days not 365. This should be looped over to improve presentation

    years_array[0,0:n] = data_time_series[0:n];   # 1979

    years_array[1,0:n+k] = data_time_series[n:2*n+k] #1980
    years_array[2,0:n] = data_time_series[2*n+k:3*n+k]
    years_array[3,0:n] = data_time_series[3*n+k:4*n+k]
    years_array[4,0:n] = data_time_series[4*n+k:5*n+k]
    
    years_array[5,0:n+k] = data_time_series[5*n+k:6*n+2*k]   #1984
    years_array[6,0:n] = data_time_series[6*n+2*k:7*n+2*k]
    years_array[7,0:n] = data_time_series[7*n+2*k:8*n+2*k]
    years_array[8,0:n] = data_time_series[8*n+2*k:9*n+2*k]
    
    years_array[9,0:n+k] = data_time_series[9*n+2*k:10*n+3*k]   # 1988
    years_array[10,0:n] = data_time_series[10*n+3*k:11*n+3*k]
    years_array[11,0:n] = data_time_series[11*n+3*k:12*n+3*k]
    years_array[12,0:n] = data_time_series[12*n+3*k:13*n+3*k]

    years_array[13,0:n+k] = data_time_series[13*n+3*k:14*n+4*k];   # 1992
    years_array[14,0:n] = data_time_series[14*n+4*k:15*n+4*k];
    years_array[15,0:n] = data_time_series[15*n+4*k:16*n+4*k];
    years_array[16,0:n] = data_time_series[16*n+4*k:17*n+4*k];

    years_array[17,0:n+k] = data_time_series[17*n+4*k:18*n+5*k];   # 1996
    years_array[18,0:n] = data_time_series[18*n+5*k:19*n+5*k];
    years_array[19,0:n] = data_time_series[19*n+5*k:20*n+5*k];
    years_array[20,0:n] = data_time_series[20*n+5*k:21*n+5*k];

    years_array[21,0:n+k]= data_time_series[21*n+5*k:22*n+6*k];   # 2000
    years_array[22,0:n] = data_time_series[22*n+6*k:23*n+6*k];
    years_array[23,0:n] = data_time_series[23*n+6*k:24*n+6*k];
    years_array[24,0:n] = data_time_series[24*n+6*k:25*n+6*k];

    years_array[25,0:n+k] = data_time_series[25*n+6*k:26*n+7*k];   # 2004
    years_array[26,0:n] = data_time_series[26*n+7*k:27*n+7*k];
    years_array[27,0:n] = data_time_series[27*n+7*k:28*n+7*k];
    years_array[28,0:n] = data_time_series[28*n+7*k:29*n+7*k];

    years_array[29,0:n+k] = data_time_series[29*n+7*k:30*n+8*k];   # 2008
    years_array[30,0:n] = data_time_series[30*n+8*k:31*n+8*k];
    years_array[31,0:n]= data_time_series[31*n+8*k:32*n+8*k];
    years_array[32,0:n] = data_time_series[32*n+8*k:33*n+8*k];

    years_array[33,0:n+k] = data_time_series[33*n+8*k:34*n+9*k];   # 2012
    years_array[34,0:n] = data_time_series[34*n+9*k:35*n+9*k];   # 2013
    years_array[35,0:n] = data_time_series[35*n+9*k:36*n+9*k];   # 2014
    years_array[36,0:n] = data_time_series[36*n+9*k:37*n+9*k];   # 2015

    years_array[37,0:n+k] = data_time_series[37*n+9*k:38*n+10*k];   # 2016
    years_array[38,0:n] = data_time_series[38*n+10*k:39*n+10*k];   # 2017
    years_array[39,0:n] = data_time_series[39*n+10*k:40*n+10*k];   # 2018
    years_array[40,0:n] = data_time_series[40*n+10*k:41*n+10*k];   # 2019
 
    # turn zeros to nans in missing points
    years_array[years_array ==0.] = np.nan

    return years_array
