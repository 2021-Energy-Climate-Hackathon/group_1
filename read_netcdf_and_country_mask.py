########
#
# in this script we will read in some ERA5 data and then country mask it.
#
#
# This script is a simple starter script, it can be adapted to read in
# multiple years of data or to read in different fields. 
# 
# Other functions are also available to load the data in the libraries: 
# - cfpython 
# - xarray
# - iris
#
#########

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import shapely.geometry


def load_country_mask(COUNTRY,data_dir,filename,nc_key):

    '''

    This function loads the country masks for the ERA5 data grid we have been using

    Args:
        COUNTRY (str): This must be a name of a country (or set of) e.g. 
            'United Kingdom','France','Czech Republic'
 
       data_dir (str): The parth for where the data is stored.
            e.g '/home/users/zd907959/'

        filename (str): The filename of a .netcdf file
            e.g. 'ERA5_1979_01.nc'

        nc_key (str): The string you need to load the .nc data 
            e.g. 't2m','rsds'

    Returns:
       MASK_MATRIX_RESHAPE (array): Dimensions [lat,lon] where there are 1's if 
           the data is within a country border and zeros if data is outside a 
           country border. 


    '''


    # first loop through the countries and extract the appropraite shapefile
    countries_shp = shpreader.natural_earth(resolution='10m',category='cultural',
                                            name='admin_0_countries')
    country_shapely = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes['NAME_LONG'] == COUNTRY:
            print('Found country')
            country_shapely.append(country.geometry)

    # load in the data you wish to mask
    file_str = data_dir + filename
    dataset = Dataset(file_str,mode='r')
    lons = dataset.variables['longitude'][:]
    lats = dataset.variables['latitude'][:]
    data = dataset.variables[nc_key][:] # data in shape [time,lat,lon]
    dataset.close()

    # get data in appropriate units for models
    if nc_key == 't2m':
        data = data-273.15 # convert to Kelvin from Celsius
    if nc_key == 'ssrd':
        data = data/3600. # convert Jh-1m-2 to Wm-2

    LONS, LATS = np.meshgrid(lons,lats) # make grids of the lat and lon data
    x, y = LONS.flatten(), LATS.flatten() # flatten these to make it easier to 
    #loop over.
    points = np.vstack((x,y)).T
    MASK_MATRIX = np.zeros((len(x),1))
    # loop through all the lat/lon combinations to get the masked points
    for i in range(0,len(x)):
        my_point = shapely.geometry.Point(x[i],y[i]) 
        if country_shapely[0].contains(my_point) == True: 
            MASK_MATRIX[i,0] = 1.0 # creates 1s and 0s where the country is
    
    MASK_MATRIX_RESHAPE = np.reshape(MASK_MATRIX,(len(lats),len(lons)))


    return(MASK_MATRIX_RESHAPE)


def load_10mwindspeed_data(data_dir,filename):

    """
    This function takes the ERA5 reanalysis data, loads it and applied a 
    country mask (ready for conversion to energy) it then returns
    the array (of original size) with all irrelvelant gridpoints 
    set to zeros.

    You will need the shpreader.natural_earth data downloaded 
    to find the shapefiles.

    Args:

        data_dir (str): The parth for where the data is stored.
            e.g '/home/users/zd907959/'

        filename (str): The filename of a .netcdf file
            e.g. 'ERA5_1979_01.nc'

    Returns:

        wind_speed_data (array): 10m wind speed data, dimensions 
            [time,lat,lon].

        latitudes (array): array of latitudes

        longitudes (array): array of longitudes



    """

  
    # load in the data you wish to mask
    file_str = data_dir + filename
    dataset = Dataset(file_str,mode='r')
    lons = dataset.variables['longitude'][:]
    lats = dataset.variables['latitude'][:]
    data1 = dataset.variables['u10'][:] # data in shape [time,lat,lon]
    data2 = dataset.variables['v10'][:] # data in shape [time,lat,lon]
    dataset.close()

    wind_speed_data = np.sqrt(data1*data1 + data2*data2)

    return(wind_speed_data,lats,lons)



country_mask = load_country_mask('United Kingdom','/gws/pw/j05/cop26_hackathons/oxford/Data/ERA5_data_EU_domain/field_set_1/','ERA5_1hr_field_set_1_2018_01.nc','t2m')
plt.imshow(country_mask)
plt.show()



speed10m_data,lats,lons =load_10mwindspeed_data('/gws/pw/j05/cop26_hackathons/oxford/Data/ERA5_data_EU_domain/field_set_1/','ERA5_1hr_field_set_1_2018_01.nc')


gridded_lons, gridded_lats = np.meshgrid(lons,lats)
plt.pcolor(gridded_lons,gridded_lats,speed10m_data[0,:,:])
plt.show()


# country mask the data
country_masked_data = np.zeros(np.shape(speed10m_data))
for i in range(0,len(country_masked_data)):
    country_masked_data[i,:,:] = speed10m_data[i,:,:]*MASK_MATRIX_RESHAPE



gridded_lons, gridded_lats = np.meshgrid(lons,lats)
plt.pcolor(gridded_lons,gridded_lats,country_mask_data[0,:,:])
plt.show()


# to make this a national timeseries average over the existing country points.

# replace zeros with nans
country_mask_data[country_mask_data == 0.] = np.nan

# spatially average
country_timeseries = np.nanmean(np.nanmean(country_mask_data,axis=2),axis=1)

plt.plot(country_timeseries)
plt.show()
