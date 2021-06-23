import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
import shapely.geometry
import iris
import iris.coord_categorisation as coordcat
import iris.analysis.cartography
import numpy.ma as ma

def national_average_netcdf(years, country_mask, varname = 't2m_1979_2016', variable = '2 metre temperature', 
                            data_dir = '/gws/pw/j05/cop26_hackathons/oxford/Data/ERA5_data_EU_domain/field_set_1/',
                            filename = 'ERA5_1hr_field_set_1_', area_weights = False):
    """
    Load ERA5 hourly data for all years in years, for single given variable (easily extended to multiple)
    Remove file history
    Mask the UK 
    Take area nanmean (not latitude weighted, but should be)
    Add additional coordinates of:
        hour of day, day of year, and year, in case useful
    Concatenate all years into single object
    Save
    
    Note: the area weighting functionality has not been checked/debugged, bit unsure how iris handles masked arrays
    If in doubt, keep False
    """
    
    
    iananmean = iris.analysis.Aggregator('nanmean', np.nanmean)
    
    all_cubelist = iris.cube.CubeList([])
    
    for year in years:
        
        if variable == '10 metre wind':
            
            U = iris.load(data_dir + filename + str(int(year)) + '*.nc', '10 metre U wind component')
            V = iris.load(data_dir + filename + str(int(year)) + '*.nc', '10 metre V wind component')
            
            #print(iris.load(data_dir + filename + str(int(year)) + '*.nc'))
            
            for cube in U:
                cube.attributes.pop('history')
                # this is required to merge into a single cube
            for cube in V:
                cube.attributes.pop('history')
                # this is required to merge into a single cube

            U_cube = U.concatenate_cube()
            V_cube = V.concatenate_cube()
            
            var_cube = U_cube.copy()
            var_cube.rename(variable)
            
            var_cube.data = np.sqrt(U_cube.data**2 + V_cube.data**2)
            
        else:
        
            cubelist = iris.load(data_dir + filename + str(int(year)) + '*.nc', variable)
        
            for cube in cubelist:
                cube.attributes.pop('history')
                # this is required to merge into a single cube

            var_cube = cubelist.concatenate_cube()
        
        var_mask = var_cube.copy()
        
        if area_weights:
        
            var_mask.data = ma.masked_array(var_mask, mask = country_mask)

            var_mask.coord('latitude').guess_bounds()
            var_mask.coord('longitude').guess_bounds()
            grid_areas = iris.analysis.cartography.area_weights(var_mask)

            var_national = var_mask.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
            
        else:
        
            var_mask.data = var_cube.data * country_mask
            var_national = var_mask.collapsed('latitude', iananmean).collapsed('longitude', iananmean)
        
        iris.coord_categorisation.add_hour(var_national, 'time')
        iris.coord_categorisation.add_day_of_year(var_national, 'time')
        iris.coord_categorisation.add_year(var_national, 'time')
        
        all_cubelist.append(var_national)
        
    long_timeseries = all_cubelist.concatenate_cube()
    
    #iris.save(long_timeseries, '/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/Data/UK_mean_timeseries_' + varname + '.nc')
    iris.save(long_timeseries, 'UK_mean_timeseries_' + varname + '.nc')
    #return long_timeseries
    
    
    
def create_national_average_T():
    
    data_dir = '/gws/pw/j05/cop26_hackathons/oxford/Data/ERA5_data_EU_domain/field_set_1/'
    filename = 'ERA5_1hr_field_set_1_'
    country_mask = load_country_mask('United Kingdom',data_dir,filename+'1979_01.nc','t2m')
    country_mask[country_mask == 0.] = np.nan
    years = np.arange(1979, 2017)

    national_average_netcdf(years, country_mask, varname = 'test')
    
    
    
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