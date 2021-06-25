import iris
import numpy as np
from eofs.standard import Eof
import pandas as pd
from datetime import datetime

n = 14

cubelist = iris.load('/gws/pw/j05/cop26_hackathons/oxford/Group_folders/group_1/Data/European_MSLP_and_MSLP_anomaly_data.nc')

MSLP_anom = cubelist.extract_strict(iris.Constraint(name = 'MSLP_anom'))
reshaped_MSLP_anom = np.reshape(MSLP_anom.data, (len(MSLP_anom.coord('year').points)*len(MSLP_anom.coord('day').points),
                                                 len(MSLP_anom.coord('lat').points)*len(MSLP_anom.coord('lon').points)))

print('solving for EOFs')

solver = Eof(reshaped_MSLP_anom)
# this is slow

print('extracting Principal components')

pcNEOFS = solver.pcs(npcs=n)
# shape (year*day, n)

print('we have PCs')
print(np.shape(pcNEOFS))

eofs = solver.eofs(neofs=n)

eof_latlon = np.reshape(eofs, (n, len(MSLP_anom.coord('lat').points), len(MSLP_anom.coord('lon').points)))
# shape (n, lat, lon)

varfrac = solver.varianceFraction(neigs=n)
# shape (n)

pc_dic = {}
pc_cubelist = iris.cube.CubeList([])
eof_cubelist = iris.cube.CubeList([])

print('creating data objects, EOF:')

for i in np.arange(n):
    
    print(n)
    
    pc_dic['mslp_pc_' + str(i) + '_daily'] = pcNEOFS[:, i]
    
    pc_cube = iris.cube.Cube(np.reshape(pcNEOFS[:, i], (len(MSLP_anom.coord('year').points), len(MSLP_anom.coord('day').points))),
                             long_name = 'mslp_principal_component_' + str(i), 
                             dim_coords_and_dims = [(MSLP_anom.coord('year'), 0), (MSLP_anom.coord('day'), 1)])
    
    pc_cubelist.append(pc_cube)
    
    eof_cube = iris.cube.Cube(eof_latlon[i], long_name = 'MSLP_EOF_' + str(i), 
                              dim_coords_and_dims = [(MSLP_anom.coord('lat'), 0), (MSLP_anom.coord('lon'), 1)])
    
    eof_cubelist.append(eof_cube)
    
varfrac_cube = iris.cube.Cube(varfrac, long_name = 'Variance_fraction_for_EOFs')

print('saving as netcdf')

iris.save(pc_cubelist, 'MSLP_principal_component_daily_timeseries.nc')
iris.save(eof_cubelist, 'MSLP_EOFs.nc')
iris.save(varfrac_cube, 'MSLP_EOF_variance_fraction.nc')

print('saving as csv')

df = pd.DataFrame(pc_dic)
days = pd.date_range(datetime(1979,1,1,0), datetime(2020,12,31,0))
ERA5_days = days[days.strftime("%m%d") != "0229"]
df.index = ERA5_days

df.to_csv("ERA5_MSLP_anom_EOF_principal_component_daily.csv")
    