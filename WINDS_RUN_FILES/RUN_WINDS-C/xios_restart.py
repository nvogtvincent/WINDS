#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to reformat XIOS output to be used as a restart file by CROCO
Noam Vogt-Vincent 2020 with snippets from
stackoverflow.com/questions/27349196/
"""

from netCDF4 import Dataset
import numpy as np
import sys
import os

##############################################################################
# File locations #############################################################
##############################################################################

# root_dir = ('/home/noam/Documents/Scripts/')
output_dir = os.path.dirname(os.path.realpath(__file__))

input_file = output_dir + '/' + 'XIOS_RST.nc'  # The file generated by XIOS
output_file = output_dir + '/' + 'xios_ini.nc'  # The correct croco_ini.nc file

# Use provided argument as time-step (LAST time-step from previous run) or
# otherwise use time-step in script
try:
    ts = int(sys.argv[1])
    print('First time-step set to ' + str(ts))
    ts += 1
except (TypeError, IndexError):
    print('No time-step provided!')
    ts = 0
    print('Setting first time-step to 0')

# XIOS output file
with Dataset(input_file, mode='r') as nc:
    print('Loading fields from input netcdf...')
    # Load the time from the XIOS output (the scrum_time in the new file)
#    scrum_time = np.float64(nc.variables['time'][-1])
    scrum_time = np.float64(3600*24*365*14)
    time_step = np.int32([ts, 1, 1, 0])

    # Load the other variables (last time slice by default)
    salt = np.float64(nc.variables['salt'][-1, :, :, :])
    salt = salt[None, :, :, :]
    temp = np.float64(nc.variables['temp'][-1, :, :, :])
    temp = temp[None, :, :, :]

    u = np.float64(nc.variables['u'][-1, :, :, :])
    u = u[None, :, :, :]
    ubar = np.float64(nc.variables['ubar'][-1, :, :])
    ubar = ubar[None, :, :]

    v = np.float64(nc.variables['v'][-1, :, :, :])
    v = v[None, :, :, :]
    vbar = np.float64(nc.variables['vbar'][-1, :, :])
    vbar = vbar[None, :, :]

    zeta = np.float64(nc.variables['zeta'][-1, :, :])
    zeta = zeta[None, :, :]


# Create the new initial file
with Dataset(output_file, mode='w') as nc:
    print('Writing fields to output netcdf...')
    # Create the dimensions
    # auxil = nc.createDimension('auxil', 4)
    # time = nc.createDimension('time', 1)
    nc.createDimension('auxil', 4)
    nc.createDimension('time', 1)
    nc.createDimension('s_rho', np.shape(salt)[1])
    nc.createDimension('eta_rho', np.shape(salt)[2])
    nc.createDimension('eta_v', np.shape(v)[2])
    nc.createDimension('xi_rho', np.shape(salt)[3])
    nc.createDimension('xi_u', np.shape(u)[3])

    # Create the time variables
    nc.createVariable('scrum_time', 'f8', ('time'), zlib=True)
    nc.variables['scrum_time'].long_name = 'time since initialization'
    nc.variables['scrum_time'].units = 'second'
    nc.variables['scrum_time'].field = 'time, scalar, series'
    nc.variables['scrum_time'].standard_name = 'time'
    nc.variables['scrum_time'].axis = 'T'

    nc.createVariable('time', 'f8', ('time'), zlib=True)
    nc.variables['time'].long_name = 'time since initialization'
    nc.variables['time'].units = 'second'
    nc.variables['time'].field = 'time, scalar, series'
    nc.variables['time'].standard_name = 'time'
    nc.variables['time'].axis = 'T'

    nc.createVariable('time_step', 'i4', ('time', 'auxil'), zlib=True)
    nc.variables['time_step'].long_name = ('time step and record numbers from '
                                           'initialisation')

    # Create the other variables
    nc.createVariable('salt', 'f8', ('time', 's_rho', 'eta_rho', 'xi_rho'))
    nc.variables['salt'].long_name = 'salinity'
    nc.variables['salt'].units = 'PSU'
    nc.variables['salt'].field = 'salinity, scalar, series'
    nc.variables['salt'].standard_name = 'sea_water_salinity'
    nc.variables['salt'].coordinates = 'lat_rho lon_rho'

    nc.createVariable('temp', 'f8', ('time', 's_rho', 'eta_rho', 'xi_rho'))
    nc.variables['temp'].long_name = 'potential temperature'
    nc.variables['temp'].units = 'Celsius'
    nc.variables['temp'].field = 'temperature, scalar, series'
    nc.variables['temp'].standard_name = 'sea_water_potential_temperature'
    nc.variables['temp'].coordinates = 'lat_rho lon_rho'

    nc.createVariable('u', 'f8', ('time', 's_rho', 'eta_rho', 'xi_u'))
    nc.variables['u'].long_name = 'u-momentum component'
    nc.variables['u'].units = 'meter second-1'
    nc.variables['u'].field = 'u-velocity, scalar, series'
    nc.variables['u'].standard_name = 'sea_water_x_velocity_at_u_location'
    nc.variables['u'].coordinates = 'lat_u lon_u'

    nc.createVariable('ubar', 'f8', ('time', 'eta_rho', 'xi_u'))
    nc.variables['ubar'].long_name = ('vertically integrated u-momentum '
                                      'component')
    nc.variables['ubar'].units = 'meter second-1'
    nc.variables['ubar'].field = 'ubar-velocity, scalar, series'
    nc.variables['ubar'].standard_name = ('barotropic_sea_water_x_velocity_'
                                          'at_u_location')
    nc.variables['ubar'].coordinates = 'lat_u lon_u'

    nc.createVariable('v', 'f8', ('time', 's_rho', 'eta_v', 'xi_rho'))
    nc.variables['v'].long_name = 'v-momentum component'
    nc.variables['v'].units = 'meter second-1'
    nc.variables['v'].field = 'v-velocity, scalar, series'
    nc.variables['v'].standard_name = 'sea_water_y_velocity_at_v_location'
    nc.variables['v'].coordinates = 'lat_v lon_v'

    nc.createVariable('vbar', 'f8', ('time', 'eta_v', 'xi_rho'))
    nc.variables['vbar'].long_name = ('vertically integrated v-momentum '
                                      'component')
    nc.variables['vbar'].units = 'meter second-1'
    nc.variables['vbar'].field = 'vbar-velocity, scalar, series'
    nc.variables['vbar'].standard_name = ('barotropic_sea_water_y_velocity_'
                                          'at_v_location')
    nc.variables['vbar'].coordinates = 'lat_v lon_v'

    nc.createVariable('zeta', 'f8', ('time', 'eta_rho', 'xi_rho'))
    nc.variables['zeta'].long_name = 'free-surface'
    nc.variables['zeta'].units = 'meter'
    nc.variables['zeta'].field = 'free-surface, scalar, series'
    nc.variables['zeta'].standard_name = 'sea_surface_height'
    nc.variables['zeta'].coordinates = 'lat_rho lon_rho'

    # Fill variables
    nc.variables['scrum_time'][:] = scrum_time
    nc.variables['time'][:] = np.nan
    nc.variables['time_step'][:] = time_step

    nc.variables['salt'][:] = salt
    nc.variables['temp'][:] = temp
    nc.variables['u'][:] = u
    nc.variables['ubar'][:] = ubar
    nc.variables['v'][:] = v
    nc.variables['vbar'][:] = vbar
    nc.variables['zeta'][:] = zeta

    print('Completed!')
