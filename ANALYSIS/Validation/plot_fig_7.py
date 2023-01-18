#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to validate interannual surface current variability in WINDS
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmasher as cmr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from matplotlib.gridspec import GridSpec
from scipy import signal

###############################################################################
# PARAMETERS ##################################################################
###############################################################################

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../../'
dirs['fig'] = dirs['root'] + 'FIGURES/Validation/'
dirs['grid'] = dirs['root'] + 'REFERENCE/'
dirs['data'] = dirs['root'] + 'REFERENCE/Validation/'
dirs['script'] = dirs['root'] + 'ANALYSIS/Validation/'

# FILE-HANDLES
region_list = ['SECW', 'SECE', 'NMC', 'EMC', 'NWMC', 'SWMC', 'EACCN', 'EACCS', 'SECCW', 'SECCE']
text_list = ['South Equatorial Current (West)',
             'South Equatorial Current (East)',
             'North Madagascar Current',
             'East Madagascar Current',
             'NW Mozambique Channel',
             'SW Mozambique Channel',
             'East Africa Coastal Current (North)',
             'East Africa Coastal Current (South)',
             'South Equatorial Countercurrent (West)',
             'South Equatorial Countercurrent (East)']

text_dict = {item[0]: item[1] for item in zip(region_list, text_list)}

fh = {'winds': {}, 'altimetry': {}, 'cmems': {}}

for region in region_list:
    fh['winds'][region] = dirs['data'] + 'WINDS-M_' + region + '.nc'
    fh['altimetry'][region] = dirs['data'] + 'ALTIMETRY_' + region + '.nc'
    fh['cmems'][region] = dirs['data'] + 'CMEMS_OCEAN_' + region + '.nc'

fh['winds_mean'] = dirs['data'] + 'WINDS-M_SFC_mean.nc'

###############################################################################
# PROCESS DATA ################################################################
###############################################################################

# Methodology:
# Compute a monthly time-series, with the range from the daily values
data = {'winds': {}, 'altimetry': {}, 'cmems': {}}
data_lp = {'winds': {}, 'altimetry': {}, 'cmems': {}} # Low-pass filtered

order = 4
fs = 1        # (1/d)
cutoff = 1/480 # (1/d)
lp_filt = signal.butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')

for source in ['winds', 'altimetry', 'cmems']:
    for region in region_list:
        if source == 'winds':
            speed = ((xr.open_dataset(fh[source][region]).u_surf**2 +
                      xr.open_dataset(fh[source][region]).v_surf**2)**0.5)
            speed_lp = xr.zeros_like(speed)
            speed_lp.data = ((signal.sosfiltfilt(lp_filt, xr.open_dataset(fh[source][region]).u_surf, axis=0)**2 +
                              signal.sosfiltfilt(lp_filt, xr.open_dataset(fh[source][region]).v_surf, axis=0)**2)**0.5)
            speed = speed.mean(dim=('x_rho', 'y_rho'))[:9861].rename({'time_counter': 'time'})
            speed_lp = speed_lp.mean(dim=('x_rho', 'y_rho'))[:9861].rename({'time_counter': 'time'})

            speed = speed.assign_coords({'time': pd.date_range(start=datetime(year=1993, month=1, day=1),
                                                               end=datetime(year=2019, month=12, day=31),
                                                               freq='D')})
            speed_lp = speed_lp.assign_coords({'time': pd.date_range(start=datetime(year=1993, month=1, day=1),
                                                                     end=datetime(year=2019, month=12, day=31),
                                                                     freq='D')})

        elif source == 'altimetry' or source == 'cmems':
            speed = ((xr.open_dataset(fh[source][region]).uo**2 +
                      xr.open_dataset(fh[source][region]).vo**2)**0.5)
            speed_lp = xr.zeros_like(speed)
            speed_lp.data = ((signal.sosfiltfilt(lp_filt, xr.open_dataset(fh[source][region]).uo, axis=0)**2 +
                              signal.sosfiltfilt(lp_filt, xr.open_dataset(fh[source][region]).vo, axis=0)**2)**0.5)
            speed = speed.mean(dim=('longitude', 'latitude'))[:9861]
            speed_lp = speed_lp.mean(dim=('longitude', 'latitude'))[:9861]
            speed = speed.assign_coords({'time': pd.date_range(start=datetime(year=1993, month=1, day=1),
                                                               end=datetime(year=2019, month=12, day=31),
                                                               freq='D')}).drop('depth')
            speed_lp = speed_lp.assign_coords({'time': pd.date_range(start=datetime(year=1993, month=1, day=1),
                                                                     end=datetime(year=2019, month=12, day=31),
                                                                     freq='D')}).drop('depth')

        # Get monthly min, max, and mean
        data[source][region] = {}
        try:
            data[source][region]['min'] = speed.resample(time='1M').min(dim='time')
            data[source][region]['max'] = speed.resample(time='1M').max(dim='time')
            data[source][region]['mean'] = speed.resample(time='1M').mean(dim='time')
            data[source][region]['mean_lp'] = speed_lp.resample(time='1M').mean(dim='time')
            data[source][region]['mean_anom'] = (speed.groupby('time.month') - speed.groupby('time.month').mean('time')).resample(time='1M').mean(dim='time')

        except:
            print()

###############################################################################
# PLOT DATA ###################################################################
###############################################################################

# Firstly plot the monthly data
f = plt.figure(constrained_layout=True, figsize=(10, 14))
gs = GridSpec(10, 1, figure=f)
ax = []
for i in range(10):
    ax.append(f.add_subplot(gs[i, 0]))

for panel, region in enumerate(region_list):
    ax[panel].plot(data['winds'][region]['mean'].time,
                   data['winds'][region]['mean'].values,
                   'k-', linewidth=1, zorder=5, label='WINDS')
    ax[panel].fill_between(data['winds'][region]['mean'].time,
                           data['winds'][region]['min'].values,
                           data['winds'][region]['max'].values,
                           fc='k', alpha=0.2, zorder=1)

    ax[panel].plot(data['cmems'][region]['mean'].time,
                   data['cmems'][region]['mean'].values,
                   'r-', linewidth=1, zorder=4, alpha=0.5, label='CMEMS')

    ax[panel].plot(data['altimetry'][region]['mean'].time,
                   data['altimetry'][region]['mean'].values,
                   'b-', linewidth=1, zorder=3, alpha=0.5, label='GlobCurrent')

    ax[panel].set_xlim([datetime(year=1993, month=1, day=1),
                        datetime(year=2019, month=12, day=31)])
    ax[panel].set_ylabel('$\overline{v}_s$')

    for tick_year in [1996, 2000, 2004, 2008, 2012, 2016]:
        ax[panel].axvline(x=datetime(year=tick_year, month=1, day=1), color='k',
                          linestyle='--', linewidth=0.5, alpha=0.2, zorder=2)

    if panel != 9:
        ax[panel].set_xticks([])

    if panel == 0:
        legend = ax[panel].legend(ncol=3)
        legend.get_frame().set_alpha(0)

    ax[panel].spines.right.set_visible(False)
    ax[panel].spines.top.set_visible(False)

    ax[panel].set_title(text_dict[region], fontsize=10)

plt.savefig(dirs['fig'] + 'sfc_speed_time_series_comparison.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Now plot the low-pass filtered data
f = plt.figure(constrained_layout=True, figsize=(10, 14))
gs = GridSpec(10, 1, figure=f)
ax = []
for i in range(10):
    ax.append(f.add_subplot(gs[i, 0]))

for panel, region in enumerate(region_list):
    ax[panel].plot(data['winds'][region]['mean_lp'].time,
                   data['winds'][region]['mean_lp'].values - np.mean(data['winds'][region]['mean_lp'].values),
                   'k-', linewidth=1, zorder=5, label='WINDS')

    ax[panel].plot(data['cmems'][region]['mean_lp'].time,
                   data['cmems'][region]['mean_lp'].values - np.mean(data['cmems'][region]['mean_lp'].values),
                   'r-', linewidth=1, zorder=4, alpha=0.5, label='CMEMS')

    ax[panel].plot(data['altimetry'][region]['mean_lp'].time,
                   data['altimetry'][region]['mean_lp'].values - np.mean(data['altimetry'][region]['mean_lp'].values),
                   'b-', linewidth=1, zorder=3, alpha=0.5, label='GlobCurrent')

    ax[panel].set_xlim([datetime(year=1993, month=1, day=1),
                        datetime(year=2019, month=12, day=31)])
    ax[panel].set_ylabel('Low-pass $\overline{v}_s$\'')

    for tick_year in [1996, 2000, 2004, 2008, 2012, 2016]:
        ax[panel].axvline(x=datetime(year=tick_year, month=1, day=1), color='k',
                          linestyle='--', linewidth=0.5, alpha=0.2, zorder=2)

    if panel != 9:
        ax[panel].set_xticks([])

    if panel == 0:
        legend = ax[panel].legend(ncol=3)
        legend.get_frame().set_alpha(0)

    ax[panel].spines.right.set_visible(False)
    ax[panel].spines.top.set_visible(False)

    ax[panel].set_title(text_dict[region], fontsize=10)

plt.savefig(dirs['fig'] + 'sfc_speed_low_pass_time_series_comparison.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Now plot the monthly anomalies
f = plt.figure(constrained_layout=True, figsize=(10, 14))
gs = GridSpec(10, 1, figure=f)
ax = []
for i in range(10):
    ax.append(f.add_subplot(gs[i, 0]))

for panel, region in enumerate(region_list):
    ax[panel].plot(data['winds'][region]['mean_anom'].time,
                   data['winds'][region]['mean_anom'].values,
                   'k-', linewidth=1, zorder=5, label='WINDS')

    ax[panel].plot(data['cmems'][region]['mean_anom'].time,
                   data['cmems'][region]['mean_anom'].values,
                   'r-', linewidth=1, zorder=4, alpha=0.5, label='CMEMS')

    ax[panel].plot(data['altimetry'][region]['mean_anom'].time,
                   data['altimetry'][region]['mean_anom'].values,
                   'b-', linewidth=1, zorder=3, alpha=0.5, label='GlobCurrent')

    ax[panel].set_xlim([datetime(year=1993, month=1, day=1),
                        datetime(year=2019, month=12, day=31)])
    ax[panel].set_ylabel('$\overline{v}_s$ anomaly')

    for tick_year in [1996, 2000, 2004, 2008, 2012, 2016]:
        ax[panel].axvline(x=datetime(year=tick_year, month=1, day=1), color='k',
                          linestyle='--', linewidth=0.5, alpha=0.2, zorder=2)

    if panel != 9:
        ax[panel].set_xticks([])

    if panel == 0:
        legend = ax[panel].legend(ncol=3)
        legend.get_frame().set_alpha(0)

    ax[panel].spines.right.set_visible(False)
    ax[panel].spines.top.set_visible(False)

    ax[panel].set_title(text_dict[region], fontsize=10)

plt.savefig(dirs['fig'] + 'sfc_speed_anomalies_time_series_comparison.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Now plot the geographical key
f = plt.figure(constrained_layout=True, figsize=(10, 5.1))

gs = GridSpec(1, 2, figure=f, width_ratios=[1, 0.03])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1]))

winds_mean = xr.open_dataset(fh['winds_mean']).coarsen(lon=10, lat=10, boundary='trim').mean()
winds_mean_degraded = xr.open_dataset(fh['winds_mean']).coarsen(lon=50, lat=50, boundary='trim').mean()

lsm = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='k', zorder=1)
pcm = ax[0].pcolormesh(winds_mean.lon, winds_mean.lat, ((winds_mean.u_surf**2 + winds_mean.v_surf**2)**0.5).values[0, :, :],
                       cmap=cmr.get_sub_cmap('cmr.neutral_r', 0, 0.33), vmin=0, vmax=1, transform=ccrs.PlateCarree())
ax[0].quiver(winds_mean_degraded.lon, winds_mean_degraded.lat,
             winds_mean_degraded.u_surf.values[0, :, :],
             winds_mean_degraded.v_surf.values[0, :, :],
             units='inches', scale=4, width=0.01, alpha=0.5,
             transform=ccrs.PlateCarree())

ax[0].add_feature(lsm)

# Plot bounds
def make_shape(lon0, lon1, lat0, lat1):
    x_pnts = [lon0, lon1, lon1, lon0]
    y_pnts = [lat0, lat0, lat1, lat1]

    return x_pnts, y_pnts

for region in region_list:
    lon0 = float(xr.open_dataset(fh['cmems'][region]).longitude[0])
    lon1 = float(xr.open_dataset(fh['cmems'][region]).longitude[-1])
    lat0 = float(xr.open_dataset(fh['cmems'][region]).latitude[0])
    lat1 = float(xr.open_dataset(fh['cmems'][region]).latitude[-1])

    x_pnt, y_pnt = make_shape(lon0, lon1, lat0, lat1)

    ax[0].fill(x_pnt, y_pnt, fc='None', ec='k', linestyle='--', transform=ccrs.PlateCarree())
    ax[0].text(lon1+0.05, lat1+0.05, region, fontsize=8, ha='left', va='bottom')

cb = plt.colorbar(pcm, cax=ax[-1], fraction=0.05, orientation='vertical')
cb.set_label('Surface velocity (m/s)', size=12)
ax[-1].tick_params(axis='x', labelsize=12)

plt.savefig(dirs['fig'] + 'sfc_speed_key.pdf', bbox_inches='tight', dpi=300)
plt.close()
