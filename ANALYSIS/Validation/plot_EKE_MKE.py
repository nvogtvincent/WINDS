#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare EKE/MKE between gridded observations and WINDS.

- Input: daily means (filtering out tidal signal)
- Carry out high-pass filter with 30d cutoff to extract components associated
  with mesoscale variability (u_mes, v_mes)
- Compute MKE and EKE:
  MKE = 0.5*(mean(u)**2 + mean(v)**2)
  EKE = 0.5*mean(u_mes**2 + v_mes**2)
  q.v. von Appen et al. (2022)
- Also compare with available RAMA moored current meters

@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cmasher as cmr
import xarray as xr
import dask as ds
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dask.diagnostics import ProgressBar
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from datetime import timedelta
from scipy import signal
from tqdm import tqdm
from glob import glob

###############################################################################
# PARAMETERS ##################################################################
###############################################################################

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../../'
dirs['fig'] = dirs['root'] + 'FIGURES/Validation/'
dirs['ref'] = dirs['root'] + 'REFERENCE/Validation/'

# FILE-HANDLES
fh = {}
fh['sat_uv'] = dirs['ref'] + 'dataset-uv-rep-daily.nc'
fh['winds_uv'] = dirs['ref'] + 'WINDS-M_SFC.nc'
fh['gdp_uv'] = dirs['ref'] + 'drifter_monthlymeans.nc'
fh['rama'] = sorted(glob(dirs['ref'] + 'RAMA/*.nc'))

###############################################################################
# FILTER DATA (AND SPLIT BY MONTH) ############################################
###############################################################################

rama_data = {'lon': [],
             'lat': [],
             'mke': [],
             'eke': []}

# Set up filter
order = 4
fs = 1        # (1/d)
cutoff = 1/30 # (1/d)
hp_filt = signal.butter(order, cutoff, fs=fs, btype='high', analog=False, output='sos')

print('Extracting MKE and EKE from RAMA records')
# Pass through data records
for rama_fh in tqdm(fh['rama'], total=len(fh['rama'])):
    with xr.open_dataset(rama_fh) as file:
        rama_temp_data = {'lon': [],
                          'lat': [],
                          'mke': [],
                          'eke': [],
                          'len': []}

        rama_temp_data['lon'] = file.lon.values[0]
        rama_temp_data['lat'] = file.lat.values[0]

        u_segments = []
        v_segments = []
        u_hp_segments = []
        v_hp_segments = []

        for depth in file.depth:
            rama_time = file.time
            rama_u = file.U_320.loc[:, depth, rama_temp_data['lat'], rama_temp_data['lon']].squeeze().drop(['depth', 'lat', 'lon'])/100
            rama_u = rama_u.interpolate_na(dim='time', method='linear', max_gap=timedelta(days=5))
            rama_v = file.V_321.loc[:, depth, rama_temp_data['lat'], rama_temp_data['lon']].squeeze().drop(['depth', 'lat', 'lon'])/100
            rama_v = rama_v.interpolate_na(dim='time', method='linear', max_gap=timedelta(days=5))

            # Now split into contiguous segments at least 4 months long and
            # take high-pass filter within each segment
            splits = np.split(rama_u, np.where(np.isnan(rama_u))[0])
            u_segments.append([segment[~np.isnan(segment)] for segment in splits if len(segment) > 120])
            u_hp_segments.append([signal.sosfiltfilt(hp_filt, segment[~np.isnan(segment)], axis=0) for segment in splits if len(segment) > 120])

            splits = np.split(rama_v, np.where(np.isnan(rama_v))[0])
            v_segments.append([segment[~np.isnan(segment)] for segment in splits if len(segment) > 120])
            v_hp_segments.append([signal.sosfiltfilt(hp_filt, segment[~np.isnan(segment)], axis=0) for segment in splits if len(segment) > 120])

        # Merge segments
        u_segments = xr.concat([segment for depth in u_segments for segment in depth], dim='time')
        u_hp_segments = np.concatenate([segment for depth in u_hp_segments for segment in depth])
        v_segments = xr.concat([segment for depth in v_segments for segment in depth], dim='time')
        v_hp_segments = np.concatenate([segment for depth in v_hp_segments for segment in depth])

        rama_data['lon'].append(rama_temp_data['lon'])
        rama_data['lat'].append(rama_temp_data['lat'])
        rama_data['eke'].append(((u_hp_segments**2 + v_hp_segments**2)*0.5).mean())
        rama_data['mke'].append(((u_segments.mean(dim='time')**2 + v_segments.mean(dim='time')**2)*0.5).values)

with xr.open_dataset(fh['winds_uv'], chunks={'lon': 500, 'lat': 500, 'time_counter': 90}) as file:
    # Set up filter
    order = 4
    fs = 1        # (1/d)
    cutoff = 1/30 # (1/d)
    hp_filt = signal.butter(order, cutoff, fs=fs, btype='high', analog=False, output='sos')

    # Coarsen to 1/4 deg
    print('Coarsening WINDS u-velocity')
    with ProgressBar():
        u_surf = file.u_surf.coarsen({'time_counter': 1, 'lon': 5, 'lat': 5}, boundary='trim').mean().rename({'time_counter': 'time', 'lon': 'longitude', 'lat': 'latitude'}).compute()
    print('Coarsening WINDS v-velocity')
    with ProgressBar():
        v_surf = file.v_surf.coarsen({'time_counter': 1, 'lon': 5, 'lat': 5}, boundary='trim').mean().rename({'time_counter': 'time', 'lon': 'longitude', 'lat': 'latitude'}).compute()

    # Compute EKE and MKE
    EKE_winds = xr.full_like(u_surf, fill_value=0)
    EKE_winds.data = (signal.sosfiltfilt(hp_filt, u_surf, axis=0)**2 +
                      signal.sosfiltfilt(hp_filt, v_surf, axis=0)**2)*0.5
    EKE_winds_monclim = EKE_winds.groupby('time.month').mean()
    EKE_winds_mean = EKE_winds.mean(dim='time')

    del EKE_winds

    u_monclim = u_surf.groupby('time.month').mean()
    v_monclim = v_surf.groupby('time.month').mean()
    u_mean = u_surf.mean(dim='time')
    v_mean = v_surf.mean(dim='time')

    MKE_winds_monclim = (u_monclim**2 + v_monclim**2)*0.5
    MKE_winds_mean = (u_mean**2 + v_mean**2)*0.5

    del u_monclim, v_monclim, u_mean, v_mean


with xr.open_dataset(fh['sat_uv']) as file:
    # Set up filter
    order = 4
    fs = 1        # (1/d)
    cutoff = 1/30 # (1/d)
    hp_filt = signal.butter(order, cutoff, fs=fs, btype='high', analog=False, output='sos')

    # Compute EKE and MKE
    EKE_sat = xr.full_like(file.uo[:, 0, :, :], fill_value=0).drop('depth')
    EKE_sat.data = (signal.sosfiltfilt(hp_filt, file.uo[:, 0, :, :], axis=0)**2 +
                    signal.sosfiltfilt(hp_filt, file.vo[:, 0, :, :], axis=0)**2)*0.5
    EKE_sat_monclim = EKE_sat.groupby('time.month').mean()
    EKE_sat_mean = EKE_sat.mean(dim='time')

    del EKE_sat

    u_monclim = file.uo[:, 0, :, :].drop('depth').groupby('time.month').mean()
    v_monclim = file.vo[:, 0, :, :].drop('depth').groupby('time.month').mean()
    u_mean = file.uo[:, 0, :, :].drop('depth').mean(dim='time')
    v_mean = file.vo[:, 0, :, :].drop('depth').mean(dim='time')

    MKE_sat_monclim = (u_monclim**2 + v_monclim**2)*0.5
    MKE_sat_mean = (u_mean**2 + v_mean**2)*0.5

    del u_monclim, v_monclim, u_mean, v_mean

with xr.open_dataset(fh['gdp_uv'], decode_times=False) as file:
    uv_gdp = file.sel(longitude = (file.Lon > 34)*(file.Lon < 78),
                      latitude = (file.Lat > -24)*(file.Lat < 0.5))

    # Note - GDP profiles cannot be filtered by year, data not available.
    uv_gdp = uv_gdp.rename({'Lon': 'longitude', 'Lat': 'latitude'})
    uv_gdp = uv_gdp.assign_coords({'time': np.arange(1,13)}).drop('Time')

    u_monclim = uv_gdp.U
    v_monclim = uv_gdp.V
    u_mean = uv_gdp.U.mean(dim='time')
    v_mean = uv_gdp.V.mean(dim='time')

    MKE_gdp_monclim = (u_monclim**2 + v_monclim**2)*0.5
    MKE_gdp_mean = (u_mean**2 + v_mean**2)*0.5

    del uv_gdp, u_mean, v_mean, u_monclim, v_monclim

###############################################################################
# PLOT MKE ####################################################################
###############################################################################
f = plt.figure(constrained_layout=True, figsize=(17, 14))

gs = GridSpec(5, 3, figure=f, height_ratios=[1, 1, 1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[4, :]))

plot_list = []
gl = []

lon_bnd_w = np.append(MKE_winds_mean.longitude, 2*MKE_winds_mean.longitude[-1]-MKE_winds_mean.longitude[-2])
lon_bnd_w -= 0.5*(lon_bnd_w[-1]-lon_bnd_w[-2])
lat_bnd_w = np.append(MKE_winds_mean.latitude, 2*MKE_winds_mean.latitude[-1]-MKE_winds_mean.latitude[-2])
lat_bnd_w -= 0.5*(lat_bnd_w[-1]-lat_bnd_w[-2])

lon_bnd_g = np.append(MKE_gdp_mean.longitude, 2*MKE_gdp_mean.longitude[-1]-MKE_gdp_mean.longitude[-2])
lon_bnd_g -= 0.5*(lon_bnd_g[-1]-lon_bnd_g[-2])
lat_bnd_g = np.append(MKE_gdp_mean.latitude, 2*MKE_gdp_mean.latitude[-1]-MKE_gdp_mean.latitude[-2])
lat_bnd_g -= 0.5*(lat_bnd_g[-1]-lat_bnd_g[-2])

lon_bnd_c = np.append(MKE_sat_mean.longitude, 2*MKE_sat_mean.longitude[-1]-MKE_sat_mean.longitude[-2])
lon_bnd_c -= 0.5*(lon_bnd_c[-1]-lon_bnd_c[-2])
lat_bnd_c = np.append(MKE_sat_mean.latitude, 2*MKE_sat_mean.latitude[-1]-MKE_sat_mean.latitude[-2])
lat_bnd_c -= 0.5*(lat_bnd_c[-1]-lat_bnd_c[-2])

lev = np.arange(0, 1.7, 0.1)

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

for i, month in enumerate(['January', 'February', 'March', 'April']):
    # WINDS
    ax[3*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i].pcolormesh(lon_bnd_w, lat_bnd_w,
                                        MKE_winds_monclim.loc[i+1, :, :],
                                        cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                        transform=ccrs.PlateCarree(), rasterized=True))

    ax[3*i].add_feature(land_cst)

    gl.append(ax[3*i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i].ylabel_style = {'size': 20}
    gl[3*i].xlabel_style = {'size': 20}
    gl[3*i].xlabels_top = False
    gl[3*i].ylabels_right = False
    ax[3*i].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*3 in [0, 3, 6, 9]:
        gl[3*i].ylabels_left = True
    else:
        gl[3*i].ylabels_left = False

    gl[3*i].xlabels_bottom = False

    if i*3 >= 9:
        gl[3*i].xlabels_bottom = True
    else:
        gl[3*i].xlabels_bottom = False

    gl[3*i].ylabels_left = False

    # ALTIMETRY
    ax[3*i+1].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i+1].text(77, -3.5, 'GlobCurrent', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i+1].pcolormesh(lon_bnd_c, lat_bnd_c,
                                          MKE_sat_monclim.loc[i+1, :, :],
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                          transform=ccrs.PlateCarree(), rasterized=True))

    ax[3*i+1].add_feature(land_cst)

    gl.append(ax[3*i+1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i+1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i+1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i+1].ylabel_style = {'size': 20}
    gl[3*i+1].xlabel_style = {'size': 20}
    gl[3*i+1].xlabels_top = False
    gl[3*i+1].ylabels_right = False
    ax[3*i+1].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*3+1 in [0, 3, 6, 9]:
        gl[3*i+1].ylabels_left = True
    else:
        gl[3*i+1].ylabels_left = False

    gl[3*i+1].xlabels_bottom = False

    if i*3+1 >= 9:
        gl[3*i+1].xlabels_bottom = True
    else:
        gl[3*i+1].xlabels_bottom = False

    gl[3*i+1].ylabels_left = False

    # GDP
    ax[3*i+2].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i+2].text(77, -3.5, 'GDP', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i+2].pcolormesh(lon_bnd_g, lat_bnd_g,
                                          MKE_gdp_monclim.loc[i+1, :, :].T,
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                          transform=ccrs.PlateCarree(), rasterized=True))

    ax[3*i+2].add_feature(land_cst)
    ax[3*i+2].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    gl.append(ax[3*i+2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i+2].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i+2].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i+2].ylabel_style = {'size': 20}
    gl[3*i+2].xlabel_style = {'size': 20}
    gl[3*i+2].xlabels_top = False
    gl[3*i+2].ylabels_right = False

    if i*3+2 >= 9:
        gl[3*i+2].xlabels_bottom = True
    else:
        gl[3*i+2].xlabels_bottom = False

    gl[3*i+2].ylabels_left = False

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('MKE ($m^{2}\ s^{-2}$)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'MKE_comp_1.pdf', bbox_inches='tight', dpi=300)
plt.close()

f = plt.figure(constrained_layout=True, figsize=(17, 14))

gs = GridSpec(5, 3, figure=f, height_ratios=[1, 1, 1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[4, :]))

plot_list = []
gl = []

for i, month in enumerate(['May', 'June', 'July', 'August']):
    # WINDS
    ax[3*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i].pcolormesh(lon_bnd_w, lat_bnd_w,
                                        MKE_winds_monclim.loc[i+5, :, :],
                                        cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                        transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i].add_feature(land_cst)

    gl.append(ax[3*i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i].ylabel_style = {'size': 20}
    gl[3*i].xlabel_style = {'size': 20}
    gl[3*i].xlabels_top = False
    gl[3*i].ylabels_right = False
    ax[3*i].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*3 in [0, 3, 6, 9]:
        gl[3*i].ylabels_left = True
    else:
        gl[3*i].ylabels_left = False

    gl[3*i].xlabels_bottom = False

    if i*3 >= 9:
        gl[3*i].xlabels_bottom = True
    else:
        gl[3*i].xlabels_bottom = False

    gl[3*i].ylabels_left = False

    # ALTIMETRY
    ax[3*i+1].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i+1].text(77, -3.5, 'GlobCurrent', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i+1].pcolormesh(lon_bnd_c, lat_bnd_c,
                                          MKE_sat_monclim.loc[i+5, :, :],
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                          transform=ccrs.PlateCarree(), rasterized=True))

    ax[3*i+1].add_feature(land_cst)

    gl.append(ax[3*i+1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i+1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i+1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i+1].ylabel_style = {'size': 20}
    gl[3*i+1].xlabel_style = {'size': 20}
    gl[3*i+1].xlabels_top = False
    gl[3*i+1].ylabels_right = False
    ax[3*i+1].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*3+1 in [0, 3, 6, 9]:
        gl[3*i+1].ylabels_left = True
    else:
        gl[3*i+1].ylabels_left = False

    gl[3*i+1].xlabels_bottom = False

    if i*3+1 >= 9:
        gl[3*i+1].xlabels_bottom = True
    else:
        gl[3*i+1].xlabels_bottom = False

    gl[3*i+1].ylabels_left = False

    # GDP
    ax[3*i+2].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i+2].text(77, -3.5, 'GDP', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i+2].pcolormesh(lon_bnd_g, lat_bnd_g,
                                          MKE_gdp_monclim.loc[i+5, :, :].T,
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                          transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i+2].add_feature(land_cst)
    ax[3*i+2].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    gl.append(ax[3*i+2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i+2].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i+2].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i+2].ylabel_style = {'size': 20}
    gl[3*i+2].xlabel_style = {'size': 20}
    gl[3*i+2].xlabels_top = False
    gl[3*i+2].ylabels_right = False

    if i*3+2 >= 9:
        gl[3*i+2].xlabels_bottom = True
    else:
        gl[3*i+2].xlabels_bottom = False

    gl[3*i+2].ylabels_left = False

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('MKE ($m^{2}\ s^{-2}$)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'MKE_comp_2.pdf', bbox_inches='tight', dpi=300)
plt.close()

f = plt.figure(constrained_layout=True, figsize=(17, 14))

gs = GridSpec(5, 3, figure=f, height_ratios=[1, 1, 1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 2], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[4, :]))

plot_list = []
gl = []

for i, month in enumerate(['September', 'October', 'November', 'December']):
    # WINDS
    ax[3*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i].pcolormesh(lon_bnd_w, lat_bnd_w,
                                        MKE_winds_monclim.loc[i+9, :, :],
                                        cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                        transform=ccrs.PlateCarree(), rasterized=True))

    ax[3*i].add_feature(land_cst)

    gl.append(ax[3*i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i].ylabel_style = {'size': 20}
    gl[3*i].xlabel_style = {'size': 20}
    gl[3*i].xlabels_top = False
    gl[3*i].ylabels_right = False
    ax[3*i].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*3 in [0, 3, 6, 9]:
        gl[3*i].ylabels_left = True
    else:
        gl[3*i].ylabels_left = False

    gl[3*i].xlabels_bottom = False

    if i*3 >= 9:
        gl[3*i].xlabels_bottom = True
    else:
        gl[3*i].xlabels_bottom = False

    gl[3*i].ylabels_left = False

    # ALTIMETRY
    ax[3*i+1].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i+1].text(77, -3.5, 'GlobCurrent', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i+1].pcolormesh(lon_bnd_c, lat_bnd_c,
                                          MKE_sat_monclim.loc[i+9, :, :],
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                          transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i+1].add_feature(land_cst)

    gl.append(ax[3*i+1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i+1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i+1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i+1].ylabel_style = {'size': 20}
    gl[3*i+1].xlabel_style = {'size': 20}
    gl[3*i+1].xlabels_top = False
    gl[3*i+1].ylabels_right = False
    ax[3*i+1].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*3+1 in [0, 3, 6, 9]:
        gl[3*i+1].ylabels_left = True
    else:
        gl[3*i+1].ylabels_left = False

    gl[3*i+1].xlabels_bottom = False

    if i*3+1 >= 9:
        gl[3*i+1].xlabels_bottom = True
    else:
        gl[3*i+1].xlabels_bottom = False

    gl[3*i+1].ylabels_left = False

    # GDP
    ax[3*i+2].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i+2].text(77, -3.5, 'GDP', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i+2].pcolormesh(lon_bnd_g, lat_bnd_g,
                                          MKE_gdp_monclim.loc[i+9, :, :].T,
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                          transform=ccrs.PlateCarree(), rasterized=True))

    ax[3*i+2].add_feature(land_cst)
    ax[3*i+2].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    gl.append(ax[3*i+2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[3*i+2].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[3*i+2].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[3*i+2].ylabel_style = {'size': 20}
    gl[3*i+2].xlabel_style = {'size': 20}
    gl[3*i+2].xlabels_top = False
    gl[3*i+2].ylabels_right = False

    if i*3+2 >= 9:
        gl[3*i+2].xlabels_bottom = True
    else:
        gl[3*i+2].xlabels_bottom = False

    gl[3*i+2].ylabels_left = False

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('MKE ($m^{2}\ s^{-2}$)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'MKE_comp_3.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot annual mean
f = plt.figure(constrained_layout=True, figsize=(6, 10))

gs = GridSpec(4, 1, figure=f, height_ratios=[1, 1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, :]))

plot_list = []
gl = []

# WINDS
ax[0].text(77, -0.8, 'WINDS', fontsize=18, color='w', va='top', ha='right')

plot_list.append(ax[0].pcolormesh(lon_bnd_w, lat_bnd_w, MKE_winds_mean,
                                  cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                  transform=ccrs.PlateCarree(), rasterized=True))

for rama_station in range(len(rama_data['lon'])):
    ax[0].scatter(rama_data['lon'][rama_station],
                  rama_data['lat'][rama_station],
                  s=100, c=rama_data['mke'][rama_station],
                  cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                  transform=ccrs.PlateCarree(), edgecolors='k')

ax[0].add_feature(land_cst)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[0].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[0].ylabel_style = {'size': 20}
gl[0].xlabel_style = {'size': 20}
gl[0].xlabels_top = False
gl[0].ylabels_right = False
ax[0].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())
gl[0].ylabels_left = True
gl[0].xlabels_bottom = False

# ALTIMETRY
ax[1].text(77, -0.8, 'GlobCurrent', fontsize=18, color='w', va='top', ha='right')

plot_list.append(ax[1].pcolormesh(lon_bnd_c, lat_bnd_c, MKE_sat_mean,
                                      cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                      transform=ccrs.PlateCarree(), rasterized=True))

for rama_station in range(len(rama_data['lon'])):
    ax[1].scatter(rama_data['lon'][rama_station],
                  rama_data['lat'][rama_station],
                  s=100, c=rama_data['mke'][rama_station],
                  cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                  transform=ccrs.PlateCarree(), edgecolors='k')

ax[1].add_feature(land_cst)

gl.append(ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[1].ylabel_style = {'size': 20}
gl[1].xlabel_style = {'size': 20}
gl[1].xlabels_top = False
gl[1].ylabels_right = False
ax[1].set_extent([34.5, 77, -23.5, 0], crs=ccrs.PlateCarree())
gl[1].ylabels_left = True
gl[1].xlabels_bottom = False

# GDP
ax[2].text(77, -0.8, 'GDP', fontsize=18, color='w', va='top', ha='right')

plot_list.append(ax[2].pcolormesh(lon_bnd_g, lat_bnd_g, MKE_gdp_mean.T,
                                      cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                                      transform=ccrs.PlateCarree(), rasterized=True))

for rama_station in range(len(rama_data['lon'])):
    ax[2].scatter(rama_data['lon'][rama_station],
                  rama_data['lat'][rama_station],
                  s=100, c=rama_data['mke'][rama_station],
                  cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                  transform=ccrs.PlateCarree(), edgecolors='k')

ax[2].add_feature(land_cst)
ax[2].set_extent([34.5, 77, -23.5, 0], crs=ccrs.PlateCarree())

gl.append(ax[2].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[2].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[2].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[2].ylabel_style = {'size': 20}
gl[2].xlabel_style = {'size': 20}
gl[2].xlabels_top = False
gl[2].ylabels_right = False
gl[2].xlabels_bottom = True
gl[2].ylabels_left = True

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('MKE ($m^{2}\ s^{-2}$)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'MKE_annual_mean.pdf', bbox_inches='tight', dpi=300)
plt.close()

###############################################################################
# PLOT EKE ####################################################################
###############################################################################
f = plt.figure(constrained_layout=True, figsize=(12, 14))

gs = GridSpec(5, 2, figure=f, height_ratios=[1, 1, 1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[4, :]))

plot_list = []
gl = []

lon_bnd_w = np.append(EKE_winds_mean.longitude, 2*EKE_winds_mean.longitude[-1]-EKE_winds_mean.longitude[-2])
lon_bnd_w -= 0.5*(lon_bnd_w[-1]-lon_bnd_w[-2])
lat_bnd_w = np.append(EKE_winds_mean.latitude, 2*EKE_winds_mean.latitude[-1]-EKE_winds_mean.latitude[-2])
lat_bnd_w -= 0.5*(lat_bnd_w[-1]-lat_bnd_w[-2])

lon_bnd_c = np.append(EKE_sat_mean.longitude, 2*EKE_sat_mean.longitude[-1]-EKE_sat_mean.longitude[-2])
lon_bnd_c -= 0.5*(lon_bnd_c[-1]-lon_bnd_c[-2])
lat_bnd_c = np.append(EKE_sat_mean.latitude, 2*EKE_sat_mean.latitude[-1]-EKE_sat_mean.latitude[-2])
lat_bnd_c -= 0.5*(lat_bnd_c[-1]-lat_bnd_c[-2])

lev = np.arange(0, 1.7, 0.1)

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

for i, month in enumerate(['January', 'February', 'March', 'April']):
    # WINDS
    ax[2*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[2*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[2*i].pcolormesh(lon_bnd_w, lat_bnd_w,
                                        EKE_winds_monclim.loc[i+1, :, :],
                                        cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1),
                                        transform=ccrs.PlateCarree(), rasterized=True))

    ax[2*i].add_feature(land_cst)

    gl.append(ax[2*i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[2*i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[2*i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[2*i].ylabel_style = {'size': 20}
    gl[2*i].xlabel_style = {'size': 20}
    gl[2*i].xlabels_top = False
    gl[2*i].ylabels_right = False
    ax[2*i].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*2 in [0, 3, 6, 9]:
        gl[2*i].ylabels_left = True
    else:
        gl[2*i].ylabels_left = False

    gl[2*i].xlabels_bottom = False

    if i*2 >= 9:
        gl[2*i].xlabels_bottom = True
    else:
        gl[2*i].xlabels_bottom = False

    gl[2*i].ylabels_left = False

    # ALTIMETRY
    ax[2*i+1].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[2*i+1].text(77, -3.5, 'GlobCurrent', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[2*i+1].pcolormesh(lon_bnd_c, lat_bnd_c,
                                          EKE_sat_monclim.loc[i+1, :, :],
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1),
                                          transform=ccrs.PlateCarree(), rasterized=True))

    ax[2*i+1].add_feature(land_cst)

    gl.append(ax[2*i+1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[2*i+1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[2*i+1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[2*i+1].ylabel_style = {'size': 20}
    gl[2*i+1].xlabel_style = {'size': 20}
    gl[2*i+1].xlabels_top = False
    gl[2*i+1].ylabels_right = False
    ax[2*i+1].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*2+1 in [0, 3, 6, 9]:
        gl[2*i+1].ylabels_left = True
    else:
        gl[2*i+1].ylabels_left = False

    gl[2*i+1].xlabels_bottom = False

    if i*2+1 >= 9:
        gl[2*i+1].xlabels_bottom = True
    else:
        gl[2*i+1].xlabels_bottom = False

    gl[2*i+1].ylabels_left = False

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('EKE ($m^{2}\ s^{-2}$)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'EKE_comp_1.pdf', bbox_inches='tight', dpi=300)
plt.close()

f = plt.figure(constrained_layout=True, figsize=(12, 14))

gs = GridSpec(5, 2, figure=f, height_ratios=[1, 1, 1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[4, :]))

plot_list = []
gl = []

for i, month in enumerate(['May', 'June', 'July', 'August']):
    # WINDS
    ax[2*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[2*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[2*i].pcolormesh(lon_bnd_w, lat_bnd_w,
                                        EKE_winds_monclim.loc[i+5, :, :],
                                        cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1),
                                        transform=ccrs.PlateCarree(), rasterized=True))
    ax[2*i].add_feature(land_cst)

    gl.append(ax[2*i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[2*i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[2*i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[2*i].ylabel_style = {'size': 20}
    gl[2*i].xlabel_style = {'size': 20}
    gl[2*i].xlabels_top = False
    gl[2*i].ylabels_right = False
    ax[2*i].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*2 in [0, 3, 6, 9]:
        gl[2*i].ylabels_left = True
    else:
        gl[2*i].ylabels_left = False

    gl[2*i].xlabels_bottom = False

    if i*2 >= 9:
        gl[2*i].xlabels_bottom = True
    else:
        gl[2*i].xlabels_bottom = False

    gl[2*i].ylabels_left = False

    # ALTIMETRY
    ax[2*i+1].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[2*i+1].text(77, -3.5, 'GlobCurrent', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[2*i+1].pcolormesh(lon_bnd_c, lat_bnd_c,
                                          EKE_sat_monclim.loc[i+5, :, :],
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1),
                                          transform=ccrs.PlateCarree(), rasterized=True))

    ax[2*i+1].add_feature(land_cst)

    gl.append(ax[2*i+1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[2*i+1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[2*i+1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[2*i+1].ylabel_style = {'size': 20}
    gl[2*i+1].xlabel_style = {'size': 20}
    gl[2*i+1].xlabels_top = False
    gl[2*i+1].ylabels_right = False
    ax[2*i+1].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*2+1 in [0, 3, 6, 9]:
        gl[2*i+1].ylabels_left = True
    else:
        gl[2*i+1].ylabels_left = False

    gl[2*i+1].xlabels_bottom = False

    if i*2+1 >= 9:
        gl[2*i+1].xlabels_bottom = True
    else:
        gl[2*i+1].xlabels_bottom = False

    gl[2*i+1].ylabels_left = False

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('EKE ($m^{2}\ s^{-2}$)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'EKE_comp_2.pdf', bbox_inches='tight', dpi=300)
plt.close()

f = plt.figure(constrained_layout=True, figsize=(12, 14))

gs = GridSpec(5, 2, figure=f, height_ratios=[1, 1, 1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[4, :]))

plot_list = []
gl = []

for i, month in enumerate(['September', 'October', 'November', 'December']):
    # WINDS
    ax[2*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[2*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[2*i].pcolormesh(lon_bnd_w, lat_bnd_w,
                                        EKE_winds_monclim.loc[i+9, :, :],
                                        cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1),
                                        transform=ccrs.PlateCarree(), rasterized=True))

    ax[2*i].add_feature(land_cst)

    gl.append(ax[2*i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[2*i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[2*i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[2*i].ylabel_style = {'size': 20}
    gl[2*i].xlabel_style = {'size': 20}
    gl[2*i].xlabels_top = False
    gl[2*i].ylabels_right = False
    ax[2*i].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*2 in [0, 3, 6, 9]:
        gl[2*i].ylabels_left = True
    else:
        gl[2*i].ylabels_left = False

    gl[2*i].xlabels_bottom = False

    if i*2 >= 9:
        gl[2*i].xlabels_bottom = True
    else:
        gl[2*i].xlabels_bottom = False

    gl[2*i].ylabels_left = False

    # ALTIMETRY
    ax[2*i+1].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[2*i+1].text(77, -3.5, 'GlobCurrent', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[2*i+1].pcolormesh(lon_bnd_c, lat_bnd_c,
                                          EKE_sat_monclim.loc[i+9, :, :],
                                          cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1),
                                          transform=ccrs.PlateCarree(), rasterized=True))
    ax[2*i+1].add_feature(land_cst)

    gl.append(ax[2*i+1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='w', linestyle='-', zorder=11))
    gl[2*i+1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[2*i+1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[2*i+1].ylabel_style = {'size': 20}
    gl[2*i+1].xlabel_style = {'size': 20}
    gl[2*i+1].xlabels_top = False
    gl[2*i+1].ylabels_right = False
    ax[2*i+1].set_extent([34.5, 77.5, -23.5, 0], crs=ccrs.PlateCarree())

    if i*2+1 in [0, 3, 6, 9]:
        gl[2*i+1].ylabels_left = True
    else:
        gl[2*i+1].ylabels_left = False

    gl[2*i+1].xlabels_bottom = False

    if i*2+1 >= 9:
        gl[2*i+1].xlabels_bottom = True
    else:
        gl[2*i+1].xlabels_bottom = False

    gl[2*i+1].ylabels_left = False

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('EKE ($m^{2}\ s^{-2}$)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'EKE_comp_3.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot annual mean

f = plt.figure(constrained_layout=True, figsize=(6, 7))

gs = GridSpec(3, 1, figure=f, height_ratios=[1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, :]))

plot_list = []
gl = []

# WINDS
ax[0].text(77, -0.8, 'WINDS', fontsize=18, color='w', va='top', ha='right')

plot_list.append(ax[0].pcolormesh(lon_bnd_w, lat_bnd_w, EKE_winds_mean,
                                    cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1),
                                    transform=ccrs.PlateCarree(), rasterized=True))

for rama_station in range(len(rama_data['lon'])):
    ax[0].scatter(rama_data['lon'][rama_station],
                  rama_data['lat'][rama_station],
                  s=100, c=rama_data['eke'][rama_station],
                  cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                  transform=ccrs.PlateCarree(), edgecolors='k')

ax[0].add_feature(land_cst)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[0].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[0].ylabel_style = {'size': 20}
gl[0].xlabel_style = {'size': 20}
gl[0].xlabels_top = False
gl[0].ylabels_right = False
ax[0].set_extent([34.5, 77, -23.5, 0], crs=ccrs.PlateCarree())
gl[0].ylabels_left = True
gl[0].xlabels_bottom = False

# ALTIMETRY
ax[1].text(77, -0.8, 'GlobCurrent', fontsize=18, color='w', va='top', ha='right')

plot_list.append(ax[1].pcolormesh(lon_bnd_c, lat_bnd_c, EKE_sat_mean,
                                      cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=5e-1),
                                      transform=ccrs.PlateCarree(), rasterized=True))

for rama_station in range(len(rama_data['lon'])):
    ax[1].scatter(rama_data['lon'][rama_station],
                  rama_data['lat'][rama_station],
                  s=100, c=rama_data['eke'][rama_station],
                  cmap=cmr.eclipse, norm=colors.LogNorm(vmin=1e-3, vmax=3e0),
                  transform=ccrs.PlateCarree(), edgecolors='k')

ax[1].add_feature(land_cst)

gl.append(ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[1].ylabel_style = {'size': 20}
gl[1].xlabel_style = {'size': 20}
gl[1].xlabels_top = False
gl[1].ylabels_right = False
ax[1].set_extent([34.5, 77, -23.5, 0], crs=ccrs.PlateCarree())
gl[1].ylabels_left = True
gl[1].xlabels_bottom = True


cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('EKE ($m^{2}\ s^{-2}$)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'EKE_annual_mean.pdf', bbox_inches='tight', dpi=300)
plt.close()
