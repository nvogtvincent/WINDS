#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare WINDS output to trajectories from GDP dataset
@author: Noam
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import cmasher as cmr
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection

###############################################################################
# PARAMETERS ##################################################################
###############################################################################

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../../'
dirs['fig'] = dirs['root'] + 'FIGURES/Validation/'
dirs['traj'] = dirs['root'] + 'REFERENCE/TRAJ/'
dirs['grid'] = dirs['root'] + 'REFERENCE/GRID/'

case = 'Zanzibar'

bounds = {'Zanzibar': [38.5, 40.5, -7, -4]}

year = 2019
month = 7
day = 1

fh_winds = dirs['traj'] + 'WINDS_FLOATVIS_' + case + '_' + str(year) + '_' + str(month) + '_' + str(day) + '.nc'
fh_cmems = dirs['traj'] + 'CMEMS_FLOATVIS_' + case + '_' + str(year) + '_' + str(month) + '_' + str(day) + '.nc'

out_dt = 0.5 # Hours

###############################################################################
# MAIN ROUTINE ################################################################
###############################################################################
#42497

def get_traj(fh, idx):
    # Get a particular trajectory
    file = xr.open_dataset(fh)
    pos = np.where(file.idx0.values.astype(int) == idx)[0]
    return {'lon': file.lon.loc[pos, :].values,
            'lat': file.lat.loc[pos, :].values}

winds_traj = get_traj(fh_winds, 42503)
cmems_traj = get_traj(fh_cmems, 42499) # ID has to be found manually due to grid differences

###############################################################################
# PLOTTING ####################################################################
###############################################################################

f = plt.figure(constrained_layout=True, figsize=(9.1, 6))

gs = GridSpec(1, 3, figure=f, width_ratios=[1, 1, 0.08])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 2])) # Colorbar

# Firstly plot the background (LSM, bathymetry, and corals)
# WINDS
bath_file = xr.load_dataset(dirs['grid'] + 'griddata_winds.nc')
lsm = bath_file.mask_rho.values[1:-1, 1:-1]
lsm = np.ma.masked_array(1-lsm, mask=lsm)
bath = bath_file.h.values[1:-1, 1:-1]
lon_psi = bath_file.lon_psi.values[0, :]
lat_psi = bath_file.lat_psi.values[:, 0]

ax[0].pcolormesh(lon_psi, lat_psi, bath, cmap=cmr.get_sub_cmap(cmr.neutral_r, 0, 0.4),
                 norm=colors.LogNorm(vmin=1e1, vmax=3e3), transform=ccrs.PlateCarree())
ax[0].pcolormesh(lon_psi, lat_psi, lsm, cmap=cmr.neutral_r, vmin=0, vmax=2,
                 transform=ccrs.PlateCarree())

coral_file = xr.load_dataset(dirs['grid'] + 'coral_grid.nc')
coral = coral_file.reef_cover_w[1:-1, 1:-1]
ax[0].pcolormesh(lon_psi, lat_psi, coral, cmap=cmr.get_sub_cmap(cmr.sunburst_r, 0, 1.0),
                 norm=colors.LogNorm(vmin=1e3, vmax=1e8), transform=ccrs.PlateCarree())

ax[0].set_xlim([bounds[case][0], bounds[case][1]])
ax[0].set_ylim([bounds[case][2], bounds[case][3]])

gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=0.5, color='k', linestyle='--', zorder=11)
gl.xlocator = mticker.FixedLocator(np.arange(-210, 210, 1))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 1))

gl.xlabels_top = False
gl.ylabels_right = False
gl.ylabel_style = {'size': 12}
gl.xlabel_style = {'size': 12}

# CMEMS
lsm_cmems = coral_file.lsm_c.values
lsm_cmems = np.ma.masked_array(lsm_cmems, mask=1-lsm_cmems)
coral_cmems = coral_file.reef_cover_c
lon_rho_c = coral_file.lon_rho_c.values
lat_rho_c = coral_file.lat_rho_c.values

ax[1].pcolormesh(lon_psi, lat_psi, bath, cmap=cmr.get_sub_cmap(cmr.neutral_r, 0, 0.4),
                 norm=colors.LogNorm(vmin=1e1, vmax=3e3), transform=ccrs.PlateCarree())
ax[1].pcolormesh(lon_rho_c, lat_rho_c, lsm_cmems, cmap=cmr.neutral_r, vmin=0, vmax=2,
                 transform=ccrs.PlateCarree())

ax[1].pcolormesh(lon_rho_c, lat_rho_c, coral_cmems, cmap=cmr.get_sub_cmap(cmr.sunburst_r, 0, 1.0),
                 norm=colors.LogNorm(vmin=1e3, vmax=1e8), transform=ccrs.PlateCarree())

ax[1].set_xlim([bounds[case][0], bounds[case][1]])
ax[1].set_ylim([bounds[case][2], bounds[case][3]])

gl2 = ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='k', linestyle='--', zorder=11)
gl2.xlocator = mticker.FixedLocator(np.arange(-210, 210, 1))
gl2.ylocator = mticker.FixedLocator(np.arange(-90, 120, 1))

gl2.xlabels_top = False
gl2.ylabels_right = False
gl2.ylabels_left = False
gl2.ylabel_style = {'size': 12}
gl2.xlabel_style = {'size': 12}

# Now plot trajectories
# Firstly compute speed
def haversine_np(lon1, lat1, lon2, lat2):
    # See https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367000 * c
    return m

speed_winds = haversine_np(winds_traj['lon'][:, 1:],
                           winds_traj['lat'][:, 1:],
                           winds_traj['lon'][:, :-1],
                           winds_traj['lat'][:, :-1]).T/(out_dt*3600)

speed_cmems = haversine_np(cmems_traj['lon'][:, 1:],
                           cmems_traj['lat'][:, 1:],
                           cmems_traj['lon'][:, :-1],
                           cmems_traj['lat'][:, :-1]).T/(out_dt*3600)

vel_cmap = cmr.cosmic

for trajectory in range(len(winds_traj['lon'][:, 0])):
    winds_pts = np.array([winds_traj['lon'][trajectory, :],
                          winds_traj['lat'][trajectory, :]]).T.reshape(-1, 1, 2)
    winds_segs = np.concatenate([winds_pts[:-1], winds_pts[1:]], axis=1)
    norm = plt.Normalize(vmin=0, vmax=1)
    lc_winds = LineCollection(winds_segs, cmap=vel_cmap, norm=norm,
                              transform=ccrs.PlateCarree())
    lc_winds.set_array(speed_winds[:, trajectory])
    lc_winds.set_linewidth(0.5)
    ax[0].add_collection(lc_winds)

    cmems_pts = np.array([cmems_traj['lon'][trajectory, :],
                          cmems_traj['lat'][trajectory, :]]).T.reshape(-1, 1, 2)
    cmems_segs = np.concatenate([cmems_pts[:-1], cmems_pts[1:]], axis=1)
    norm = plt.Normalize(vmin=0, vmax=1)
    lc_cmems = LineCollection(cmems_segs, cmap=vel_cmap, norm=norm,
                              transform=ccrs.PlateCarree())
    lc_cmems.set_array(speed_cmems[:, trajectory])
    lc_cmems.set_linewidth(0.5)
    ax[1].add_collection(lc_cmems)

# Plot starting locations
ax[0].scatter(winds_traj['lon'][:, 0].mean(), winds_traj['lat'][:, 0].mean(),
              marker='+', s=100, c='k', zorder=100)
ax[1].scatter(cmems_traj['lon'][:, 0].mean(), cmems_traj['lat'][:, 0].mean(),
              marker='+', s=100, c='k', zorder=100)

ax[0].text(bounds[case][0]+0.05, bounds[case][3]-0.05, 'WINDS', fontsize=16,
           ha='left', va='top', color='w', fontweight='bold', transform=ccrs.PlateCarree())
ax[1].text(bounds[case][0]+0.05, bounds[case][3]-0.05, 'GLORYS12', fontsize=16,
           ha='left', va='top', color='w', fontweight='bold', transform=ccrs.PlateCarree())

cb = plt.colorbar(lc_winds, cax=ax[-1], fraction=0.05, orientation='vertical')
cb.set_label('Particle velocity (m s$^{-1}$)', size=14)
ax[-1].tick_params(axis='y', labelsize=12)

plt.savefig(dirs['fig'] + 'close_up.png', bbox_inches='tight', dpi=300)
