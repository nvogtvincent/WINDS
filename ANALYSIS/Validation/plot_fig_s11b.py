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
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from glob import glob
from tqdm import tqdm

###############################################################################
# PARAMETERS ##################################################################
###############################################################################

# DIRECTORIES
dirs = {}
dirs['root'] = os.getcwd() + '/../../'
dirs['fig'] = dirs['root'] + 'FIGURES/Validation/'
dirs['gdp'] = dirs['root'] + 'REFERENCE/GDP/'
dirs['traj'] = dirs['root'] + 'REFERENCE/TRAJ/'

# SITE LIST
grp_dict = {'ALD': [9, 10, 12, 13],
            'ZAN': [166, 167, 169, 170, 171],
            'MAY': [49, 50, 51],
            'STB': [100],
            'MAH': [29, 30, 31],
            'CHA': [123, 124, 125]}

name_dict = {'ALD': 'Aldabra Atoll, Seychelles',
             'ZAN': 'Zanzibar Island, Tanzania',
             'MAY': 'Mayotte, France',
             'STB': 'St Brandon, Mauritius',
             'MAH': 'Mahé, Seychelles',
             'CHA': 'Southern Banks, Chagos Archipelago'}

coord_dict = {'ALD': [46.3, -9.4],
              'ZAN': [39.3, -6.1],
              'MAY': [45.1, -12.8],
              'STB': [59.7, -16.6],
              'MAH': [55.5, -4.7],
              'CHA': [71.3, -7.1]}

code_list = [key for key in grp_dict.keys()]

###############################################################################
# MAIN ROUTINE ################################################################
###############################################################################

# Haversine formula
def haversine_np(lon1, lat1, lon2, lat2):
    # See https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# Load grid
grid_file = xr.open_dataset(dirs['traj'] + 'coral_grid.nc')
grid_dim = [grid_file.lon_rho_w.values.min(),
            grid_file.lon_rho_w.values.max(),
            grid_file.lat_rho_w.values.min(),
            grid_file.lat_rho_w.values.max()]

lon_bnd = np.arange(grid_dim[0], grid_dim[1]+1/2, 1/2)
lat_bnd = np.arange(grid_dim[2], grid_dim[3]+1/2, 1/2)
lon = 0.5*(lon_bnd[:-1] + lon_bnd[1:])
lat = 0.5*(lat_bnd[:-1] + lat_bnd[1:])

# Create empty LPDF dict
lpdf_winds = {}
lpdf_cmems = {}
lpdf_ratio = {}

tot_drifters_winds = {}
tot_drifters_cmems = {}

for code in code_list:
    # Create empty LPDF
    lpdf_winds[code] = np.zeros((len(lat), len(lon)), dtype=float)
    lpdf_cmems[code] = np.zeros((len(lat), len(lon)), dtype=float)

    # Firstly, create a PDF based on the virtual drifter trajectories
    winds_traj_fh_list = sorted(glob(dirs['traj'] + 'WINDS_FLOAT_' + code + '*.nc'))
    cmems_traj_fh_list = sorted(glob(dirs['traj'] + 'CMEMS_FLOAT_' + code + '*.nc'))

    # Total drifters
    tot_drifter_winds = 0
    tot_drifter_cmems = 0

    for traj_fh in tqdm(winds_traj_fh_list, total=len(winds_traj_fh_list)):
        traj = xr.open_dataset(traj_fh)

        # Just get all points for now
        id_list_template = np.arange(len(traj.idx0))
        id_list = np.zeros_like(traj.lon.values)
        id_list[:] = id_list_template.reshape((-1, 1))
        id_list = id_list.flatten()
        lon_list = traj.lon.values.flatten()
        lat_list = traj.lat.values.flatten()

        # Grid
        freq_matrix = np.histogramdd((lon_list, lat_list, id_list),
                                     bins=(lon_bnd, lat_bnd, np.arange(len(traj.idx0)+1)-0.5))[0]
        freq_matrix[freq_matrix > 1] = 1
        lpdf_winds[code] += freq_matrix.sum(axis=-1).T

        # Count number of drifters
        tot_drifter_winds += len(traj.idx0)

    for traj_fh in tqdm(cmems_traj_fh_list, total=len(cmems_traj_fh_list)):
        traj = xr.open_dataset(traj_fh)

        # Just get all points for now
        id_list_template = np.arange(len(traj.idx0))
        id_list = np.zeros_like(traj.lon.values)
        id_list[:] = id_list_template.reshape((-1, 1))
        id_list = id_list.flatten()
        lon_list = traj.lon.values.flatten()
        lat_list = traj.lat.values.flatten()

        # Grid
        freq_matrix = np.histogramdd((lon_list, lat_list, id_list),
                                     bins=(lon_bnd, lat_bnd, np.arange(len(traj.idx0)+1)-0.5))[0]
        freq_matrix[freq_matrix > 1] = 1
        lpdf_cmems[code] += freq_matrix.sum(axis=-1).T

        # Count number of drifters
        tot_drifter_cmems += len(traj.idx0)

    # Convert to proportion of drifters
    lpdf_winds[code] = lpdf_winds[code]/tot_drifter_winds
    lpdf_cmems[code] = lpdf_cmems[code]/tot_drifter_cmems
    tot_drifters_winds[code] = tot_drifter_winds
    tot_drifters_cmems[code] = tot_drifter_cmems

    lpdf_ratio[code] = lpdf_winds[code] - lpdf_cmems[code]

    # lpdf_cmems[code][lpdf_cmems[code] == 0] = -1
    # lpdf_ratio[code] = lpdf_winds[code]/lpdf_cmems[code]
    # lpdf_ratio[code][lpdf_ratio[code] < 0] = 1e2

    ###########################################################################
    # PLOTTING ################################################################
    ###########################################################################

f = plt.figure(constrained_layout=True, figsize=(10, 10))

gs = GridSpec(4, 2, figure=f, height_ratios=[1, 1, 1, 0.1])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, 1], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[3, :]))
gl = []

for ax_pos, code in enumerate(code_list):
    lsm = cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor='w', zorder=1)
    pcm = ax[ax_pos].pcolormesh(lon_bnd, lat_bnd, lpdf_ratio[code],
                                cmap=cmr.wildfire,
                                norm=colors.SymLogNorm(vmin=-1, vmax=1, linthresh=1e-3),
                                transform=ccrs.PlateCarree())

    ax[ax_pos].add_feature(lsm)

    gl.append(ax[ax_pos].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                   linewidth=0.5, color='k', linestyle='--', zorder=11))
    gl[-1].xlocator = mticker.FixedLocator(np.arange(-210, 210, 10))
    gl[-1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))

    if ax_pos in [0, 2, 4]:
        gl[-1].ylabels_left = True
    else:
        gl[-1].ylabels_left = False

    if ax_pos in [4, 5]:
        gl[-1].xlabels_bottom = True
    else:
        gl[-1].xlabels_bottom = False

    gl[-1].xlabels_top = False
    gl[-1].ylabels_right = False
    gl[-1].ylabel_style = {'size': 12}
    gl[-1].xlabel_style = {'size': 12}

    # Now load the relevant GDP drifter trajectories
    gdp_traj = pd.read_csv(dirs['gdp'] + code + '_GDP.csv')
    gdp_id_list = np.unique(gdp_traj['ID']).astype(int)
    n_drifters = len(gdp_id_list)

    # Plot GDP trajectories
    for gdp_id in gdp_id_list:
        gdp_traj_subset = gdp_traj[gdp_traj['ID'] == gdp_id]
        lat_subset = gdp_traj_subset['Latitude'].values
        lon_subset = gdp_traj_subset['Longitude'].values

        # Plot
        ax[ax_pos].plot(lon_subset, lat_subset, linewidth=0.25, color='w',
                        alpha=1, transform=ccrs.PlateCarree())

    ax[ax_pos].set_ylim([lat.min(), 0])
    ax[ax_pos].set_xlim([lon.min(), lon.max()])

    ax[ax_pos].set_title(name_dict[code], fontsize=16)

    ax[ax_pos].set_facecolor('gray')

    if code in ['MAH', 'CHA']:
        ax[ax_pos].text(76.9, -23.0, str(n_drifters) + ' GDP drifters', va='bottom', ha='right', fontsize=10, fontweight='bold', color='w')
        ax[ax_pos].text(76.9, -21.4, str(int(tot_drifters_cmems[code]/1e3)) + 'k virtual CMEMS drifters', va='bottom', ha='right', fontsize=10, fontweight='bold', color='w')
        ax[ax_pos].text(76.9, -19.7, str(int(tot_drifters_winds[code]/1e3)) + 'k virtual WINDS drifters', va='bottom', ha='right', fontsize=10, fontweight='bold', color='w')
    else:
        ax[ax_pos].text(76.9, -0.5, str(n_drifters) + ' GDP drifters', va='top', ha='right', fontsize=10, fontweight='bold', color='w')
        ax[ax_pos].text(76.9, -2.1, str(int(tot_drifters_cmems[code]/1e3)) + 'k virtual CMEMS drifters', va='top', ha='right', fontweight='bold', fontsize=10, color='w')
        ax[ax_pos].text(76.9, -3.7, str(int(tot_drifters_winds[code]/1e3)) + 'k virtual WINDS drifters', va='top', ha='right', fontweight='bold', fontsize=10, color='w')


cb = plt.colorbar(pcm, cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('WINDS/CMEMS drifter presence likelihood', size=14)
ax[-1].tick_params(axis='x', labelsize=12)

plt.savefig(dirs['fig'] + 'GDP_validation_comparison_symlog.pdf', bbox_inches='tight', dpi=300)
