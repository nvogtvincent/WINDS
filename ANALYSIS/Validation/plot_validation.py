#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to validate various metrics in WINDS
@author: Noam Vogt-Vincent
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmasher as cmr
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

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
fh = {}

# WINDS (files preprocessed with CDO for speed)
fh['winds_monclim'] = dirs['data'] + 'WINDS-M_SUPP1D_MONCLIM.nc'         # Monthly climatology of full WINDS-M_SUPP1D 1993-2020 (w/ cdo ymonmean)
fh['winds_monmean'] = dirs['data'] + 'WINDS-M_SUPP1D_MONMEAN.nc'         # Monthly mean of full WINDS-M_SUPP1D 1993-2020 (w/ cdo monmean)
fh['winds_grid'] = dirs['grid'] + 'croco_grd.nc'                         # WINDS grid file
fh['winds_vel_monclim'] = dirs['data'] + 'WINDS_sfc_spd_monclim.nc'      # Monthly climatology of full WINDS-M_SFC (interpolated onto a regular 1/50deg grid with cdo remapbil, then calculate velocity magnitude, then ymonmean)
fh['winds_vel_std'] = dirs['data'] + 'WINDS_sfc_spd_daily_std.nc'        # Standard deviation of full WINDS-M_SFC (interpolated onto a regular 1/50deg grid with cdo remapbil, then calculate daily means, then calculate velocity magnitude, then timstd)
fh['winds_uv_monclim'] = dirs['data'] + 'WINDS_sfc_spd_monclim_full.nc'  # Monthly climatology of full WINDS-M_SFC (interpolated onto a regular 1/50deg grid with cdo remapbil, then ymonmean)
fh['winds_zeta_std'] = dirs['data'] + 'WINDS-M_SUPP1D_ZETA_STD.nc'       # Standard deviation of full WINDS-M_SUPP1D.zeta (with cdo timstd)
fh['winds_rivers'] = dirs['grid'] + 'croco_runoff.nc'                    # WINDS runoff file

# COPERNICUS-GLOBCURRENT
fh['uv_alt_daily'] = dirs['data'] + 'dataset-uv-rep-daily.nc'            # doi:10.1002/2014GL061773

# GDP surface currents
fh['gdp_sfc_vel'] = dirs['data'] + 'drifter_monthlymeans.nc'             # doi:10.1016/j.dsr.2017.04.009

# MultiObs SST/SSS
fh['multiobs_monthly'] = dirs['data'] + 'dataset-armor-3d-rep-monthly.nc' # doi:10.48670/moi-00052
fh['multiobs_monclim'] = dirs['data'] + 'dataset-armor-3d-rep-monclim.nc' # doi:10.48670/moi-00052 (cdo ymonmean)

# MultiObs SSH
fh['ssh_daily'] = dirs['data'] + 'cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D.nc' # doi:10.48670/moi-00148

# OSTIA SST
fh['ostia_monclim'] = dirs['data'] + 'OSTIA_MONCLIM.nc' # doi:10.48670/moi-00168 (cdo ymonmean)
fh['ostia_monmean'] = dirs['data'] + 'OSTIA_MONMEAN.nc' # doi:10.48670/moi-00168 (cdo monmean)

winds_monmean = xr.open_dataset(fh['winds_monmean'])
winds_monmean = winds_monmean.assign_coords(x_rho=winds_monmean.nav_lon_rho.values[0, :],
                                            y_rho=winds_monmean.nav_lat_rho.values[:, 0]).drop(['nav_lon_rho', 'nav_lat_rho'])
winds_monmean = winds_monmean.rename({'x_rho': 'longitude', 'y_rho': 'latitude', 'time_counter': 'time'})

winds_monclim = xr.open_dataset(fh['winds_monclim'])
winds_monclim = winds_monclim.assign_coords(x_rho=winds_monclim.nav_lon_rho.values[0, :],
                                            y_rho=winds_monclim.nav_lat_rho.values[:, 0]).drop(['nav_lon_rho', 'nav_lat_rho'])
winds_monclim = winds_monclim.rename({'x_rho': 'longitude', 'y_rho': 'latitude', 'time_counter': 'time'})

winds_lsm = xr.open_dataset(fh['winds_grid']).mask_rho
winds_lsm = winds_lsm.rename({'xi_rho': 'longitude', 'eta_rho': 'latitude'})
winds_lsm = winds_lsm.assign_coords(longitude=winds_monmean.longitude.values,
                                    latitude=winds_monmean.latitude.values)

# MultiObs SST/SSS
multiobs_monmean = xr.open_dataset(fh['multiobs_monthly'])
multiobs_monclim = xr.open_dataset(fh['multiobs_monclim'])

# OSTIA SST
ostia_monmean = xr.open_dataset(fh['ostia_monmean'])
ostia_monclim = xr.open_dataset(fh['ostia_monclim'])
ostia_monmean = ostia_monmean.rename({'lon': 'longitude', 'lat': 'latitude'})
ostia_monclim = ostia_monclim.rename({'lon': 'longitude', 'lat': 'latitude'})

# WINDS VEL/ZETA
winds_vel_monclim = xr.open_dataset(fh['winds_vel_monclim'])
winds_vel_monclim = winds_vel_monclim.rename({'lon': 'longitude', 'lat': 'latitude'})
winds_vel_std = xr.open_dataset(fh['winds_vel_std']).speed[0,:,:].drop('time_counter')
winds_vel_std = winds_vel_std.rename({'lon': 'longitude', 'lat': 'latitude'})
winds_zeta_std = xr.open_dataset(fh['winds_zeta_std']).zeta[0,:,:].drop('time_counter')
winds_zeta_std = winds_zeta_std.assign_coords(x_rho=winds_zeta_std.nav_lon_rho.values[0, :],
                                            y_rho=winds_zeta_std.nav_lat_rho.values[:, 0]).drop(['nav_lon_rho', 'nav_lat_rho'])
winds_zeta_std = winds_zeta_std.rename({'x_rho': 'longitude', 'y_rho': 'latitude'})

# ALTIMETER-DERIVED UV
uv_sat = xr.open_dataset(fh['uv_alt_daily'])
vel_sat = np.sqrt(uv_sat.vo**2 + uv_sat.uo**2)[:,0,:,:].drop('depth')
vel_sat_monclim = vel_sat.groupby('time.month').mean(dim='time')
uv_sat_monclim = uv_sat.groupby('time.month').mean(dim='time')

# SSH
ssh_std = xr.open_dataset(fh['ssh_daily']).sla.std(dim='time')

# WINDS UV
winds_uv_monclim = xr.open_dataset(fh['winds_uv_monclim'])
winds_uv_monclim = winds_uv_monclim.assign_coords(x_u=winds_uv_monclim.nav_lon_u.values[0, :],
                                                  y_u=winds_uv_monclim.nav_lat_u.values[:, 0],
                                                  x_v=winds_uv_monclim.nav_lon_v.values[0, :],
                                                  y_v=winds_uv_monclim.nav_lat_v.values[:, 0]).drop(['nav_lon_rho', 'nav_lat_rho', 'time'])
winds_uv_monclim = winds_uv_monclim.interp({'y_u': uv_sat_monclim.latitude,
                                            'y_v': uv_sat_monclim.latitude,
                                            'x_u': uv_sat_monclim.longitude,
                                            'x_v': uv_sat_monclim.longitude})

# GDP VELOCITIES
uv_gdp = xr.open_dataset(fh['gdp_sfc_vel'], decode_times=False)
uv_gdp = uv_gdp.sel(longitude = (uv_gdp.Lon > 34)*(uv_gdp.Lon < 78),
                    latitude = (uv_gdp.Lat > -24)*(uv_gdp.Lat < 0.5))
uv_gdp = uv_gdp.rename({'Lon': 'longitude', 'Lat': 'latitude'})
uv_gdp = uv_gdp.interp_like(uv_sat_monclim)
vel_gdp = np.sqrt(uv_gdp.V**2 + uv_gdp.U**2)

###############################################################################
# PROCESS SFC_VEL #############################################################
###############################################################################

# Plot bathymetry
f = plt.figure(constrained_layout=True, figsize=(20.1, 10))
gs = GridSpec(1, 2, figure=f, width_ratios=[1, 0.03])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[0, 1]))
ax[0].set_facecolor('gray')
grd = xr.open_dataset(fh['winds_grid'])
rivers = xr.open_dataset(fh['winds_rivers'])
levels = np.arange(0, 5500, 250)
levels[0] = 25
bathy = ax[0].contourf(grd.lon_rho[0, :], grd.lat_rho[:, 0],
                       np.ma.masked_where(winds_lsm.values == 0, grd.h.values),
                       cmap=cmr.ocean, transform=ccrs.PlateCarree(),
                       levels=levels, zorder=-2)
ax[0].set_rasterization_zorder(-1)

# Also plot rivers
r_yi = np.array([258, 820, 206, 114, 414, 416, 1083, 554, 527, 231, 136, 888])
r_xi = np.array([90, 237, 490, 443, 585, 586, 295, 714, 690, 485, 692, 214])
r_name = ['Zambeze', 'Rufiji', 'Tsiribihina', 'Mangoky', 'Ikopa', 'Betsiboka',
          'Tana', 'Mahavavy Nord', 'Sambirano', 'Manambolo', 'Mananjary', 'Ruvu']
qbar_mean = rivers.Qbar.mean(dim='qbar_time')
r_lon = grd.lon_rho[0, :][r_xi].values
r_lat = grd.lat_rho[:, 0][r_yi].values
ax[0].scatter(r_lon, r_lat, s=qbar_mean, marker='o', facecolors='none',
              edgecolors='w', linewidth=2, transform=ccrs.PlateCarree())
ax[0].scatter(r_lon, r_lat, s=50, marker='.', c='w', transform=ccrs.PlateCarree())
gl = ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=0.5, color='k', linestyle='-', zorder=11)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 225, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 120, 5))
gl.ylabel_style = {'size': 24}
gl.xlabel_style = {'size': 24}
gl.xlabels_top = False
gl.ylabels_right = False

cb = plt.colorbar(bathy, cax=ax[1], fraction=0.05, orientation='vertical')
cb.set_label('Depth (m)', size=28)
ax[-1].tick_params(axis='y', labelsize=24)
plt.savefig(dirs['fig'] + 'winds_bathymetry.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot daily zeta variability
f = plt.figure(constrained_layout=True, figsize=(16, 20))
gs = GridSpec(3, 1, figure=f, height_ratios=[1, 1, 0.05])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, :]))

plot_list = []
gl = []

lon_bnd_w = np.append(winds_zeta_std.longitude, 2*winds_zeta_std.longitude[-1]-winds_zeta_std.longitude[-2])
lon_bnd_w -= 0.5*(lon_bnd_w[-1]-lon_bnd_w[-2])
lat_bnd_w = np.append(winds_zeta_std.latitude, 2*winds_zeta_std.latitude[-1]-winds_zeta_std.latitude[-2])
lat_bnd_w -= 0.5*(lat_bnd_w[-1]-lat_bnd_w[-2])

lon_bnd_c = np.append(ssh_std.longitude, 2*ssh_std.longitude[-1]-ssh_std.longitude[-2])
lon_bnd_c -= 0.5*(lon_bnd_c[-1]-lon_bnd_c[-2])
lat_bnd_c = np.append(ssh_std.latitude, 2*ssh_std.latitude[-1]-ssh_std.latitude[-2])
lat_bnd_c -= 0.5*(lat_bnd_c[-1]-lat_bnd_c[-2])

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

plot_list = []
gl = []

plot_list.append(ax[0].contourf(winds_zeta_std.longitude, winds_zeta_std.latitude, winds_zeta_std,
                                  cmap=cmr.ghostlight, levels=np.linspace(0, 0.25, 11),
                                  transform=ccrs.PlateCarree(), zorder=-2))
ax[0].set_title('WINDS', fontsize=32)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[0].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[0].ylabel_style = {'size': 24}
gl[0].xlabel_style = {'size': 24}
gl[0].xlabels_top = False
gl[0].ylabels_right = False
ax[0].add_feature(land_cst)

plot_list.append(ax[1].contourf(ssh_std.longitude, ssh_std.latitude, ssh_std,
                                  cmap=cmr.ghostlight, levels=np.linspace(0, 0.25, 11),
                                  transform=ccrs.PlateCarree(), zorder=-2))
ax[1].set_title('CMEMS Altimeter-derived', fontsize=32)

gl.append(ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[1].ylabel_style = {'size': 24}
gl[1].xlabel_style = {'size': 24}
gl[1].xlabels_top = False
gl[1].ylabels_right = False
ax[1].add_feature(land_cst)

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('Daily surface height standard deviation (m)', size=28)
ax[-1].tick_params(axis='x', labelsize=24)

plt.savefig(dirs['fig'] + 'zeta_std.pdf', bbox_inches='tight', dpi=300)
plt.close()


# Plot WINDS sfc velocity against GlobCurrent and GDP compilation (split into 3 figures)
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

lon_bnd_w = np.append(winds_vel_monclim.longitude, 2*winds_vel_monclim.longitude[-1]-winds_vel_monclim.longitude[-2])
lon_bnd_w -= 0.5*(lon_bnd_w[-1]-lon_bnd_w[-2])
lat_bnd_w = np.append(winds_vel_monclim.latitude, 2*winds_vel_monclim.latitude[-1]-winds_vel_monclim.latitude[-2])
lat_bnd_w -= 0.5*(lat_bnd_w[-1]-lat_bnd_w[-2])

lon_bnd_g = np.append(vel_gdp.longitude, 2*vel_gdp.longitude[-1]-vel_gdp.longitude[-2])
lon_bnd_g -= 0.5*(lon_bnd_g[-1]-lon_bnd_g[-2])
lat_bnd_g = np.append(vel_gdp.latitude, 2*vel_gdp.latitude[-1]-vel_gdp.latitude[-2])
lat_bnd_g -= 0.5*(lat_bnd_g[-1]-lat_bnd_g[-2])

lon_bnd_c = np.append(vel_sat_monclim.longitude, 2*vel_sat_monclim.longitude[-1]-vel_sat_monclim.longitude[-2])
lon_bnd_c -= 0.5*(lon_bnd_c[-1]-lon_bnd_c[-2])
lat_bnd_c = np.append(vel_sat_monclim.latitude, 2*vel_sat_monclim.latitude[-1]-vel_sat_monclim.latitude[-2])
lat_bnd_c -= 0.5*(lat_bnd_c[-1]-lat_bnd_c[-2])

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

for i, month in enumerate(['January', 'February', 'March', 'April']):
    # WINDS
    ax[3*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i].pcolormesh(lon_bnd_w, lat_bnd_w, winds_vel_monclim.speed[i, :, :],
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i].quiver(uv_sat_monclim.longitude[3::6], uv_sat_monclim.latitude[3::6],
                     winds_uv_monclim.u_surf[i, 3::6, 3::6],
                     winds_uv_monclim.v_surf[i, 3::6, 3::6],
                     units='inches', scale=2, color='w')
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

    plot_list.append(ax[3*i+1].pcolormesh(lon_bnd_c, lat_bnd_c, vel_sat_monclim[i, :, :],
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i+1].quiver(uv_sat_monclim.longitude[3::6], uv_sat_monclim.latitude[3::6],
                     uv_sat_monclim.uo[i, 0, 3::6, 3::6],
                     uv_sat_monclim.vo[i, 0, 3::6, 3::6],
                     units='inches', scale=2, color='w')
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
    # ALTIMETRY
    ax[3*i+2].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i+2].text(77, -3.5, 'GDP', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i+2].pcolormesh(lon_bnd_g, lat_bnd_g, vel_gdp[i, :, :].T,
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i+2].quiver(uv_gdp.longitude[3::6], uv_gdp.latitude[3::6],
                     uv_gdp.U[i, 3::6, 3::6].T,
                     uv_gdp.V[i, 3::6, 3::6].T,
                     units='inches', scale=2, color='w')
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
cb.set_label('Surface velocity (m/s)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'sfc_current_comp_1.pdf', bbox_inches='tight', dpi=300)
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

lon_bnd_w = np.append(winds_vel_monclim.longitude, 2*winds_vel_monclim.longitude[-1]-winds_vel_monclim.longitude[-2])
lon_bnd_w -= 0.5*(lon_bnd_w[-1]-lon_bnd_w[-2])
lat_bnd_w = np.append(winds_vel_monclim.latitude, 2*winds_vel_monclim.latitude[-1]-winds_vel_monclim.latitude[-2])
lat_bnd_w -= 0.5*(lat_bnd_w[-1]-lat_bnd_w[-2])

lon_bnd_g = np.append(vel_gdp.longitude, 2*vel_gdp.longitude[-1]-vel_gdp.longitude[-2])
lon_bnd_g -= 0.5*(lon_bnd_g[-1]-lon_bnd_g[-2])
lat_bnd_g = np.append(vel_gdp.latitude, 2*vel_gdp.latitude[-1]-vel_gdp.latitude[-2])
lat_bnd_g -= 0.5*(lat_bnd_g[-1]-lat_bnd_g[-2])

lon_bnd_c = np.append(vel_sat_monclim.longitude, 2*vel_sat_monclim.longitude[-1]-vel_sat_monclim.longitude[-2])
lon_bnd_c -= 0.5*(lon_bnd_c[-1]-lon_bnd_c[-2])
lat_bnd_c = np.append(vel_sat_monclim.latitude, 2*vel_sat_monclim.latitude[-1]-vel_sat_monclim.latitude[-2])
lat_bnd_c -= 0.5*(lat_bnd_c[-1]-lat_bnd_c[-2])

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

for i, month in enumerate(['May', 'June', 'July', 'August']):
    # WINDS
    ax[3*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i].pcolormesh(lon_bnd_w, lat_bnd_w, winds_vel_monclim.speed[i+4, :, :],
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i].quiver(uv_sat_monclim.longitude[3::6], uv_sat_monclim.latitude[3::6],
                     winds_uv_monclim.u_surf[i+4, 3::6, 3::6],
                     winds_uv_monclim.v_surf[i+4, 3::6, 3::6],
                     units='inches', scale=2, color='w')
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

    plot_list.append(ax[3*i+1].pcolormesh(lon_bnd_c, lat_bnd_c, vel_sat_monclim[i+4, :, :],
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i+1].quiver(uv_sat_monclim.longitude[3::6], uv_sat_monclim.latitude[3::6],
                     uv_sat_monclim.uo[i+4, 0, 3::6, 3::6],
                     uv_sat_monclim.vo[i+4, 0, 3::6, 3::6],
                     units='inches', scale=2, color='w')
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

    plot_list.append(ax[3*i+2].pcolormesh(lon_bnd_g, lat_bnd_g, vel_gdp[i+4, :, :].T,
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i+2].quiver(uv_gdp.longitude[3::6], uv_gdp.latitude[3::6],
                     uv_gdp.U[i+4, 3::6, 3::6].T,
                     uv_gdp.V[i+4, 3::6, 3::6].T,
                     units='inches', scale=2, color='w')
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
cb.set_label('Surface velocity (m/s)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'sfc_current_comp_2.pdf', bbox_inches='tight', dpi=300)
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

lon_bnd_w = np.append(winds_vel_monclim.longitude, 2*winds_vel_monclim.longitude[-1]-winds_vel_monclim.longitude[-2])
lon_bnd_w -= 0.5*(lon_bnd_w[-1]-lon_bnd_w[-2])
lat_bnd_w = np.append(winds_vel_monclim.latitude, 2*winds_vel_monclim.latitude[-1]-winds_vel_monclim.latitude[-2])
lat_bnd_w -= 0.5*(lat_bnd_w[-1]-lat_bnd_w[-2])

lon_bnd_g = np.append(vel_gdp.longitude, 2*vel_gdp.longitude[-1]-vel_gdp.longitude[-2])
lon_bnd_g -= 0.5*(lon_bnd_g[-1]-lon_bnd_g[-2])
lat_bnd_g = np.append(vel_gdp.latitude, 2*vel_gdp.latitude[-1]-vel_gdp.latitude[-2])
lat_bnd_g -= 0.5*(lat_bnd_g[-1]-lat_bnd_g[-2])

lon_bnd_c = np.append(vel_sat_monclim.longitude, 2*vel_sat_monclim.longitude[-1]-vel_sat_monclim.longitude[-2])
lon_bnd_c -= 0.5*(lon_bnd_c[-1]-lon_bnd_c[-2])
lat_bnd_c = np.append(vel_sat_monclim.latitude, 2*vel_sat_monclim.latitude[-1]-vel_sat_monclim.latitude[-2])
lat_bnd_c -= 0.5*(lat_bnd_c[-1]-lat_bnd_c[-2])

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

for i, month in enumerate(['September', 'October', 'November', 'December']):
    # WINDS
    ax[3*i].text(77, -0.8, month, fontsize=22, color='w', va='top', ha='right',
               fontweight='bold')
    ax[3*i].text(77, -3.5, 'WINDS', fontsize=18, color='w', va='top', ha='right')

    plot_list.append(ax[3*i].pcolormesh(lon_bnd_w, lat_bnd_w, winds_vel_monclim.speed[i+8, :, :],
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i].quiver(uv_sat_monclim.longitude[3::6], uv_sat_monclim.latitude[3::6],
                     winds_uv_monclim.u_surf[i+8, 3::6, 3::6],
                     winds_uv_monclim.v_surf[i+8, 3::6, 3::6],
                     units='inches', scale=2, color='w')
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

    plot_list.append(ax[3*i+1].pcolormesh(lon_bnd_c, lat_bnd_c, vel_sat_monclim[i+8, :, :],
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i+1].quiver(uv_sat_monclim.longitude[3::6], uv_sat_monclim.latitude[3::6],
                     uv_sat_monclim.uo[i+8, 0, 3::6, 3::6],
                     uv_sat_monclim.vo[i+8, 0, 3::6, 3::6],
                     units='inches', scale=2, color='w')

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

    plot_list.append(ax[3*i+2].pcolormesh(lon_bnd_g, lat_bnd_g, vel_gdp[i+8, :, :].T,
                                 cmap=cmr.eclipse, vmin=0, vmax=1,
                                 transform=ccrs.PlateCarree(), rasterized=True))
    ax[3*i+2].quiver(uv_gdp.longitude[3::6], uv_gdp.latitude[3::6],
                     uv_gdp.U[i+8, 3::6, 3::6].T,
                     uv_gdp.V[i+8, 3::6, 3::6].T,
                     units='inches', scale=2, color='w')

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
cb.set_label('Surface velocity (m/s)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'sfc_current_comp_3.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot sfc velocity variability
vel_sat_std = vel_sat.std(dim='time')

f = plt.figure(constrained_layout=True, figsize=(16, 20))
gs = GridSpec(3, 1, figure=f, height_ratios=[1, 1, 0.05])
ax = []
ax.append(f.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[1, 0], projection = ccrs.PlateCarree()))
ax.append(f.add_subplot(gs[2, :]))

plot_list = []
gl = []

lon_bnd_w = np.append(winds_vel_std.longitude, 2*winds_vel_std.longitude[-1]-winds_vel_std.longitude[-2])
lon_bnd_w -= 0.5*(lon_bnd_w[-1]-lon_bnd_w[-2])
lat_bnd_w = np.append(winds_vel_std.latitude, 2*winds_vel_std.latitude[-1]-winds_vel_std.latitude[-2])
lat_bnd_w -= 0.5*(lat_bnd_w[-1]-lat_bnd_w[-2])

lon_bnd_c = np.append(vel_sat_std.longitude, 2*vel_sat_std.longitude[-1]-vel_sat_std.longitude[-2])
lon_bnd_c -= 0.5*(lon_bnd_c[-1]-lon_bnd_c[-2])
lat_bnd_c = np.append(vel_sat_std.latitude, 2*vel_sat_std.latitude[-1]-vel_sat_std.latitude[-2])
lat_bnd_c -= 0.5*(lat_bnd_c[-1]-lat_bnd_c[-2])

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

plot_list = []
gl = []

plot_list.append(ax[0].contourf(winds_vel_std.longitude, winds_vel_std.latitude,
                                winds_vel_std, cmap=cmr.ghostlight,
                                levels=np.linspace(0, 0.5, 6),
                                transform=ccrs.PlateCarree()))
ax[0].set_title('WINDS', fontsize=28)

gl.append(ax[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[0].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[0].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[0].ylabel_style = {'size': 20}
gl[0].xlabel_style = {'size': 20}
gl[0].xlabels_top = False
gl[0].ylabels_right = False
ax[0].add_feature(land_cst)

plot_list.append(ax[1].contourf(vel_sat_std.longitude, vel_sat_std.latitude,
                                vel_sat_std, cmap=cmr.ghostlight,
                                levels=np.linspace(0, 0.5, 6),
                                transform=ccrs.PlateCarree()))
ax[1].set_title('Copernicus GlobCurrent Surface', fontsize=28)

gl.append(ax[1].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='w', linestyle='-', zorder=11))
gl[1].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
gl[1].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
gl[1].ylabel_style = {'size': 20}
gl[1].xlabel_style = {'size': 20}
gl[1].xlabels_top = False
gl[1].ylabels_right = False
ax[1].add_feature(land_cst)

# ax[0].set_rasterization_zorder(-1)
# ax[1].set_rasterization_zorder(-1)

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('Daily mean surface current standard deviation (m/s)', size=26)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'sfc_current_std.pdf', bbox_inches='tight', dpi=300)
plt.close()

###############################################################################
# PROCESS SST/SSS #############################################################
###############################################################################

# Plot SST monclim anomalies (OSTIA)
f = plt.figure(constrained_layout=True, figsize=(20, 16))
gs = GridSpec(5, 3, figure=f, height_ratios=[1, 1, 1, 1, 0.1])
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

lon_bnd = np.append(winds_monclim.longitude, 2*winds_monclim.longitude[-1]-winds_monclim.longitude[-2])
lon_bnd -= 0.5*(lon_bnd[-1]-lon_bnd[-2])
lat_bnd = np.append(winds_monclim.latitude, 2*winds_monclim.latitude[-1]-winds_monclim.latitude[-2])
lat_bnd -= 0.5*(lat_bnd[-1]-lat_bnd[-2])

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

for i, month in enumerate(['January', 'February', 'March', 'April', 'May',
                           'June', 'July', 'August', 'September', 'October',
                           'November', 'December']):


    winds_frame_sst = winds_monclim.temp_surf[i, :, :].where(winds_lsm == 1)
    obs_frame_sst = ostia_monclim.analysed_sst[i, :, :].interp_like(winds_frame_sst)-273.15
    sst_anom = winds_frame_sst - obs_frame_sst
    rms = (((winds_frame_sst - obs_frame_sst)**2).mean())**0.5
    mae = abs(winds_frame_sst - obs_frame_sst).mean()

    ax[i].text(77, -0.8, month, fontsize=24, color='k', va='top', ha='right',
               fontweight='bold')
    ax[i].text(77, -3.5, 'MAE: ' + str(np.round(mae.values, 2)) + 'C', fontsize=20, color='k', va='top', ha='right')

    plot_list.append(ax[i].contourf(sst_anom.longitude, sst_anom.latitude, sst_anom,
                                    cmap=cmr.fusion_r, levels=np.linspace(-1, 1, 11),
                                    transform=ccrs.PlateCarree(), zorder=-2))

    ax[i].add_feature(land_cst)

    gl.append(ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='k', linestyle='-', zorder=11))
    gl[i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[i].ylabel_style = {'size': 20}
    gl[i].xlabel_style = {'size': 20}
    gl[i].xlabels_top = False
    gl[i].ylabels_right = False

    if i in [0, 3, 6, 9]:
        gl[i].ylabels_left = True
    else:
        gl[i].ylabels_left = False
    if i in [9, 10, 11]:
        gl[i].xlabels_bottom = True
    else:
        gl[i].xlabels_bottom = False

    ax[i].set_rasterization_zorder(-1)

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('SST anomaly, C (WINDS - OSTIA)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'monclim_SST_OSTIA_error.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Plot SST monclim anomalies (MultiObs)
f = plt.figure(constrained_layout=True, figsize=(20, 16))
gs = GridSpec(5, 3, figure=f, height_ratios=[1, 1, 1, 1, 0.1])
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

coords = winds_monclim.temp_surf[0, :, :].coarsen(longitude=5, latitude=5, boundary='trim').mean()
lon_bnd = np.append(coords.longitude, 2*coords.longitude[-1]-coords.longitude[-2])
lon_bnd -= 0.5*(lon_bnd[-1]-lon_bnd[-2])
lat_bnd = np.append(coords.latitude, 2*coords.latitude[-1]-coords.latitude[-2])
lat_bnd -= 0.5*(lat_bnd[-1]-lat_bnd[-2])

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

for i, month in enumerate(['January', 'February', 'March', 'April', 'May',
                           'June', 'July', 'August', 'September', 'October',
                           'November', 'December']):


    winds_frame_sst = winds_monclim.temp_surf[i, :, :].where(winds_lsm == 1).coarsen(longitude=5, latitude=5, boundary='trim').mean()
    obs_frame_sst = multiobs_monclim.to[i, 0, :, :].interp_like(winds_frame_sst)
    sst_anom = winds_frame_sst - obs_frame_sst
    rms = (((winds_frame_sst - obs_frame_sst)**2).mean())**0.5
    mae = abs(winds_frame_sst - obs_frame_sst).mean()

    ax[i].text(77, -0.8, month, fontsize=24, color='k', va='top', ha='right',
               fontweight='bold')
    ax[i].text(77, -3.5, 'MAE: ' + str(np.round(mae.values, 2)) + 'C', fontsize=20, color='k', va='top', ha='right')

    plot_list.append(ax[i].contourf(sst_anom.longitude, sst_anom.latitude, sst_anom,
                                    cmap=cmr.fusion_r, levels=np.linspace(-1, 1, 11),
                                    transform=ccrs.PlateCarree(), zorder=-2))
    ax[i].add_feature(land_cst)

    gl.append(ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='k', linestyle='-', zorder=11))
    gl[i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[i].ylabel_style = {'size': 20}
    gl[i].xlabel_style = {'size': 20}
    gl[i].xlabels_top = False
    gl[i].ylabels_right = False

    if i in [0, 3, 6, 9]:
        gl[i].ylabels_left = True
    else:
        gl[i].ylabels_left = False
    if i in [9, 10, 11]:
        gl[i].xlabels_bottom = True
    else:
        gl[i].xlabels_bottom = False

    ax[i].set_rasterization_zorder(-1)

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('SST anomaly, C (WINDS - CMEMS MultiObs)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'monclim_SST_MultiObs_error.pdf', bbox_inches='tight', dpi=300)
plt.close()

# SSS
f = plt.figure(constrained_layout=True, figsize=(20, 16))
gs = GridSpec(5, 3, figure=f, height_ratios=[1, 1, 1, 1, 0.1])
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

coords = winds_monclim.salt_surf[0, :, :].coarsen(longitude=5, latitude=5, boundary='trim').mean()
lon_bnd = np.append(coords.longitude, 2*coords.longitude[-1]-coords.longitude[-2])
lon_bnd -= 0.5*(lon_bnd[-1]-lon_bnd[-2])
lat_bnd = np.append(coords.latitude, 2*coords.latitude[-1]-coords.latitude[-2])
lat_bnd -= 0.5*(lat_bnd[-1]-lat_bnd[-2])

land_cst = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='k',
                                        facecolor='w',
                                        zorder=1)

for i, month in enumerate(['January', 'February', 'March', 'April', 'May',
                           'June', 'July', 'August', 'September', 'October',
                           'November', 'December']):

    winds_frame_sss = winds_monclim.salt_surf[i, :, :].where(winds_lsm == 1).coarsen(longitude=5, latitude=5, boundary='trim').mean()
    obs_frame_sss = multiobs_monclim.so[i, 0, :, :].interp_like(winds_frame_sss)
    sss_anom = winds_frame_sss - obs_frame_sss
    rms = (((winds_frame_sss - obs_frame_sss)**2).mean())**0.5
    mae = abs(winds_frame_sss - obs_frame_sss).mean()

    ax[i].text(77, -0.8, month, fontsize=24, color='k', va='top', ha='right',
               fontweight='bold')
    ax[i].text(77, -3.5, 'MAE: ' + str(np.round(mae.values, 2)) + 'PSU', fontsize=20, color='k', va='top', ha='right')

    plot_list.append(ax[i].contourf(sss_anom.longitude, sss_anom.latitude, sss_anom,
                                    cmap=cmr.fusion_r, levels=np.linspace(-0.5, 0.5, 11),
                                    transform=ccrs.PlateCarree(), zorder=-2))
    ax[i].add_feature(land_cst)

    gl.append(ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=0.5, color='k', linestyle='-', zorder=11))
    gl[i].xlocator = mticker.FixedLocator(np.arange(-180, 225, 10))
    gl[i].ylocator = mticker.FixedLocator(np.arange(-90, 120, 10))
    gl[i].ylabel_style = {'size': 20}
    gl[i].xlabel_style = {'size': 20}
    gl[i].xlabels_top = False
    gl[i].ylabels_right = False

    if i in [0, 3, 6, 9]:
        gl[i].ylabels_left = True
    else:
        gl[i].ylabels_left = False
    if i in [9, 10, 11]:
        gl[i].xlabels_bottom = True
    else:
        gl[i].xlabels_bottom = False

    ax[i].set_rasterization_zorder(-1)

cb = plt.colorbar(plot_list[0], cax=ax[-1], fraction=0.05, orientation='horizontal')
cb.set_label('SSS anomaly, PSU (WINDS - ARMOR3D)', size=24)
ax[-1].tick_params(axis='x', labelsize=22)

plt.savefig(dirs['fig'] + 'monclim_SSS_error.pdf', bbox_inches='tight', dpi=300)
plt.close()

# Compute monthly anomalies
rms = {'sst': np.zeros(winds_monmean.time.shape),
       'sss': np.zeros(winds_monmean.time.shape)}
mae = {'sst': np.zeros(winds_monmean.time.shape),
       'sss': np.zeros(winds_monmean.time.shape)}

for t_i in tqdm(range(len(winds_monmean.time))):
    # Mask, coarsen, interpolate, and compare
    winds_frame_sst = winds_monmean.temp_surf[t_i, :, :].where(winds_lsm == 1)
    obs_frame_sst = ostia_monmean.analysed_sst[t_i,:,:].interp_like(winds_frame_sst)-273.15 # HiRes OSTIA data

    winds_frame_sss = winds_monmean.salt_surf[t_i, :, :].where(winds_lsm == 1).coarsen(longitude=10, latitude=10, boundary='trim').mean()
    obs_frame_sss = multiobs_monmean.so[t_i,0,:,:].interp_like(winds_frame_sss)

    rms['sst'][t_i] = (((winds_frame_sst - obs_frame_sst)**2).mean())**0.5
    mae['sst'][t_i] = abs(winds_frame_sst - obs_frame_sst).mean()
    rms['sss'][t_i] = (((winds_frame_sss - obs_frame_sss)**2).mean())**0.5
    mae['sss'][t_i] = abs(winds_frame_sss - obs_frame_sss).mean()

# Plot SST/SSS MAE
f = plt.figure(constrained_layout=True, figsize=(14, 10))
gs = GridSpec(2, 1, figure=f)
ax = []
ax.append(f.add_subplot(gs[0, 0])) # SST
ax.append(f.add_subplot(gs[1, 0])) # SSS

ax[0].plot(winds_monmean.time, rms['sst'], 'k-', label='RMS Error')
ax[0].plot(winds_monmean.time, mae['sst'], 'k--', label='MAE Error')
ax[0].set_xlim([winds_monmean.time[0], winds_monmean.time[-1]])
ax[0].set_ylabel('Error (C)', fontsize=18)
ax[0].tick_params(axis='x', labelsize=16)
ax[0].tick_params(axis='y', labelsize=16)
ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
ax[0].set_title('Sea Surface Temperature (WINDS - OSTIA)', fontsize=20)
leg1 = ax[0].legend(fontsize=18)
leg1.get_frame().set_linewidth(0)

ax[1].plot(winds_monmean.time, rms['sss'], 'k-', label='RMS Error')
ax[1].plot(winds_monmean.time, mae['sss'], 'k--', label='MAE Error')
ax[1].set_xlim([winds_monmean.time[0], winds_monmean.time[-1]])
ax[1].set_ylabel('Error (PSU)', fontsize=18)
ax[1].set_xlabel('Year', fontsize=18)
ax[1].tick_params(axis='x', labelsize=16)
ax[1].tick_params(axis='y', labelsize=16)
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)
ax[1].set_title('Sea Surface Salinity (WINDS - ARMOR3D)', fontsize=20)
leg2 = ax[1].legend(fontsize=18)
leg2.get_frame().set_linewidth(0)

plt.savefig(dirs['fig'] + 'monthly_SST_SSS_error.pdf', bbox_inches='tight', dpi=300)
plt.close()

