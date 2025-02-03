!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Plot CH4 L3B plume NetCDF products and save them as png files."""

import gc
import logging
import os
import re
import warnings
from ast import literal_eval
from glob import glob
from itertools import chain

import cartopy.crs as ccrs
import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.crs import epsg as ccrs_from_epsg
from hypergas.plume_utils import plot_mask
from matplotlib import rcParams
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry.point import Point

from utils import get_dirs

warnings.filterwarnings("ignore")

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# set filename pattern to load data automatically
PATTERNS = ['ENMAP01-____L3B*.nc', 'EMIT_L3B*.nc', 'PRS_L3_*.nc']

# set basic matplotlib settings
rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['tex gyre heros']

font_size = 10
rcParams['axes.titlesize'] = font_size
rcParams['axes.labelsize'] = font_size - 2
rcParams['xtick.labelsize'] = font_size - 2
rcParams['ytick.labelsize'] = font_size - 2
rcParams['legend.fontsize'] = font_size
rcParams['figure.titlesize'] = font_size
rcParams['figure.titleweight'] = 'bold'


def replace_number_with_subscript(input_string):
    def replace(match):
        return match.group(1) + '$_{' + match.group(2) + '}$'

    return re.sub(r'([A-Za-z]+)(\d+)', replace, input_string)


def get_cartopy_crs_from_epsg(epsg_code):
    if epsg_code:
        try:
            return ccrs_from_epsg(epsg_code)
        except ValueError:
            if epsg_code == 4326:
                return ccrs.PlateCarree()
            else:
                raise NotImplementedError('The show_map() method currently does not support the given '
                                          'projection.')
    else:
        raise ValueError(f'Expected a valid EPSG code. Got {epsg_code}.')


def sron_ime(fig, ax, ds_all, ds, df, gas, proj, plot_minimal, pad=None):
    """Plot SRON IME results"""
    if pad is None:
        # calculate the pad around source center
        lon_min, lat_min, lon_max, lat_max = df['plume_bounds'].item()
        lon_width = lon_max - lon_min
        lat_width = lat_max - lat_min

        pad = max(lon_width, lat_width)

    # set extent
    lon_min = df['plume_longitude'] - pad
    lon_max = df['plume_longitude'] + pad
    lat_min = df['plume_latitude'] - pad
    lat_max = df['plume_latitude'] + pad
    extent = (lon_min, lon_max, lat_min, lat_max)

    ax.set_extent(extent, crs=proj)

    # add high-res background
    cx.add_basemap(ax, crs=proj, source=cx.providers.Esri.WorldImagery)
    # remove watermark
    ax.texts[0].remove()

    # set colorbar limit
    if gas == 'ch4':
        base = 50  # ppb
        # vmax = 300  # ppb
    elif gas == 'co2':
        base = 5  # ppm
        # vmax = 50  # ppm
    else:
        raise ValueError("Please set cmap limit for {gas} here.")

    # plot rgb and gas data
    # auto vmax (upround)
    vmax = ds_all[gas].quantile(0.99)
    vmax = int(np.ceil(vmax / base)) * base

    m = ds_all[gas].plot(x='longitude', y='latitude', vmin=0, vmax=vmax, cmap='plasma', add_colorbar=False,
                         # cbar_kwargs={'label': 'CH$_4$ Enhancement (ppb)', 'orientation': 'horizontal', 'shrink': 0.7}
                         )

    # add colorbar
    # https://stackoverflow.com/a/47790537/7347925
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes("bottom", size="7%", pad="2%", axes_class=plt.Axes)
    fig.add_axes(ax_cb)

    out_bnd = (ds[gas] < 0).any().item() + (ds[gas] > vmax).any().item()

    if out_bnd == 0:
        extend = None
    elif out_bnd == 2:
        extend = 'both'
    elif (ds[gas] < 0).any():
        extend = 'min'
    else:
        extend = 'max'

    cb_label = replace_number_with_subscript(f"{gas.upper()} Enhancement ({ds_all[gas].attrs['units']})")
    plt.colorbar(m, cax=ax_cb, extend=extend, label=cb_label, orientation='horizontal')

    # add source point marker
    ax.scatter(df['plume_longitude'], df['plume_latitude'], color='yellow',
               linewidth=2, marker='o', fc='none', s=200)

    # get wind info
    u_era5 = ds['u10'].sel(source='ERA5').mean()
    v_era5 = ds['v10'].sel(source='ERA5').mean()
    u_geosfp = ds['u10'].sel(source='GEOS-FP').mean()
    v_geosfp = ds['v10'].sel(source='GEOS-FP').mean()
    wspd_era5 = np.sqrt(u_era5**2+v_era5**2).item()
    wspd_geosfp = np.sqrt(u_geosfp**2+v_geosfp**2).item()

    if plot_minimal:
        title = df['datetime'].item().replace('T', ' ') + '\n' \
            + str(round(df['emission'].item()/1e3, 2)) + ' t/h $\pm$ ' \
            + str(round(df['emission_uncertainty']/df['emission']*100, 2).item()) + '%'
    else:
        # plot wind quivers
        q_era5 = ax.quiver(lon_min+pad/10, lat_min+pad/3,
                           u_era5, v_era5, transform=proj, color='w',
                           )
        q_geosfp = ax.quiver(lon_min+pad/10, lat_min+pad/10,
                             u_geosfp, v_geosfp, transform=proj, color='w',
                             )
        ax.text(lon_min+pad/4, lat_min+pad/4, f'{np.round(wspd_era5, 2)} m/s (ERA5)',
                transform=proj, color='w', fontsize=10, weight='bold')
        ax.text(lon_min+pad/4, lat_min+pad/9, f'{np.round(wspd_geosfp, 2)} m/s (GEOS-FP)',
                transform=proj, color='w', fontsize=10, weight='bold')

        title = df['datetime'].item().replace('T', ' ') + '\n' \
            + 'Lat: ' + str(df['plume_latitude'].round(4).item()) + ' Lon: ' + str(df['plume_longitude'].round(4).item()) + '\n' \
            + str(round(df['emission'].item()/1e3, 2)) + ' t/h $\pm$ ' \
            + str(round(df['emission_uncertainty']/df['emission']*100, 2).item()) + '%'

    # add name to title if exists
    if not df['name'].isnull().item():
        title = str(df['name'].item()) + '\n' + title
    ax.set_title(title, fontweight='bold')

    return extent, vmax


def sron_csf(fig, ax_ime, ax_csf, ds_csf, df):
    """Plot SRON CSF results"""
    if ds_csf.rio.crs:
        data_proj = get_cartopy_crs_from_epsg(ds_csf.rio.crs.to_epsg())
        transform = data_proj
    else:
        transform = None

    # plot csf lines
    ax_ime.plot(xr.concat([ds_csf['x_start'], ds_csf['x_end']], dim='loc'),
                xr.concat([ds_csf['y_start'], ds_csf['y_end']], dim='loc'),
                transform=transform, c='skyblue', alpha=0.3,
                )

    # plot centerline
    ax_ime.plot(ds_csf['x_center'], ds_csf['y_center'], transform=transform, c='darkorange', alpha=0.5)

    # --- plot emission rates ---
    (ds_csf['emission_rate']/1e3).plot(ax=ax_csf, c='C0')
    ax_csf.axhline(y=ds_csf['emission_rate'].mean()/1e3, c='orange', linestyle='--')
    ax_csf.set_xlabel('CSF lines')
    ax_csf.set_ylabel('Emission Rate (t h$^{-1}$)', c='C0')
    title = str(round(df['emission_csf'].item()/1e3, 2)) + ' t/h $\pm$ ' \
        + str(round(df['emission_csf_uncertainty']/df['emission_csf']*100, 2).item()) + '%'
    ax_csf.set_title(f'CSF: {title}', fontweight='bold')

    ds_csf.close()


def cm_ime(fig, ax, ds_all, ds, df, gas, proj, plot_minimal, extent, vmax):
    """Plot Carbon Mapper IME results"""
    ax.set_extent(extent, crs=proj)

    # add high-res background
    cx.add_basemap(ax, crs=proj, source=cx.providers.Esri.WorldImagery)
    # remove watermark
    ax.texts[0].remove()

    # plot rgb and gas data
    gas = gas+'_cm'
    m = ds_all[gas].plot(x='longitude', y='latitude', vmin=0, vmax=vmax, cmap='plasma', add_colorbar=False,
                         # cbar_kwargs={'label': 'CH$_4$ Enhancement (ppb)', 'orientation': 'horizontal', 'shrink': 0.7}
                         )

    # add source point marker
    ax.scatter(df['plume_longitude'], df['plume_latitude'], color='yellow',
               linewidth=2, marker='o', fc='none', s=200)

    title = 'CM IME: ' + str(round(df['emission_cm'].item()/1e3, 2)) + ' t/h'  # + ' t/h $\pm$ ' \
    # + str(round(df['emission_uncertainty']/df['emission']*100, 2).item()) + '%'

    ax.set_title(title, fontweight='bold')


def plot_data(filename, savename, plot_csf, plot_cm, plot_minimal, plot_full_field, pad):
    """Plot L3 data"""
    LOG.info(f'Plotting {filename}')

    # read nc and csv plume files
    ds = xr.open_dataset(filename)
    df = pd.read_csv(filename.replace('.nc', '.csv'), converters={'plume_bounds': literal_eval})
    gas = df['gas'].item().lower()
    plume_longitude = df['plume_longitude'].item()
    plume_latitude = df['plume_latitude'].item()

    # --- plot html ---
    l2b_filename = ('_'.join(filename.split('_')[:-1])+'.nc').replace('L3', 'L2')
    plume_num = re.search('plume(.*).nc', os.path.basename(filename)).group(1)
    plume_name = 'plume' + plume_num
    plot_mask(filename=l2b_filename.replace('.nc', '.html'),
              ds=ds,  # read L3 plume
              gas=gas,
              mask=np.full(ds[gas].shape, True),  # L3 data is already masked
              lon_target=plume_longitude,
              lat_target=plume_latitude,
              pick_plume_name=plume_name,
              only_plume=True)

    # --- plot scientific png ---
    # copy for plotting
    if plot_full_field:
        # read the L2 data
        ds_all = xr.open_dataset(l2b_filename)
    else:
        ds_all = ds.copy()

    # subset data to plume
    ds = ds.where(~ds[gas].isnull(), drop=True)

    proj = ccrs.PlateCarree()

    fig = plt.figure(layout='compressed')

    # only plot csf if CSF data is available
    if plot_minimal:
        plot_csf = False
    elif (plot_csf) & ('emission_csf' in df.columns):
        if df['emission_csf'].notnull().item():
            plot_csf = True
        else:
            plot_csf = False
    else:
        plot_csf = False

    # only plot Carbon Mapper IME results when cm data is available
    if plot_minimal:
        plot_cm = False
    elif (plot_cm) & ('emission_cm' in df.columns):
        if df['emission_cm'].notnull().item():
            plot_cm = True
        else:
            plot_cm = False
    else:
        plot_cm = False

    if plot_csf & plot_cm:
        ncols = 3
    elif plot_csf | plot_cm:
        ncols = 2
    else:
        ncols = 1
    ax_ime = fig.add_subplot(1, ncols, 1, projection=ccrs.PlateCarree())

    # plot SRON IME results
    extent, vmax = sron_ime(fig, ax_ime, ds_all, ds, df, gas, proj, plot_minimal, pad)

    # plot SRON csf results
    if plot_csf:
        # read csf file
        ds_csf = xr.open_dataset(filename.replace('.nc', '_csf.nc'), decode_coords='all')
        ax_csf = fig.add_subplot(1, ncols, 2)
        sron_csf(fig, ax_ime, ax_csf, ds_csf, df)

    # plot CM IME results
    if plot_cm:
        ax_cm = fig.add_subplot(1, ncols, ncols, projection=ccrs.PlateCarree())
        cm_ime(fig, ax_cm, ds, ds_all, df, gas, proj, plot_minimal, extent, vmax)

    # add scalebar
    # Geographic WGS 84 - degrees
    scale_points = gpd.GeoSeries([Point(plume_longitude-1, plume_latitude),
                                  Point(plume_longitude, plume_latitude)],
                                 crs=4326)
    # UTM projection
    utm_crs_list = query_utm_crs_info(
        datum_name='WGS 84',
        area_of_interest=AreaOfInterest(
            west_lon_degree=plume_longitude,
            south_lat_degree=plume_latitude,
            east_lon_degree=plume_longitude,
            north_lat_degree=plume_latitude,
        ),
    )
    utm_epsg = CRS.from_epsg(utm_crs_list[0].code).to_epsg()

    # Projected WGS 84 - meters
    scale_points = scale_points.to_crs(utm_epsg)
    distance_meters = scale_points[0].distance(scale_points[1])
    if plot_minimal:
        scale_bar_location = 'lower left'
    else:
        # the left position is saved for wind info
        scale_bar_location = 'lower right'

    scalebar = ScaleBar(distance_meters,
                        location=scale_bar_location,
                        color='white',
                        box_alpha=0,
                        font_properties={'size': 10},
                        )

    ax_ime.add_artist(scalebar)

    LOG.info(f'Exported to {savename}')
    fig.savefig(savename, bbox_inches='tight', pad_inches=0, dpi=300)

    del ds, df, fig
    gc.collect()


def main(skip_exist=True, plot_csf=True, plot_minimal=False, plot_full_field=False, pad=None):
    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))

    # only read plume NetCDF file
    filelist = [file for file in filelist if re.search(r'_plume\d+\.nc$', file)]

    # sort list
    filelist = list(sorted(filelist))

    if len(filelist) == 0:
        return

    for filename in filelist:
        savename = filename.replace('.nc', '.png')
        if skip_exist:
            if os.path.isfile(savename):
                LOG.info(f'{savename} exists, skip ...')
            else:
                plot_data(filename, savename, plot_csf, plot_cm, plot_minimal, plot_full_field, pad)
        else:
            plot_data(filename, savename, plot_csf, plot_cm, plot_minimal, plot_full_field, pad)


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI_landfill/'
    lowest_dirs = get_dirs(root_dir)

    # whether skip dir which contains exported png file
    skip_exist = True

    # whether plot CSF results
    plot_csf = True

    # whether plot reproduced Carbon Mapper IME results
    plot_cm = True

    # whether plot minimal image (only title (sitename+datetime+emission_rate) and plume)
    plot_minimal = False

    # pad (degree) around the plume source
    #   if it is None, we will set the pad based on the plume automatically
    pad = None

    # whether plot the full field instead of plume
    plot_full_field = False

    for data_dir in lowest_dirs:
        LOG.info(f'Plotting data under {data_dir}')
        main(skip_exist, plot_csf, plot_minimal, plot_full_field, pad)
