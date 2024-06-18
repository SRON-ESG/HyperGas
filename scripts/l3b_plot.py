#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Plot CH4 L3B plume NetCDF products and save them as png files."""

from matplotlib import rcParams
import gc
import re
import logging
import os
from ast import literal_eval
from glob import glob
from itertools import chain
import numpy as np

import cartopy.crs as ccrs
import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import get_dirs

import warnings
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
rcParams['font.sans-serif'] = ['tex gyre heros']

font_size = 15
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

def plot_data(filename, savename):
    """Plot L3 data"""
    LOG.info(f'Plotting {filename}')

    # read nc and csv plume files
    ds = xr.open_dataset(filename)
    df = pd.read_csv(filename.replace('.nc', '.csv'), converters={'plume_bounds': literal_eval})

    # copy for plotting
    ds_all = ds.copy()

    # subset data to plume
    gas = df['gas'].item().lower()
    ds = ds.where(~ds[gas].isnull(), drop=True)

    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           subplot_kw={'projection': proj},
                           # constrained_layout=True
                           )

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

    ax.set_extent((lon_min, lon_max, lat_min, lat_max), crs=proj)

    # add high-res background
    cx.add_basemap(ax, crs=proj, source=cx.providers.Esri.WorldImagery)
    # remove watermark
    ax.texts[0].remove()

    # set colorbar limit
    if gas == 'ch4':
        vmax = 300  # ppb
    elif gas == 'co2':
        vmax = 10  # ppm
    else:
        raise ValueError("Please set cmap limit for {gas} here.")

    # plot rgb and gas data
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

    cb_label = replace_number_with_subscript(f'{gas.upper()} Enhancement (ppb)')
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

    # plot wind quivers
    q_era5 = ax.quiver(lon_min+pad/10, lat_min+pad/3,
                       u_era5, v_era5, transform=proj, color='w',
                       )
    q_geosfp = ax.quiver(lon_min+pad/10, lat_min+pad/10,
                         u_geosfp, v_geosfp, transform=proj, color='w',
                         )

    ax.text(lon_min+pad/4, lat_min+pad/4, f'{np.round(wspd_era5, 2)} m/s (ERA5)', transform=proj, color='w', fontsize=10, weight='bold')
    ax.text(lon_min+pad/4, lat_min+pad/9, f'{np.round(wspd_geosfp, 2)} m/s (GEOS-FP)', transform=proj, color='w', fontsize=10, weight='bold')

    # # calculate wind max for wind arrow legend
    # wspd_max = int(max(wspd_era5, wspd_geosfp))
    # ax.quiverkey(q_era5, 0.8, 0.1, wspd_max, f'{wspd_max} m/s (ERA5)', fontproperties={'size': 8}, labelcolor='w')
    # ax.quiverkey(q_geosfp, 0.8, 0, wspd_max, f'{wspd_max} m/s (GEOS-FP)', fontproperties={'size': 8}, labelcolor='w')

    # add shape error if the source type is waste
    if df['ipcc_sector'].item() == 'Solid Waste (6A)':
        df['emission_uncertainty'] = np.sqrt(df['emission_uncertainty']**2+(df['emission']*0.05)**2)

    title = df['datetime'].item().replace('T', ' ') + '\n' \
            + 'Lat: ' + str(df['plume_latitude'].item()) + ' Lon: ' + str(df['plume_longitude'].item()) + '\n' \
            + str(round(df['emission'].item()/1e3, 2)) + ' t/h $\pm$ ' \
            + str(round(df['emission_uncertainty']/df['emission']*100, 2).item()) + '%'

    # add name to title if exists
    if not df['name'].isnull().item():
        title = df['name'].item() + '\n' + title

    ax.set_title('')
    plt.suptitle(title)

    LOG.info(f'Exported to {savename}')
    fig.tight_layout()
    fig.savefig(savename, dpi=300)

    del ds, df, fig
    gc.collect()


def main(skip_exist=True):
    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    if len(filelist) == 0:
        return

    for filename in filelist:
        savename = filename.replace('.nc', '.png')
        if skip_exist:
            if os.path.isfile(savename):
                LOG.info(f'{savename} exists, skip ...')
            else:
                plot_data(filename, savename)
        else:
            plot_data(filename, savename)


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI_landfill/'
    lowest_dirs = get_dirs(root_dir)

    # whether skip dir which contains exported html
    skip_exist = True

    for data_dir in lowest_dirs:
        LOG.info(f'Plotting data under {data_dir}')
        main(skip_exist)
