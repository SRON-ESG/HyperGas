#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Recalculate emission rates using L3B NetCDF files."""

import logging
import os
import re
import gc
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd
import xarray as xr
from hypergas.plume_utils import calc_emiss, calc_emiss_fetch, mask_data
from utils import get_dirs

# calculate IME (kg m-2)
mass = 16.04e-3  # molar mass CH4 [kg/mol]
mass_dry_air = 28.964e-3  # molas mass dry air [kg/mol]
grav = 9.8  # gravity (m s-2)

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# set global attrs for exported NetCDF file
AUTHOR = 'Xin Zhang'
EMAIL = 'xin.zhang@sron.nl; xinzhang1215@gmail.com'
INSTITUTION = 'SRON Netherlands Institute for Space Research'

# set filename pattern to load data automatically
PATTERNS = ['ENMAP01-____L3B*.csv', 'EMIT_L3B*.csv', 'PRS_L3_*.csv']

def reprocess_data(filename, reprocess_nc, wind_data):
    LOG.info('Reading csv and nc files ...')
    df = pd.read_csv(filename, dtype={'wind_weights': bool})
    plume_filename = filename.replace('.csv', '.nc')
    platform = df['platform'].item()

    # set pixel resolution
    if platform == 'EMIT':
        pixel_res = 60  # meter
    elif platform in ['EnMAP', 'PRISMA']:
        pixel_res = 30  # meter

    if wind_data == 'auto':
        wind_source = df['wind_source'].item()
        wspd = df['wind_speed'].item()
    elif wind_data in ['ERA5', 'GEOS-FP']:
        wind_source = wind_data
        ds = xr.open_dataset(plume_filename)
        u10 = ds['u10'].sel(source=wind_source).item()
        v10 = ds['v10'].sel(source=wind_source).item()
        wspd = np.sqrt(u10**2 + v10**2)
        ds.close()

    else:
        raise ValueError(f'{wind_data} is not supported. Please use "ERA5" or "GEOS-FP"')

    if reprocess_nc:
        # read L2B data
        l2b_filename = ('_'.join(plume_filename.split('_')[:-1])+'.nc').replace('L3', 'L2')
        ds_l2b = xr.open_dataset(l2b_filename, decode_coords='all')

        # read plume source location
        longitude = df['plume_longitude'].item()
        latitude = df['plume_latitude'].item()

        # read plume mask settings
        plume_varname = 'ch4_comb_denoise'
        land_only = True
        land_mask_source = 'GSHHS'
        only_plume = True
        plume_num = re.search('plume(.*).nc', os.path.basename(plume_filename)).group(1)
        pick_plume_name = 'plume' + plume_num
        html_filename = os.path.join(os.path.dirname(plume_filename),
                                     os.path.basename(plume_filename).replace(f'_{pick_plume_name}.nc', '.html')
                                     )

        wind_weights = df['wind_weights'].item()
        niter = df['niter'].item()
        size_median = df['size_median'].item()
        sigma_guass = df['sigma_guass'].item()
        quantile = df['quantile'].item()

        # create new mask
        mask, lon_mask, lat_mask, plume_html_filename = mask_data(html_filename, ds_l2b, longitude, latitude,
                                                                  pick_plume_name, plume_varname,
                                                                  wind_source, wind_weights, land_only, land_mask_source,
                                                                  niter, size_median, sigma_guass, quantile,
                                                                  only_plume)

        # mask data
        ch4_mask = ds_l2b['ch4'].where(mask)

        # calculate mean wind and surface pressure in the plume
        u10 = ds_l2b['u10'].where(mask).mean(dim=['y', 'x'])
        v10 = ds_l2b['v10'].where(mask).mean(dim=['y', 'x'])
        sp = ds_l2b['sp'].where(mask).mean(dim=['y', 'x'])

        # save useful number for attrs
        sza = ds_l2b['ch4'].attrs['sza']
        vza = ds_l2b['ch4'].attrs['vza']
        start_time = ds_l2b['ch4'].attrs['start_time']

        # merge data
        ds_merge = xr.merge([ch4_mask, u10, v10, sp])

        # add crs info
        if ds_l2b.rio.crs:
            ds_merge.rio.write_crs(ds_l2b.rio.crs, inplace=True)

        # clear attrs
        ds_merge.attrs = ''
        # set global attributes
        header_attrs = {'author': AUTHOR,
                        'email': EMAIL,
                        'institution': INSTITUTION,
                        'filename': ds_l2b.attrs['filename'],
                        'start_time': start_time,
                        'sza': sza,
                        'vza': vza,
                        'plume_longitude': longitude,
                        'plume_latitude': latitude,
                        }

        ds_merge.attrs = header_attrs

        LOG.info(f'Updating {plume_filename} ...')
        ds_merge.to_netcdf(plume_filename)

        # close file
        ds_l2b.close()

        # clean
        del ds_l2b, ds_merge
        gc.collect()

    LOG.info('Recalculating emission rates ...')
    wspd, wdir, l_eff, u_eff, IME, Q, Q_err, \
        err_random, err_wind  = calc_emiss(f_ch4_mask=plume_filename,
                                           pick_plume_name=df['plume_id'].item().split('-')[-1],
                                           pixel_res=pixel_res,
                                           alpha1=df['alpha1'].item(), alpha2=df['alpha2'].item(), alpha3=df['alpha3'].item(),
                                           wind_source=wind_source, wspd=wspd,
                                           land_only=True,
                                           )

    LOG.info('Recalculating emission rates (fetch) ...')
    Q_fetch, Q_fetch_err, err_ime_fetch, err_wind_fetch \
        = calc_emiss_fetch(f_ch4_mask=plume_filename,
                           pixel_res=pixel_res,
                           wind_source=wind_source,
                           wspd=wspd
                           )

    # calculate plume bounds
    with xr.open_dataset(plume_filename) as ds:
        plume_mask = ~ds['ch4'].isnull()
        lon_mask = ds['longitude'].where(plume_mask, drop=True)
        lat_mask = ds['latitude'].where(plume_mask, drop=True)

    bounds = [lon_mask.min().item(), lat_mask.min().item(),
              lon_mask.max().item(), lat_mask.max().item()]

    # update emission data
    LOG.info(f'Updating {filename} ...')
    df['wind_speed'] = wspd
    df['wind_direction'] = wdir
    df['leff_ime'] = l_eff
    df['ueff_ime'] = u_eff
    df['ime'] = IME
    df['plume_bounds'] = [bounds]
    df['emission'] = Q
    df['emission_uncertainty'] = Q_err
    df['emission_uncertainty_random'] = err_random
    df['emission_uncertainty_wind'] = err_wind
    df['emission_fetch'] = Q_fetch
    df['emission_fetch_uncertainty'] = Q_fetch_err
    df['emission_fetch_uncertainty_ime'] = err_ime_fetch
    df['emission_fetch_uncertainty_wind'] = err_wind_fetch

    df.to_csv(filename, index=False)


def main():
    # whether regenerate the L3 plume NetCDF file based on csv settings
    reprocess_nc = False

    # which wind source data to be applied
    wind_data = 'auto'  # 'auto', 'ERA5', or 'GEOS-FP'

    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    if len(filelist) == 0:
        return

    for filename in filelist:
        LOG.info(f'Reprocessing {filename} ...')
        reprocess_data(filename, reprocess_nc, wind_data)


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI_plume/'
    lowest_dirs = get_dirs(root_dir)

    if input("This script will update emission data in all csv files. Are you sure? (y/n)") != "y":
        exit()

    # whether skip dir which contains exported html
    for data_dir in lowest_dirs:
        LOG.info(f'Reprocessing data under {data_dir}')
        main()
