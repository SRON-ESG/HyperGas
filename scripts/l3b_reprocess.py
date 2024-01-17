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
import sys
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('../app')
from utils import mask_data

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


def get_dirs(root_dir):
    """Get all lowest directories"""
    lowest_dirs = list()

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d[0] == '.']
        if files and not dirs:
            lowest_dirs.append(root)

    return list(sorted(lowest_dirs))


def calc_wind_error(wspd, IME, l_eff,
                    alpha1, alpha2, alpha3,
                    uncertainty=0.5):
    """Calculate wind error with random distribution"""
    # Generate U10 distribution with 50% uncertainty
    wspd_distribution = np.random.normal(wspd, wspd * uncertainty, size=1000)

    # Calculate Ueff distribution
    u_eff_distribution = alpha1 * np.log(wspd) + alpha2 + alpha3 * wspd_distribution

    # Calculate Q distribution
    Q_distribution = u_eff_distribution * IME / l_eff

    # Calculate standard deviation of Q distribution
    wind_error = np.std(Q_distribution)

    return wind_error


def calc_random_err(ch4, ch4_mask, area, sp):
    """Calculate random error by moving plume around the whole scene"""
    # crop ch4 to valid region
    ch4_mask_crop = ch4_mask.where(~ch4_mask.isnull()).dropna(dim='y', how='all').dropna(dim='x', how='all')

    # get the shape of input data and mask
    bkgd_rows, bkgd_cols = ch4_mask.shape
    mask_rows, mask_cols = ch4_mask_crop.shape

    # Insert plume mask data at a random position
    IME_noplume = []

    while len(IME_noplume) <= 500:
        # Generate random row and column index to place b inside a
        row_idx = np.random.randint(0, bkgd_rows - mask_rows)
        col_idx = np.random.randint(0, bkgd_cols - mask_cols)

        if not np.any(ch4[row_idx:row_idx+mask_rows, col_idx:col_idx+mask_cols].isnull()):
            ch4_bkgd_mask = xr.zeros_like(ch4)
            ch4_bkgd_mask[row_idx:row_idx+mask_rows, col_idx:col_idx+mask_cols] = ch4_mask_crop.values
            ch4_bkgd_mask = ch4_bkgd_mask.fillna(0)
            IME_noplume.append(ch4.where(ch4_bkgd_mask, drop=True).sum().values *
                               1.0e-9 * (mass / mass_dry_air) * sp / grav * area)

    return np.array(IME_noplume).std()


def calc_emiss(df, ds, ds_l2b, land_only=True):
    """Calculate emission (modified from app)"""
    # read csv info
    platform = df['platform'].item()
    wind_source = df['wind_source'].item()
    alpha1 = df['alpha1'].item()
    alpha2 = df['alpha2'].item()
    alpha3 = df['alpha3'].item()

    # get the masked plume data
    ch4_mask = ds.dropna(dim='y', how='all').dropna(dim='x', how='all')['ch4']

    # area of pixel in m2
    if platform == 'EMIT':
        pixel_res = 60
    elif platform in ['EnMAP', 'PRISMA']:
        pixel_res = 30
    else:
        raise ValueError("{platform} is not supprted yet.")
    area = pixel_res*pixel_res

    # calculate Leff using the root method in meter
    plume_pixel_num = (~ch4_mask.isnull()).sum()
    l_eff = np.sqrt(plume_pixel_num * area).item()

    # calculate IME
    sp = ds['sp'].mean().item()  # use the mean surface pressure (Pa)
    delta_omega = ch4_mask * 1.0e-9 * (mass / mass_dry_air) * sp / grav
    IME = np.nansum(delta_omega * area)

    # get wind info
    u10 = ds['u10'].sel(source=wind_source).mean().item()
    v10 = ds['v10'].sel(source=wind_source).mean().item()
    wspd = np.sqrt(u10**2 + v10**2)
    wdir = (270-np.rad2deg(np.arctan2(v10, u10))) % 360

    # effective wind speed
    u_eff = alpha1 * np.log(wspd) + alpha2 + alpha3 * wspd

    # calculate the emission rate (kg/s)
    Q = (u_eff / l_eff * IME)

    # ---- uncertainty ----
    # 1. random
    if land_only:
        from roaring_landmask import RoaringLandmask
        roaring = RoaringLandmask.new()
        landmask = roaring.contains_many(ds['longitude'].stack(z=['y', 'x']).astype('float64').values,
                                         ds['latitude'].stack(z=['y', 'x']).astype('float64').values).reshape(ds['longitude'].shape)
        # save to DataArray
        segmentation = xr.DataArray(landmask, dims=['y', 'x'])
        ds_l2b['ch4'] = ds_l2b['ch4'].where(segmentation)
        ds['ch4'] = ds['ch4'].where(segmentation)

    IME_std = calc_random_err(ds_l2b['ch4'], ds['ch4'], area, sp)
    err_random = u_eff / l_eff * IME_std

    # 2. wind error
    err_wind = calc_wind_error(wspd, IME, l_eff, alpha1, alpha2, alpha3)

    # 3. alpha2 (shape)
    err_shape = (IME / l_eff) * (alpha2 * (0.66-0.42)/0.42)

    Q_err = np.sqrt(err_random**2 + err_wind**2 + err_shape**2)

    return wspd, wdir, l_eff, u_eff, IME, Q*3600, Q_err*3600, \
        err_random*3600, err_wind*3600, err_shape*3600  # kg/h


def reprocess_data(filename, reprocess_nc):
    df = pd.read_csv(filename, dtype={'wind_weights': bool})
    plume_filename = filename.replace('.csv', '.nc')
    l2b_filename = ('_'.join(plume_filename.split('_')[:-1])+'.nc').replace('L3', 'L2')
    ds_plume = xr.open_dataset(plume_filename)
    ds_l2b = xr.open_dataset(l2b_filename, decode_coords='all')



    if reprocess_nc:
        ds_plume.close()
        # read plume source location
        longitude = df['plume_longitude'].item()
        latitude = df['plume_latitude'].item()

        # read plume mask settings
        plume_varname = 'ch4_comb_denoise'
        land_only = True
        only_plume = True
        plume_num = re.search('plume(.*).nc', os.path.basename(plume_filename)).group(1)
        pick_plume_name = 'plume' + plume_num
        html_filename = os.path.join(os.path.dirname(plume_filename),
                                     os.path.basename(plume_filename).replace(f'_{pick_plume_name}.nc', '.html')
                                     )
        wind_source = df['wind_source'].item()
        wind_weights = df['wind_weights'].item()
        niter = df['niter'].item()
        size_median = df['size_median'].item()
        sigma_guass = df['sigma_guass'].item()
        quantile = df['quantile'].item()

        # create new mask
        mask, lon_mask, lat_mask, plume_html_filename = mask_data(html_filename, ds_l2b, longitude, latitude,
                                                                  pick_plume_name, plume_varname,
                                                                  wind_source, wind_weights, land_only,
                                                                  niter, size_median, sigma_guass, quantile,
                                                                  only_plume)

        # mask data
        ch4_mask = ds_l2b['ch4'].where(mask)

        # calculate mean wind and surface pressure in the plume
        u10 = ds_l2b['u10'].where(mask).mean(dim=['y', 'x'])
        v10 = ds_l2b['v10'].where(mask).mean(dim=['y', 'x'])
        sp = ds_l2b['sp'].where(mask).mean(dim=['y', 'x'])

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
                        }
        ds_merge.attrs = header_attrs

        LOG.info(f'Updating {plume_filename} ...')
        ds_merge.to_netcdf(plume_filename)

        wspd, wdir, l_eff, u_eff, IME, Q, Q_err, \
            err_random, err_wind, err_shape = calc_emiss(df, ds_merge, ds_l2b)
    else:
        wspd, wdir, l_eff, u_eff, IME, Q, Q_err, \
            err_random, err_wind, err_shape = calc_emiss(df, ds_plume, ds_l2b)

    # update emission data
    LOG.info(f'Updating {filename} ...')
    df['wind_speed'] = wspd
    df['wind_direction'] = wdir
    df['leff_ime'] = l_eff
    df['ueff_ime'] = u_eff
    df['ime'] = IME
    df['emission'] = Q
    df['emission_uncertainty'] = Q_err
    df['emission_uncertainty_random'] = err_random
    df['emission_uncertainty_wind'] = err_wind
    df['emission_uncertainty_shape'] = err_shape

    df.to_csv(filename, index=False)


def main():
    # whether regenerate the L3 plume NetCDF file based on csv settings
    reprocess_nc = False

    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    if len(filelist) == 0:
        return

    for filename in filelist:
        LOG.info(f'Reprocessing {filename} ...')
        reprocess_data(filename, reprocess_nc)


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
