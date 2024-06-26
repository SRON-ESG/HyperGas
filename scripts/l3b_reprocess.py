#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Recalculate emission rates using existed L3 csv and NetCDF files."""

import gc
import logging
import os
import re
from glob import glob
from itertools import chain

import pandas as pd
import xarray as xr
from hypergas.emiss import Emiss

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


def reprocess_data(filename, wind_data):
    LOG.info('Reading csv and nc files ...')
    df = pd.read_csv(filename)
    plume_filename = filename.replace('.csv', '.nc')

    # read settings in csv file
    azimuth_diff_max = df['azimuth_diff_max'].item()
    ipcc_sector = df['ipcc_sector'].item()
    gas = df['gas'].item().lower()
    name = df['name'].item()
    land_only = df['land_only'].item()
    land_mask_source = df['land_mask_source'].item()

    # set wind source
    if wind_data == 'auto':
        # use same wind source
        wind_source = df['wind_source'].item()
    elif wind_data in ['ERA5', 'GEOS-FP']:
        wind_source = wind_data
    else:
        raise ValueError(f'{wind_data} is not supported. Please use "ERA5" or "GEOS-FP"')

    # read L2B data
    l2b_filename = ('_'.join(plume_filename.split('_')[:-1])+'.nc').replace('L3', 'L2')
    ds_l2b = xr.open_dataset(l2b_filename, decode_coords='all')

    # read plume source location
    longitude = df['plume_longitude'].item()
    latitude = df['plume_latitude'].item()

    plume_num = re.search('plume(.*).nc', os.path.basename(plume_filename)).group(1)
    plume_name = 'plume' + plume_num

    # create Emiss class
    emiss = Emiss(ds=ds_l2b, gas=gas, plume_name=plume_name)

    # select connected mask data
    emiss.mask_data(longitude, latitude,
                    wind_source=wind_source,
                    land_only=land_only,
                    land_mask_source=land_mask_source,
                    only_plume=True,
                    azimuth_diff_max=azimuth_diff_max,
                    )

    # export to NetCDF file
    emiss.export_plume_nc()

    # calculate emission rate and export csv file
    emiss.estimate(ipcc_sector, wspd_manual=None, land_only=land_only, name=name)

    ds_l2b.close()
    del emiss, ds_l2b
    gc.collect()


def main():
    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    if len(filelist) == 0:
        return

    for filename in filelist:
        LOG.info(f'Reprocessing {filename} ...')
        # regenerate the L3 plume NetCDF file based on csv settings and export new csv files
        #   if users wanna change the alpha values, just change them in the `emiss.py` and run this script again
        reprocess_data(filename, wind_data)


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI_landfill/'
    lowest_dirs = get_dirs(root_dir)

    # which wind source data to be applied
    wind_data = 'auto'  # 'auto', 'ERA5', or 'GEOS-FP'

    # whether skip dir which contains exported html
    for data_dir in lowest_dirs:
        LOG.info(f'Reprocessing data under {data_dir}')
        main()
