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

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# set filename pattern to load data automatically
PATTERNS = ['ENMAP01-____L3B*.csv', 'EMIT_L3B*.csv', 'PRS_L3_*.csv']

def reprocess_data(filename, wind_data):
    LOG.info('Reading csv and nc files ...')
    df = pd.read_csv(filename)
    plume_filename = filename.replace('.csv', '.nc')

    # read settings in csv file
    azimuth_diff_max = df['azimuth_diff_max'].item()
    if 'dist_max' in df.columns:
        dist_max = df['dist_max'].item()
    else:
        dist_max = 180
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

    if reprocess_nc:
        # read L2B data
        l2b_filename = ('_'.join(plume_filename.split('_')[:-1])+'.nc').replace('L3', 'L2')

        if os.path.isfile(l2b_filename):
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
                            dist_max=dist_max,
                            )

            # export to NetCDF file
            emiss.export_plume_nc()

            # calculate emission rate and export csv file
            emiss.estimate(ipcc_sector, wspd_manual=wspd_manual, land_only=land_only, name=name)

            ds_l2b.close()
            del emiss, ds_l2b
            gc.collect()
        else:
            LOG.warning(f'{l2b_filename} does not exist. Skip !!!!!')
    else:
        ds_l3 = xr.open_dataset(plume_filename, decode_coords='all')

        # read plume source location
        plume_num = re.search('plume(.*).nc', os.path.basename(plume_filename)).group(1)
        plume_name = 'plume' + plume_num

        # create Emiss class
        emiss = Emiss(ds=ds_l3, gas=gas, plume_name=plume_name)

        # calculate emission rate and export csv file
        ipcc_sector = df['ipcc_sector'].item()

        # create Emiss class
        emiss = Emiss(ds=ds_l3, gas=gas, plume_name=plume_name)

        # assign nevessary variables
        emiss.longitude = df['plume_longitude'].item()
        emiss.latitude = df['plume_latitude'].item()
        emiss.plume_nc_filename = plume_filename
        emiss.plume_name = plume_name
        emiss.gas = df['gas'].item().lower()
        emiss.wind_source = df['wind_source'].item()
        emiss.land_only = land_only
        emiss.land_mask_source = land_mask_source
        emiss.azimuth_diff_max = azimuth_diff_max
        emiss.dist_max = dist_max

        # estimates
        emiss.estimate(ipcc_sector, wspd_manual=wspd_manual, name=name)

        ds_l3.close()
        del emiss, ds_l3
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
        #   if users wanna change the alpha values, just change them in the `config.yaml` file and run this script again
        reprocess_data(filename, wind_data)


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '../hypergas/resources/test_data/ch4_cases/'
    lowest_dirs = get_dirs(root_dir)

    # which wind source data to be applied
    wind_data = 'auto'  # 'auto', 'ERA5', or 'GEOS-FP'

    # whether reprocess the NetCDF file
    #   True: update the L3 NetCDF file, regenerate the html file, and calculate emission rates
    #   False: read the existed L3 NetCDF file to calculate emission rates
    reprocess_nc = True

    # replace reanalysis wind speed (m/s)
    wspd_manual = None

    # whether skip dir which contains exported html
    for data_dir in lowest_dirs:
        LOG.info(f'Reprocessing data under {data_dir}')
        main()
