#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Create plume masks and calculate emission rates."""

import gc
import logging
import os
from glob import glob
from itertools import chain

import geopandas as gpd
import rioxarray
import xarray as xr
from hypergas.emiss import Emiss

from utils import get_dirs

# set filename pattern to load data automatically
PATTERNS = ['ENMAP01-____L2B*.geojson', 'EMIT_L2B*.geojson', 'PRS_L2_*.geojson']

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)


def process_data(gas, filename):
    geo_df = gpd.read_file(filename)
    geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

    for index, loc in enumerate(geo_df_list):
        # loop plume by plume
        latitude = loc[0]
        longitude = loc[1]
        plume_name = f'plume{index}'

        csv_savename = filename.replace('.geojson', f'_{plume_name}.csv').replace('L2', 'L3')
        if skip_exist:
            skip = os.path.exists(csv_savename)
        else:
            skip = False

        if not skip:
            LOG.info(f'Anzlyzing {plume_name}')

            l2b_filename = filename.replace('.geojson', '.nc')
            ds_l2b = xr.open_dataset(l2b_filename, decode_coords='all')

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
            emiss.estimate(ipcc_sector, wspd_manual=wspd_manual, land_only=land_only)

            ds_l2b.close()
            del emiss, ds_l2b
            gc.collect()
        else:
            LOG.info(
                f"Skip processing {filename.replace('.geojson', '.nc')} because {csv_savename} is already existed.")


def main():
    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    if len(filelist) == 0:
        return

    for filename in filelist:
        LOG.info(f'Processing {filename} ...')
        process_data(gas, filename)


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI_landfill/'
    lowest_dirs = get_dirs(root_dir)

    # if skip already processed file
    skip_exist = True

    # gas to be processed
    gas = 'ch4'

    # ipcc sector
    #   'Electricity Generation (1A1)', 'Coal Mining (1B1a)',
    #   'Oil & Gas (1B2)', 'Livestock (4B)', 'Solid Waste (6A)', 'Other'
    ipcc_sector = 'Solid Waste (6A)'

    # wind reanalysis source: 'ERA5' or 'GEOS-FP'
    wind_source = 'ERA5'

    # only considering land pixels
    land_only = True

    # replace reanalysis wind speed (m/s)
    wspd_manual = None

    # land mask source: 'GSHHS' or 'Natural Earth'
    land_mask_source = 'GSHHS'

    # maximum of azimuth of minimum rotated rectangle
    # keep this default unless you find obvious wrong plume pixels
    azimuth_diff_max = 30

    for data_dir in lowest_dirs:
        LOG.info(f'Processing data under {data_dir}')
        main()
