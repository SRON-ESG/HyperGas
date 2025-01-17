#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Regenerate L2B NetCDF products with L3 plume masks."""

import re
import logging
import os
from glob import glob
from pathlib import Path
from itertools import chain

import numpy as np
import pandas as pd
import xarray as xr
from pyresample import geometry, kd_tree

from l2b_process import L2B
from utils import get_dirs

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# set filename pattern to load data automatically
PATTERNS = ['ENMAP01-____L3B*plume0.nc', 'EMIT_L3B*plume0.nc', 'PRS_L3_*plume0.nc']


def reprocess_data(filename, prefix, species, land_mask_source, rad_dist):
    """Reprocess L2 by L3 plume masks"""
    # read all L3 nc data with same prefix
    l3_filelist = glob(prefix+'*nc')
    l3_filelist = [file for file in l3_filelist if re.search(r'_plume\d+\.nc$', file)]
    ds = xr.open_mfdataset(l3_filelist,
                           concat_dim=[pd.Index(np.arange(len(l3_filelist)), name='plume_id')],
                           combine='nested',
                           )
    dirname = os.path.dirname(filename)

    # create mask, 0 (plume): not included in matched filter
    mask = (~ds[species].isnull()).sum(dim='plume_id')
    mask = mask == 0
    swath_def = geometry.SwathDefinition(lons=ds['longitude'], lats=ds['latitude'])

    # read L1 data
    l2b_scene = L2B(ds.attrs['filename'], species=species, skip_exist=False)
    rads_area = l2b_scene.hyp.scene['radiance'].attrs['area']

    mask_resample = kd_tree.resample_nearest(swath_def, mask.values, rads_area, radius_of_influence=100)
    mask_resample = xr.DataArray(mask_resample, dims=['y', 'x']).rename('plume_mask')

    # run retrieval and save L2 NetCDF file
    LOG.info('Retrieving with background where plume pixels are excluded.')
    l2b_scene.savename = os.path.join(dirname, os.path.basename(str(Path(ds.attrs['filename'].replace('L1', 'L2').replace('_RAD', '')).with_suffix('.nc'))))
    l2b_scene.retrieve(plume_mask=mask_resample, land_mask_source=land_mask_source, rad_dist=rad_dist)
    l2b_scene.denoise()
    l2b_scene.ortho()
    l2b_scene.plume_mask()
    l2b_scene.to_netcdf()


def main():
    species = 'ch4'
    rad_dist = 'normal'  # 'lognormal' or 'normal'
    land_mask_source = 'OSM'  # 'OSM', 'GSHHS' or 'Natural Earth'

    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))

    # only read plume NetCDF file
    filelist = [file for file in filelist if re.search(r'_plume\d+\.nc$', file)]

    # sort list
    filelist = list(sorted(filelist))

    if len(filelist) == 0:
        return

    for file_plume0 in filelist:
        # get file with same prefix because one scene may have multiple plumes
        prefix = file_plume0.split('plume0')[0]
        filenames = glob(f"{prefix}*.nc")

        LOG.info(f"Reprocessing {prefix.replace('L3', 'L2')} ...")
        # only need to reprocess L2 once
        reprocess_data(filenames[0], prefix, species, land_mask_source, rad_dist)

if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI_landfill/'
    lowest_dirs = get_dirs(root_dir)

    for data_dir in lowest_dirs:
        LOG.info(f'Reprocessing data under {data_dir}')
        main()
