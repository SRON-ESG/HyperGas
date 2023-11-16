#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Process L1 data into orthorectified CH4 L2B NetCDF file."""

import logging
import os
import warnings
from glob import glob
from itertools import chain
from pathlib import Path

from hyperch4 import Hyper

warnings.filterwarnings("ignore")


# set filename pattern to load data automatically
PATTERNS = ['ENMAP01-____L1B*.ZIP', 'EMIT_L1B_RAD*.nc', 'PRS_L1_*.zip']

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# set global attrs for exported NetCDF file
AUTHOR = 'Xin Zhang'
EMAIL = 'xin.zhang@sron.nl; xinzhang1215@gmail.com'
INSTITUTION = 'SRON Netherlands Institute for Space Research'


class L2B():
    """The L2B Class."""

    def __init__(self, filename, skip_exist=True):
        """Init class."""
        if 'ENMAP' in filename:
            reader = 'hsi_l1b'
        elif 'EMIT' in filename:
            reader = 'emit_l1b'
        elif 'PRS' in filename:
            reader = 'hyc_l1'
        else:
            raise ValueError(f"{filename} is not supported. Please check the filename pattern: {PATTERNS}")

        self.reader = reader
        self.filename = filename

        # set output name
        self.savename = str(Path(self.filename.replace('L1', 'L2').replace('_RAD', '')).with_suffix('.nc'))

        if skip_exist:
            self.skip = os.path.exists(self.savename)
        else:
            self.skip = False

        if not self.skip:
            self.hyp = self.load()
            self.denoise()
            self.ortho()
            self.to_netcdf()
        else:
            LOG.info(f'Skipped processing {self.filename}, because L2 data is already existed.')

    def load(self):
        """Load data and run retrieval."""
        if self.reader == 'emit_l1b':
            # read both RAD and OBS data
            filename = [self.filename, self.filename.replace('RAD', 'OBS')]
        else:
            filename = [self.filename]

        hyp = Hyper(filename, reader=self.reader)

        LOG.info('Loading L1 data')
        hyp.load()

        # retrieve ch4
        LOG.info('Retrieving ch4')
        hyp.retrieve(wvl_intervals=[1300, 2500])
        ch4_swir = hyp.scene['ch4']
        hyp.retrieve(wvl_intervals=[2100, 2450])
        ch4 = hyp.scene['ch4']

        # calculate ch4_comb
        diff = ch4_swir - ch4
        scale = ch4.std()/ch4_swir.std()
        # scale if ch4_swir < ch4
        hyp.scene['ch4_comb'] = ch4.where(diff > 0, ch4_swir*scale).rename('ch4_comb')

        return hyp

    def _ortho_enmap(self):
        rgb_corr = self.hyp.terrain_corr(varname='rgb', rpcs=self.hyp.scene['rpc_coef_vnir'].sel(
            bands_vnir=650, method='nearest').item())
        ch4_corr = self.hyp.terrain_corr(varname='ch4', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=2300, method='nearest').item())
        ch4_comb_corr = self.hyp.terrain_corr(
            varname='ch4_comb', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=2300, method='nearest').item())
        ch4_denoise_corr = self.hyp.terrain_corr(
            varname='ch4_denoise', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=2300, method='nearest').item())
        ch4_comb_denoise_corr = self.hyp.terrain_corr(
            varname='ch4_comb_denoise', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=2300, method='nearest').item())

        rgb = rgb_corr.interp_like(ch4_corr)
        rgb.attrs['area'] = ch4_corr.attrs['area']

        self.hyp.scene['rgb'] = rgb
        self.hyp.scene['ch4'] = ch4_corr
        self.hyp.scene['ch4_comb'] = ch4_comb_corr
        self.hyp.scene['ch4_denoise'] = ch4_denoise_corr
        self.hyp.scene['ch4_comb_denoise'] = ch4_comb_denoise_corr

    def _ortho_emit(self):
        # orthorectification
        rgb_corr = self.hyp.terrain_corr(varname='rgb')
        ch4_corr = self.hyp.terrain_corr(varname='ch4')
        ch4_comb_corr = self.hyp.terrain_corr(varname='ch4_comb')
        ch4_denoise_corr = self.hyp.terrain_corr(varname='ch4_denoise')
        ch4_comb_denoise_corr = self.hyp.terrain_corr(varname='ch4_comb_denoise')

        # save into Scene
        self.hyp.scene['rgb'] = rgb_corr
        self.hyp.scene['ch4'] = ch4_corr
        self.hyp.scene['ch4_comb'] = ch4_comb_corr
        self.hyp.scene['ch4_denoise'] = ch4_denoise_corr
        self.hyp.scene['ch4_comb_denoise'] = ch4_comb_denoise_corr

    def ortho(self):
        """Apply orthorectification."""
        LOG.info('Applying orthorectification')
        if self.reader == 'hsi_l1b':
            self._ortho_enmap()
        elif self.reader == 'emit_l1b':
            self._ortho_emit()
        elif self.reader == 'hyc_l1':
            LOG.info('We do not support applying orthorectification to PRISMA L1 data yet.')

    def denoise(self):
        """Denoise random noise"""
        LOG.info('Denoising ch4 data')
        self.hyp.scene['ch4_denoise'] = self.hyp.denoise(varname='ch4')
        self.hyp.scene['ch4_comb_denoise'] = self.hyp.denoise(varname='ch4_comb')

    def to_netcdf(self):
        """Save scene to netcdf file."""
        LOG.info(f'Exporting to {self.savename}')

        # set global attributes
        header_attrs = {'author': AUTHOR,
                        'email': EMAIL,
                        'institution': INSTITUTION,
                        # 'description': 'Orthorectified L2B data',
                        }

        # set compression and saved variables
        vnames = ['u10', 'v10', 'sp', 'rgb', 'ch4', 'ch4_comb', 'ch4_denoise', 'ch4_comb_denoise']

        # remove the bands dim for ch4
        self.hyp.scene['ch4'] = self.hyp.scene['ch4'].squeeze()
        self.hyp.scene['ch4_comb'] = self.hyp.scene['ch4_comb'].squeeze()
        self.hyp.scene['ch4_denoise'] = self.hyp.scene['ch4_denoise'].squeeze()
        self.hyp.scene['ch4_comb_denoise'] = self.hyp.scene['ch4_comb_denoise'].squeeze()

        # export to NetCDF file
        try:
            self.hyp.scene.save_datasets(datasets=vnames, filename=self.savename,
                                         header_attrs=header_attrs, writer='cf')
        except:
            # wind data is not available
            vnames = ['rgb', 'ch4', 'ch4_comb', 'ch4_denoise', 'ch4_comb_denoise']
            self.hyp.scene.save_datasets(datasets=vnames, filename=self.savename,
                                         header_attrs=header_attrs, writer='cf')

        # # --- backup --- #
        # # xarray version with compression
        # comp = dict(zlib=True, complevel=7)
        # ds = self.hyp.scene.to_xarray(datasets=vnames, header_attrs=header_attrs)
        # encoding = {var: comp for var in ds.data_vars}

        # # export data
        # ds.to_netcdf(path=savename, engine='netcdf4', encoding=encoding)


def main():
    filelist = list(chain(*[glob(os.path.join(data_dir, '**', pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    for filename in filelist:
        LOG.info(f'Processing {filename}')
        l2b_scene = L2B(filename, skip_exist)

        del l2b_scene


if __name__ == '__main__':
    skip_exist = True
    data_dir = '/data/xinz/Hyper_TROPOMI/'
    main()
