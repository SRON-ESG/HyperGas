#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Process L1 data into orthorectified CH4 L2B NetCDF file."""

import logging
import os
import warnings
from glob import glob
from itertools import chain
from pathlib import Path

from hypergas import Hyper

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

    def __init__(self, filename, skip_exist=True, plume_mask=None):
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
            self.load()
        else:
            LOG.info(f'Skipped processing {self.filename}, because L2 data is already existed.')

    def load(self):
        """Load L1 data."""
        if self.reader == 'emit_l1b':
            # read both RAD and OBS data
            filename = [self.filename, self.filename.replace('RAD', 'OBS')]
        else:
            filename = [self.filename]

        hyp = Hyper(filename, reader=self.reader)

        LOG.info('Loading L1 data')
        hyp.load()

        self.hyp = hyp


    def retrieve(self, land_mask=True, plume_mask=None, rad_dist='normal'):
        """run retrieval"""
        # retrieve ch4
        LOG.info('Retrieving ch4')
        self.hyp.retrieve(wvl_intervals=[1300, 2500],
                          land_mask=land_mask,
                          plume_mask=plume_mask,
                          rad_dist=rad_dist,
                          )
        ch4_swir = self.hyp.scene['ch4']
        self.hyp.retrieve(wvl_intervals=[2100, 2450],
                          land_mask=land_mask,
                          plume_mask=plume_mask,
                          rad_dist=rad_dist,
                          )
        ch4 = self.hyp.scene['ch4']

        # calculate ch4_comb
        diff = ch4_swir - ch4
        scale = ch4.std()/ch4_swir.std()
        # scale if ch4_swir < ch4
        self.hyp.scene['ch4_comb'] = ch4.where(diff > 0, ch4_swir*scale).rename('ch4_comb')


    def _update_scene(self):
        # update Scene values
        self.hyp.scene['rgb'] = self.rgb_corr
        self.hyp.scene['segmentation'] = self.segmentation_corr
        self.hyp.scene['radiance_2100'] = self.radiance_2100_corr
        self.hyp.scene['ch4'] = self.ch4_corr
        self.hyp.scene['ch4_comb'] = self.ch4_comb_corr
        self.hyp.scene['ch4_denoise'] = self.ch4_denoise_corr
        self.hyp.scene['ch4_comb_denoise'] = self.ch4_comb_denoise_corr

        if self.hyp.wind:
            self.hyp.scene['u10'] = self.u10_corr
            self.hyp.scene['v10'] = self.v10_corr
            self.hyp.scene['sp'] = self.sp_corr

    def _ortho_enmap(self):
        self.rgb_corr = self.hyp.terrain_corr(varname='rgb', rpcs=self.hyp.scene['rpc_coef_vnir'].sel(
            bands_vnir=650, method='nearest').item())
        self.segmentation_corr = self.hyp.terrain_corr(varname='segmentation', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=2300, method='nearest').item())
        self.radiance_2100_corr = self.hyp.terrain_corr(varname='radiance_2100', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=2300, method='nearest').item())
        self.ch4_corr = self.hyp.terrain_corr(varname='ch4', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=2300, method='nearest').item())
        self.ch4_comb_corr = self.hyp.terrain_corr(
            varname='ch4_comb', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=2300, method='nearest').item())
        self.ch4_denoise_corr = self.hyp.terrain_corr(
            varname='ch4_denoise', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=2300, method='nearest').item())
        self.ch4_comb_denoise_corr = self.hyp.terrain_corr(
            varname='ch4_comb_denoise', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=2300, method='nearest').item())

        if self.hyp.wind:
            self.u10_corr = self.hyp.terrain_corr(varname='u10', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
                bands_swir=2300, method='nearest').item())
            self.v10_corr = self.hyp.terrain_corr(varname='v10', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
                bands_swir=2300, method='nearest').item())
            self.sp_corr = self.hyp.terrain_corr(varname='sp', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
                bands_swir=2300, method='nearest').item())

        self.rgb_corr = self.rgb_corr.interp_like(self.ch4_corr)
        self.rgb_corr.attrs['area'] = self.ch4_corr.attrs['area']

        self._update_scene()

    def _ortho_emit(self):
        # orthorectification
        self.rgb_corr = self.hyp.terrain_corr(varname='rgb')
        self.segmentation_corr = self.hyp.terrain_corr(varname='segmentation')

        if self.hyp.wind:
            self.u10_corr = self.hyp.terrain_corr(varname='u10')
            self.v10_corr = self.hyp.terrain_corr(varname='v10')
            self.sp_corr = self.hyp.terrain_corr(varname='sp')

        self.radiance_2100_corr = self.hyp.terrain_corr(varname='radiance_2100')
        self.ch4_corr = self.hyp.terrain_corr(varname='ch4')
        self.ch4_comb_corr = self.hyp.terrain_corr(varname='ch4_comb')
        self.ch4_denoise_corr = self.hyp.terrain_corr(varname='ch4_denoise')
        self.ch4_comb_denoise_corr = self.hyp.terrain_corr(varname='ch4_comb_denoise')

        self._update_scene()

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

        # we need a higher weight for denoising PRISMA data
        if self.reader == 'hyc_l1':
            weight = 90
        else:
            weight = 50

        self.hyp.scene['ch4_denoise'] = self.hyp.denoise(varname='ch4', weight=weight)
        self.hyp.scene['ch4_comb_denoise'] = self.hyp.denoise(varname='ch4_comb', weight=weight)

    def to_netcdf(self):
        """Save scene to netcdf file."""
        LOG.info(f'Exporting to {self.savename}')

        # set global attributes
        header_attrs = {'author': AUTHOR,
                        'email': EMAIL,
                        'institution': INSTITUTION,
                        # 'description': 'Orthorectified L2B data',
                        }

        # set saved variables
        vnames = ['u10', 'v10', 'sp', 'rgb', 'segmentation',
                  'radiance_2100',
                  'ch4', 'ch4_comb', 'ch4_denoise', 'ch4_comb_denoise'
                  ]
        loaded_names = [x['name'] for x in self.hyp.scene.keys()]
        # drop not loaded vnames
        vnames = [vname for vname in vnames if vname in loaded_names]

        # assign filename to global attrs
        header_attrs['filename'] = self.hyp.scene[vnames[0]].attrs['filename']

        # remove the bands dim for ch4
        for vname in vnames:
            self.hyp.scene[vname] = self.hyp.scene[vname].squeeze()
            # remove the useless filename attrs which has already been saved as global attrs
            del self.hyp.scene[vname].attrs['filename']

        # export to NetCDF file
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
    # set params
    skip_exist = True
    data_dir = '/data/xinz/Hyper_TROPOMI_plume/'

    filelist = list(chain(*[glob(os.path.join(data_dir, '**', pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    # # multiprocessing
    # import functools
    # from concurrent.futures import ProcessPoolExecutor as Pool

    # with Pool(max_workers=3) as pool:
    #     # data process in parallel
    #     try:
    #         pool.map(functools.partial(L2B, skip_exist=skip_exist), filelist)
    #     except Exception as exc:
    #         LOG.info(exc)

    for filename in filelist:
        LOG.info(f'Processing {filename}')
        l2b_scene = L2B(filename, skip_exist)

        if not l2b_scene.skip:
            l2b_scene.retrieve(rad_dist='normal')
            l2b_scene.denoise()
            l2b_scene.ortho()
            l2b_scene.to_netcdf()

        del l2b_scene


if __name__ == '__main__':
    main()
