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
import yaml
import warnings
from glob import glob
import numpy as np
from itertools import chain
from pathlib import Path

import hypergas
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

    def __init__(self, filename, species, skip_exist=True, plume_mask=None):
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

        # load settings
        _dirname = os.path.dirname(hypergas.__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)
        self.species_setting = settings['species']

        if species == 'all':
            self.species = list(self.species_setting.keys())
        elif type(species) is list:
            self.species = species
        elif type(species) is str:
            self.species = [species]
        else:
            raise ValueError(f"species should be one str or list of str. {species} is not supported")

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

    def retrieve(self, land_mask=True, plume_mask=None, land_mask_source='GSHHS', cluster=False, rad_dist='normal'):
        """run retrieval"""
        # retrieve trace gas
        for species in self.species:
            LOG.info(f'Retrieving {species}')
            # set the broad retrieval window to denoise background signals
            full_wvl_interval = self.species_setting[species]['full_wavelength']
            self.hyp.retrieve(wvl_intervals=full_wvl_interval,
                              land_mask=land_mask,
                              plume_mask=plume_mask,
                              land_mask_source=land_mask_source,
                              cluster=cluster,
                              rad_dist=rad_dist,
                              species=species,
                              )
            gas_fullwvl = self.hyp.scene[species]
            self.hyp.retrieve(land_mask=land_mask,
                              plume_mask=plume_mask,
                              land_mask_source=land_mask_source,
                              cluster=cluster,
                              rad_dist=rad_dist,
                              species=species,
                              )
            gas = self.hyp.scene[species]

            # calculate gas_comb
            diff = gas_fullwvl - gas
            scale = gas.std()/gas_fullwvl.std()
            # scale if gas_fullwvl < gas
            self.hyp.scene[f'{species}_comb'] = gas.where(diff > 0, gas_fullwvl*scale).rename(f'{species}_comb')

    def _update_scene(self):
        # update Scene values
        self.hyp.scene['rgb'] = self.rgb_corr
        self.hyp.scene['segmentation'] = self.segmentation_corr
        self.hyp.scene['radiance_2100'] = self.radiance_2100_corr

        for species in self.species:
            self.hyp.scene[f'{species}'] = getattr(self, f'{species}_corr')
            self.hyp.scene[f'{species}_comb'] = getattr(self, f'{species}_comb_corr')
            self.hyp.scene[f'{species}_denoise'] = getattr(self, f'{species}_denoise_corr')
            self.hyp.scene[f'{species}_comb_denoise'] = getattr(self, f'{species}_comb_denoise_corr')

        if self.hyp.wind:
            self.hyp.scene['u10'] = self.u10_corr
            self.hyp.scene['v10'] = self.v10_corr
            self.hyp.scene['sp'] = self.sp_corr

    def _ortho_enmap(self):
        # calculate the mean wavelength for retrieving the first species
        #   to make sure all results share a same proj later
        #   this can cause some pixel offsets if the wvls of different species span a wide range
        retrieval_wavelength = self.species_setting[self.species[0]]['wavelength']
        mean_wvl = np.mean(retrieval_wavelength)

        for species in self.species:
            # use the mean wvl as the bands for correcting gas enhancement field
            setattr(self, f'{species}_corr', self.hyp.terrain_corr(
                varname=species, rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=mean_wvl, method='nearest').item())
            )
            setattr(self, f'{species}_comb_corr', self.hyp.terrain_corr(
                varname=f'{species}_comb', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=mean_wvl, method='nearest').item())
            )
            setattr(self, f'{species}_denoise_corr', self.hyp.terrain_corr(
                varname=f'{species}_denoise', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=mean_wvl, method='nearest').item())
            )
            setattr(self, f'{species}_comb_denoise_corr', self.hyp.terrain_corr(
                varname=f'{species}_comb_denoise', rpcs=self.hyp.scene['rpc_coef_swir'].sel(bands_swir=mean_wvl, method='nearest').item())
            )

        self.rgb_corr = self.hyp.terrain_corr(varname='rgb', rpcs=self.hyp.scene['rpc_coef_vnir'].sel(
            bands_vnir=650, method='nearest').item())
        self.segmentation_corr = self.hyp.terrain_corr(varname='segmentation', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())
        self.radiance_2100_corr = self.hyp.terrain_corr(varname='radiance_2100', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())

        if self.hyp.wind:
            self.u10_corr = self.hyp.terrain_corr(varname='u10', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
                bands_swir=mean_wvl, method='nearest').item())
            self.v10_corr = self.hyp.terrain_corr(varname='v10', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
                bands_swir=mean_wvl, method='nearest').item())
            self.sp_corr = self.hyp.terrain_corr(varname='sp', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
                bands_swir=mean_wvl, method='nearest').item())

        self.rgb_corr = self.rgb_corr.interp_like(getattr(self, f'{species}_corr'))
        self.rgb_corr.attrs['area'] = getattr(self, f'{species}_corr').attrs['area']

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

        for species in self.species:
            setattr(self, f'{species}_corr', self.hyp.terrain_corr(varname=species))
            setattr(self, f'{species}_comb_corr', self.hyp.terrain_corr(varname=f'{species}_comb'))
            setattr(self, f'{species}_denoise_corr', self.hyp.terrain_corr(varname=f'{species}_denoise'))
            setattr(self, f'{species}_comb_denoise_corr', self.hyp.terrain_corr(varname=f'{species}_comb_denoise'))

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
        LOG.info(f'Denoising {self.species} data')

        # we need a higher weight for denoising PRISMA data
        if self.reader == 'hyc_l1':
            weight = 150
        else:
            weight = 50

        for species in self.species:
            self.hyp.scene[f'{species}_denoise'] = self.hyp.denoise(varname=species, weight=weight)
            self.hyp.scene[f'{species}_comb_denoise'] = self.hyp.denoise(varname=f'{species}_comb', weight=weight)

    def plume_mask(self):
        """Create a priori plume mask"""
        LOG.info(f'Creating {self.species} plume mask')

        for species in self.species:
            self.hyp.scene[f'{species}_plume_mask'] = self.hyp.plume_mask(varname=f'{species}_comb')

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
        species_vnames = []
        for species in self.species:
            species_vnames.extend([species, f'{species}_comb', f'{species}_denoise', f'{species}_comb_denoise', f'{species}_plume_mask'])
        vnames = ['u10', 'v10', 'sp', 'rgb', 'segmentation', 'radiance_2100']
        vnames.extend(species_vnames)
        loaded_names = [x['name'] for x in self.hyp.scene.keys()]

        # drop not loaded vnames
        vnames = [vname for vname in vnames if vname in loaded_names]

        # assign filename to global attrs
        header_attrs['filename'] = self.hyp.scene[vnames[0]].attrs['filename']

        # remove the bands dim for gas
        for vname in vnames:
            self.hyp.scene[vname] = self.hyp.scene[vname].squeeze()
            # remove the useless filename attrs which has already been saved as global attrs
            if 'filename' in self.hyp.scene[vname].attrs:
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
    # ---- settings --- #
    # if skip already processed file
    skip_exist = True

    # whether cluster pixels for matched filter

    # set species to be retrieved (3 formats)
    #   'all': retrieve all supported gases
    #   single gas name str, e.g. 'ch4'
    #   list of gas names, e.g. ['ch4', 'co2']
    species = 'all'

    # root directory of input data
    data_dir = '/data/xinz/Hyper_TROPOMI/'
    # ---- settings --- #

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
        l2b_scene = L2B(filename, species, skip_exist)

        if not l2b_scene.skip:
            # rad_dist: 'normal', 'lognormal'
            # land_mask_source: 'GSHHS', 'Natural Earth'
            l2b_scene.retrieve(rad_dist='normal', cluster=False, land_mask_source='GSHHS')
            l2b_scene.denoise()
            l2b_scene.ortho()
            l2b_scene.plume_mask()
            l2b_scene.to_netcdf()

        del l2b_scene


if __name__ == '__main__':
    main()
