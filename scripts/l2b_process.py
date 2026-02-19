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
import xarray as xr
import pandas as pd
from rasterio.crs import CRS

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


class L2B():
    """The L2B Class."""

    def __init__(self, filename, species, skip_exist=True, plume_mask=None, drop_waterbands=True):
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
            self.load(drop_waterbands=drop_waterbands)
        else:
            LOG.info(f'Skipped processing {self.filename}, because L2 data is already existed.')

    def load(self, drop_waterbands=True):
        """Load L1 data."""
        if self.reader == 'emit_l1b':
            # read both RAD and OBS data
            filename = [self.filename, self.filename.replace('RAD', 'OBS')]
        else:
            filename = [self.filename]

        hyp = Hyper(filename, reader=self.reader)

        LOG.info('Loading L1 data')
        hyp.load(drop_waterbands=drop_waterbands)

        self.hyp = hyp

    def retrieve(self, land_mask=True, plume_mask=None, land_mask_source='OSM', cluster=False, skip_water=True, skip_cloud=True, rad_dist='normal'):
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
                              skip_water=skip_water,
                              skip_cloud=skip_cloud,
                              rad_dist=rad_dist,
                              species=species,
                              )
            gas_fullwvl = self.hyp.scene[species]

            # retrieval only using the strong window
            self.hyp.retrieve(land_mask=land_mask,
                              plume_mask=plume_mask,
                              land_mask_source=land_mask_source,
                              cluster=cluster,
                              skip_water=skip_water,
                              skip_cloud=skip_cloud,
                              rad_dist=rad_dist,
                              species=species,
                              )
            gas = self.hyp.scene[species]

            # update data if gas_fullwvl < gas
            diff = gas_fullwvl - gas
            self.hyp.scene[f'{species}_comb'] = gas.copy()

            # calculat the scale factor by segmentation
            with xr.set_options(keep_attrs=True):
                for label in np.unique(self.hyp.scene['segmentation']):
                    # calculate gas_comb by segmentation
                    segmentation_mask = self.hyp.scene['segmentation'] == label
                    scale = gas.where(segmentation_mask).std()/gas_fullwvl.where(segmentation_mask).std()

                    # scale if gas_fullwvl < gas
                    self.hyp.scene[f'{species}_comb'] = xr.where(segmentation_mask, gas.where(
                        diff > 0, gas_fullwvl*scale), self.hyp.scene[f'{species}_comb'])

            # update attrs
            self.hyp.scene[f'{species}_comb'] = self.hyp.scene[f'{species}_comb'].rename(f'{species}_comb')
            self.hyp.scene[f'{species}_comb'].attrs['description'] = gas_fullwvl.attrs['description']
            self.hyp.scene[f'{species}_comb'] = self.hyp.scene[f'{species}_comb'].transpose(..., 'y', 'x')

    def _update_scene(self):
        # update Scene values
        self.hyp.scene['rgb'] = self.rgb_corr
        self.hyp.scene['quality_mask'] = self.quality_mask_corr
        self.hyp.scene['segmentation'] = self.segmentation_corr
        self.hyp.scene['radiance_2100'] = self.radiance_2100_corr

        self.hyp.scene['sza'] = self.sza_corr
        self.hyp.scene['saa'] = self.saa_corr
        self.hyp.scene['vza'] = self.vza_corr
        self.hyp.scene['vaa'] = self.vaa_corr
        self.hyp.scene['raa'] = self.raa_corr
        self.hyp.scene['sga'] = self.sga_corr

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
        self.quality_mask_corr = self.hyp.terrain_corr(varname='quality_mask', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())
        self.segmentation_corr = self.hyp.terrain_corr(varname='segmentation', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())
        self.radiance_2100_corr = self.hyp.terrain_corr(varname='radiance_2100', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())

        # angles
        self.sza_corr = self.hyp.terrain_corr(varname='sza', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())
        self.saa_corr = self.hyp.terrain_corr(varname='saa', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())
        self.vza_corr = self.hyp.terrain_corr(varname='vza', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())
        self.vaa_corr = self.hyp.terrain_corr(varname='vaa', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())
        self.raa_corr = self.hyp.terrain_corr(varname='raa', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
            bands_swir=mean_wvl, method='nearest').item())
        self.sga_corr = self.hyp.terrain_corr(varname='sga', rpcs=self.hyp.scene['rpc_coef_swir'].sel(
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

    def _ortho_emit_prisma(self, rpcs=None):
        # read gcps if available
        gcps, gcp_crs = self._read_gcps()

        # orthorectification
        self.rgb_corr = self.hyp.terrain_corr(varname='rgb', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
        self.quality_mask_corr = self.hyp.terrain_corr(varname='quality_mask', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
        self.segmentation_corr = self.hyp.terrain_corr(varname='segmentation', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
        self.sza_corr = self.hyp.terrain_corr(varname='sza', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
        self.saa_corr = self.hyp.terrain_corr(varname='saa', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
        self.vza_corr = self.hyp.terrain_corr(varname='vza', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
        self.vaa_corr = self.hyp.terrain_corr(varname='vaa', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
        self.raa_corr = self.hyp.terrain_corr(varname='raa', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
        self.sga_corr = self.hyp.terrain_corr(varname='sga', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)

        if self.hyp.wind:
            self.u10_corr = self.hyp.terrain_corr(varname='u10', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
            self.v10_corr = self.hyp.terrain_corr(varname='v10', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)
            self.sp_corr = self.hyp.terrain_corr(varname='sp', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)

        self.radiance_2100_corr = self.hyp.terrain_corr(varname='radiance_2100', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs)

        for species in self.species:
            setattr(self, f'{species}_corr', self.hyp.terrain_corr(
                varname=species, rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs))
            setattr(self, f'{species}_comb_corr', self.hyp.terrain_corr(
                varname=f'{species}_comb', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs))
            setattr(self, f'{species}_denoise_corr', self.hyp.terrain_corr(
                varname=f'{species}_denoise', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs))
            setattr(self, f'{species}_comb_denoise_corr', self.hyp.terrain_corr(
                varname=f'{species}_comb_denoise', rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs))

        self._update_scene()

    def _read_gcps(self):
        gcp_file = self.savename.replace('.nc', '.points')
        if os.path.isfile(gcp_file):
            with open(gcp_file, 'r') as f:
                first_line = f.readline()
                wkt = first_line[len('#CRS:'):].strip()
                gcp_crs = CRS.from_wkt(wkt).to_epsg()

            df_gcp = pd.read_csv(gcp_file, delimiter=',', comment='#', header=0)

            return df_gcp, gcp_crs
        else:
            return None, None

    def ortho(self):
        """Apply orthorectification."""
        LOG.info('Applying orthorectification')
        if self.reader == 'hsi_l1b':
            self._ortho_enmap()
        elif self.reader in ['emit_l1b', 'hyc_l1']:
            self._ortho_emit_prisma()

    def denoise(self):
        """Denoise random noise"""
        LOG.info(f'Denoising {self.species} data')
        for species in self.species:
            self.hyp.scene[f'{species}_denoise'] = self.hyp.denoise(varname=species)
            self.hyp.scene[f'{species}_comb_denoise'] = self.hyp.denoise(varname=f'{species}_comb')

    def plume_mask(self):
        """Create a priori plume mask"""
        LOG.info(f'Creating {self.species} plume mask')

        for species in self.species:
            self.hyp.scene[f'{species}_mask'] = self.hyp.plume_mask(varname=f'{species}_comb_denoise')

    def to_netcdf(self):
        """Save scene to netcdf file."""
        LOG.info(f'Exporting to {self.savename}')

        # set global attributes
        header_attrs = {'version': hypergas.__name__+'_'+hypergas.__version__,
                        # 'description': 'Orthorectified L2B data',
                        }

        # set saved variables
        species_vnames = []
        for species in self.species:
            species_vnames.extend([species, f'{species}_comb', f'{species}_denoise',
                                  f'{species}_comb_denoise', f'{species}_mask'])
        vnames = ['u10', 'v10', 'sp', 'rgb', 'quality_mask', 'segmentation', 'radiance_2100', 'sza', 'saa', 'vza', 'vaa','raa', 'sga']
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

    # whether export unortho L2 species data
    unortho_export = False

    # whether cluster pixels for matched filter

    # set species to be retrieved (3 formats)
    #   'all': retrieve all supported gases
    #   single gas name str, e.g. 'ch4'
    #   list of gas names, e.g. ['ch4', 'co2']
    species = 'ch4'

    # whether drop the waterband
    drop_waterbands = True

    # whether skip water pixels in the retrieval
    skip_water = True

    # whether skip cloudy pixels in the retrieval
    skip_cloud = True

    # print the settings
    LOG.info(f'Settings: {{species={species}, skip_water={skip_water}, skip_cloud={skip_cloud}}}.')

    # root directory of input data
    data_dir = '../hypergas/resources/test_data/ch4_cases/'
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
        l2b_scene = L2B(filename, species, skip_exist, drop_waterbands)

        if not l2b_scene.skip:
            # rad_dist: 'normal', 'lognormal'
            # land_mask_source: 'OSM', 'GSHHS', 'Natural Earth'
            l2b_scene.retrieve(rad_dist='normal', cluster=False, land_mask_source='OSM', skip_water=skip_water, skip_cloud=skip_cloud)

            if unortho_export:
                # output unortho data
                unortho_savename = l2b_scene.savename.replace('.nc', '_unortho.nc')
                LOG.info(f'Exporting unortho file: {unortho_savename}')
                l2b_scene.hyp.scene.save_datasets(
                    datasets=[species, f'{species}_comb', 'segmentation'], filename=unortho_savename, writer='cf')

            l2b_scene.denoise()
            l2b_scene.ortho()
            l2b_scene.plume_mask()
            l2b_scene.to_netcdf()

        del l2b_scene


if __name__ == '__main__':
    main()
