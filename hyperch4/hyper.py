#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Hyper object to hold hyperspectral satellite data."""
import logging
from datetime import timedelta

import numpy as np
import xarray as xr
from pyorbital import orbital
from satpy import DataQuery, Scene

from .hsi2rgb import Hsi2rgb
from .denoise import Denoise
from .orthorectification import Ortho
from .retrieve import MatchedFilter
from .tle import TLE
from .wind import Wind

AVAILABLE_READERS = ['hsi_l1b', 'emit_l1b', 'hyc_l1']
SPECIES_NAME = {'ch4': 'methane', 'co2': 'carbon dioxide'}
LOG = logging.getLogger(__name__)


class Hyper():
    """The Hyper Class.
    Example usage::
        from hyperch4 import Hyper

        # create Hyper and open files
        hyp = Hyper(filenames='/path/to/file/*')

        # load datasets from input files
        hyp.load()

        # retrieve ch4
        hyp.retrieve(self, wvl_intervals=[2110, 2450])

        # orthorectification (EnMAP and EMIT)
        hyp.terrain_corr(varname='rgb')

        # export to NetCDF file
        hyp.scene.save_datasets(datasets=['u10', 'v10', 'rgb', 'ch4'], filename='output.nc', writer='cf')
    """

    def __init__(self, filename=None, reader=None, reader_kwargs=None,
                 destrip=True):
        """Initialize Hyper.

        To load Hyper data, `filename` and `reader` must be specified::

            hyp = Hyper(filenames=glob('/path/to/hyper/files/*'), reader='hsi_l1b')

        Args:
            filename (list): The files to be loaded.
            reader (str): The name of the reader to use for loading the data.
            reader_kwargs (dict): Keyword arguments to pass to specific reader instances.
        """
        self.filename = filename
        self.reader = reader
        self.destrip = destrip

        self.available_dataset_names = self._get_dataset_names()

    def _get_dataset_names(self):
        """Get the necessary dataset_names for retrieval."""
        if self.reader == 'hsi_l1b':
            # EnMAP L1B
            swir_rad_id = DataQuery(name='swir', calibration='radiance')
            vnir_rad_id = DataQuery(name='vnir', calibration='radiance')
            dataset_names = [swir_rad_id, vnir_rad_id, 'deadpixelmap',
                             'rpc_coef_vnir', 'rpc_coef_swir']
        elif self.reader == 'emit_l1b':
            # EMIT L1B
            dataset_names = ['glt_x', 'glt_y', 'radiance', 'sza', 'vza']
        elif self.reader == 'hyc_l1':
            swir_rad_id = DataQuery(name='swir', calibration='radiance')
            vnir_rad_id = DataQuery(name='vnir', calibration='radiance')
            dataset_names = [swir_rad_id, vnir_rad_id]
        else:
            raise ValueError(f"'reader' must be a list of available readers: {AVAILABLE_READERS}")

        return dataset_names

    def _rgb_composite(self):
        """Create RGB composite"""
        try:
            # --- HSI2RGB method ---
            LOG.debug('Use HSI2RGB method for RGB image')
            # slice data to VIS range
            da_vis = self.scene['radiance'].sel(bands=slice(380, 750))
            data = da_vis.stack(z=['y', 'x']).transpose(..., 'bands')
            # generate RGB img
            rgb = Hsi2rgb(data.coords['bands'], data.data,
                          da_vis.sizes['y'], da_vis.sizes['x'],
                          65, 1e-3)

            # save to DataArray with correct dim order
            rgb = xr.DataArray(rgb, dims=['y', 'x', 'bands'], coords={'bands': np.array([650, 560, 470])})
            rgb = rgb.transpose('bands', ...)

            # copy attrs
            rgb.attrs = self.scene['radiance'].attrs
            rgb.attrs['units'] = '1'
            rgb.attrs['standard_name'] = 'true_color'

            rgb = rgb.rename('rgb')
            self.scene['rgb'] = rgb
        except:
            # --- nearest method ---
            LOG.debug('Use Nearest method for RGB image')
            def gamma_norm(band):
                """Apply gamma_norm to create RGB composite"""
                gamma = 50
                band_gamma = np.power(band, 1/gamma)
                band_min, band_max = (np.nanmin(band_gamma), np.nanmax(band_gamma))

                return ((band_gamma-band_min)/((band_max - band_min)))

            rgb = self.scene['radiance'].sel(bands=[650, 560, 470], method='nearest')

            if rgb.chunks is not None:
                rgb.load()

            rgb = xr.apply_ufunc(gamma_norm,
                                 rgb.transpose(..., 'bands'),
                                 exclude_dims=set(('y', 'x')),
                                 input_core_dims=[['y', 'x']],
                                 output_core_dims=[['y', 'x']],
                                 vectorize=True)

            # copy attrs
            rgb.attrs = self.scene['radiance'].attrs
            rgb.attrs['units'] = '1'
            rgb.attrs['standard_name'] = 'true_color'

            # remove useless attrs
            if 'calibration' in rgb.attrs:
                del rgb.attrs['calibration']

            rgb = rgb.rename('rgb')
            self.scene['rgb'] = rgb

    def _calc_sensor_angle(self):
        """Calculate the VAA and VZA from TLE file"""
        delta_day = timedelta(days=1)

        # get the TLE info
        tles = TLE(self.platform_name).get_tle(self.start_time-delta_day, self.start_time+delta_day)
        while len(tles) == 0:
            # in case tle file is not available for three days
            delta_day += timedelta(days=1)
            tles = TLE(self.platform_name).get_tle(self.start_time-delta_day, self.start_time+delta_day)

        # pass tle to Orbital
        orbit = orbital.Orbital(self.platform_name, line1=tles[0], line2=tles[1])

        # calculate the lon and lat center
        lons, lats = self.area.get_lonlats()
        lon_centre = lons.mean()
        lat_centre = lats.mean()

        # get sensor angle and elevation
        vaa, satel = orbit.get_observer_look(self.start_time,
                                             lon=lon_centre,
                                             lat=lat_centre,
                                             alt=0)

        # get VZA
        vza = 90 - satel

        return vaa, vza

    def load(self, drop_waterbands=True):
        """Load data into xarray Dataset using satpy

        Args:
            drop_waterbands (boolean): whether to drop bands affected by water. Default: True.
        """
        scn = Scene(self.filename, reader=self.reader)
        scn.load(self.available_dataset_names)

        # merge band dims into one "bands" dims if they are splited (e.g. EnMAP and PRISMA)
        if 'radiance' not in self.available_dataset_names:
            # Note that although we concat these DataArrays
            #   there are offsets between EnMAP VNIR and SWIR data
            scn['radiance'] = xr.concat([scn['vnir'].rename({'bands_vnir': 'bands', 'fwhm_vnir': 'fwhm'}),
                                         scn['swir'].rename({'bands_swir': 'bands', 'fwhm_swir': 'fwhm'})
                                         ],
                                        'bands')
        else:
            scn['radiance'] = scn['radiance']

        # drop duplicated bands and sort it
        #   this is the case for PRISMA
        scn['radiance'] = scn['radiance'].drop_duplicates(dim='bands')
        scn['radiance'] = scn['radiance'].sortby('bands')

        # get attrs
        self.start_time = scn['radiance'].attrs['start_time']
        self.platform_name = scn['radiance'].attrs['platform_name']
        self.area = scn['radiance'].attrs['area']

        # make sure the mean "sza" and "vza" are set as attrs of `scn['radiance']`
        #   we need these for radianceCalc later
        loaded_names = [x['name'] for x in scn._datasets.keys()]
        if 'sza' not in scn['radiance'].attrs:
            scn['radiance'].attrs['sza'] = scn['sza'].mean().load().item()

        if 'vza' not in scn['radiance'].attrs:
            if 'vza' in loaded_names:
                scn['radiance'].attrs['vza'] = scn['vza'].mean().load().item()
            else:
                # vza is not saved in the PRISMA L1 product
                _, vza = self._calc_sensor_angle()
                scn['radiance'].attrs['vza'] = vza

        if drop_waterbands:
            # drop water vapor bands
            bands = scn['radiance']['bands']
            water_mask = ((1358 < bands) & (bands < 1453)) | ((1814 < bands) & (bands < 1961))
            scn['radiance'] = scn['radiance'].where(~water_mask, drop=True)

        # load wind data
        try:
            wind = Wind(scn)
            scn['u10'] = wind.u10
            scn['v10'] = wind.v10
            scn['sp'] = wind.sp
        except Exception as e:
            LOG.warning(e)
            LOG.warning("It seems we can't find any wind data for the date. Please check.")

        # save into scene and generate RGB composite
        self.scene = scn
        self._rgb_composite()

    def retrieve(self, wvl_intervals=None, species='ch4',
                 algo='smf', fit_unit='lognormal',
                 mode='column', land_mask=True):
        """Retrieve methane enhancements

        Args:
            wvl_intervals (list): The wavelength range [nm] used in matched filter. It can be one list or nested list.
                e.g. [2110, 2450] or [[1600, 1750], [2110, 2450]]
                Deafult: [2110, 2450].
            algo (str): The matched filter algorithm, currently supporting these algorithms below:
                1. smf: simple matched filter
                        This is the original matched filter algorithm.
                2. ctmf: cluster-tuned matched filter
                        This algorithm clusters similar pixels to improve the mean and cov calculation.
                Default: 'smf'
            fit_unit (str): The fitting method ('lognormal', 'poly', or 'linear') to calculate the unit CH4 spectrum
                            Default: 'lognormal'
            mode (str): the mode ("column" or "scene") to apply matched filter.
                        Default: 'column'.
        """
        # set default wvl_interval and units
        if species == 'ch4':
            units = 'ppb'
            unit_scale = 1
            if wvl_intervals is None:
                wvl_intervals = [2110, 2450]
        elif species == 'co2':
            units = 'ppm'
            unit_scale = 1e-3  # output unit is ppb, scale it to ppm
            if wvl_intervals is None:
                wvl_intervals = [1930, 2200]
        else:
            raise ValueError(f"Please input a correct species name (ch4 or co2). {species} is not supported.")

        enhancement = getattr(MatchedFilter(self.scene['radiance'],
                              wvl_intervals, species, fit_unit, mode, land_mask), algo)()
        if enhancement.chunks is not None:
            # load the data
            enhancement.load()

        # copy attrs and add units
        enhancement.attrs = self.scene['radiance'].attrs
        enhancement.attrs['standard_name'] = f'{SPECIES_NAME[species]}_enhancement'
        enhancement.attrs[
            'description'] = f'{SPECIES_NAME[species]} enhancement derived by the {wvl_intervals[0]}~{wvl_intervals[1]} nm window'

        # set units
        enhancement *= unit_scale
        enhancement.attrs['units'] = units

        # remove useless attrs
        if 'calibration' in enhancement.attrs:
            del enhancement.attrs['calibration']

        self.scene[species] = enhancement.rename(species)

    def terrain_corr(self, varname='rgb', rpcs=None):
        """Apply orthorectification

        Args:
            varname (str): the variable to be orthorectified
            rpcs: the Ground Control Points (gcps) or Rational Polynomial Coefficients (rpcs)
                If `rpcs` is None, we look for glt_x/glt_y data automatically.
        """

        da_ortho = Ortho(self.scene, varname, rpcs=rpcs).apply_ortho()

        return da_ortho

    def denoise(self, varname='ch4', method='tv_filter'):
        """Denoise the random noise"""
        da_denoise = Denoise(self.scene, varname, method=method).smooth()

        return da_denoise
