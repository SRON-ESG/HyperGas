#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Hyper object to hold hyperspectral satellite data."""
import logging
from datetime import timedelta

import os
import yaml
import numpy as np
import xarray as xr
from pyorbital import orbital
from satpy import DataQuery, Scene

from .denoise import Denoise
from .hsi2rgb import Hsi2rgb
from .orthorectification import Ortho
from .retrieve import MatchedFilter
from .tle import TLE
from .wind import Wind
from .a_priori_mask import Mask

AVAILABLE_READERS = ['hsi_l1b', 'emit_l1b', 'hyc_l1']
LOG = logging.getLogger(__name__)


class Hyper():
    """The Hyper Class.

    Example usage::

        from hypergas import Hyper

        # create Hyper and open files
        hyp = Hyper(filenames='/path/to/file/*')

        # load datasets from input files
        hyp.load()

        # retrieve ch4
        hyp.retrieve(self, wvl_intervals=[2110, 2450], species='ch4')

        # orthorectification
        hyp.terrain_corr(varname='rgb')

        # export to NetCDF file
        hyp.scene.save_datasets(datasets=['u10', 'v10', 'rgb', 'ch4'], filename='output.nc', writer='cf')
    """

    def __init__(self, filename=None, reader=None):
        """Initialize Hyper.

        To load Hyper data, ``filename`` and ``reader`` must be specified::

            hyp = Hyper(filenames=glob('/path/to/hyper/files/*'), reader='hsi_l1b')

        Parameters
        ----------
        filename : list
            The files to be loaded.
        reader : str
            The name of the reader to use for loading the data.
        """
        self.filename = filename
        self.reader = reader

        self.available_dataset_names = self._get_dataset_names()

        # load settings
        _dirname = os.path.dirname(__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)

        self.species_setting = settings['species']

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
            # uncomment the line if you wanna use 2D CW
            #   note that 2D CW sometimes lead to wrong retrieval and you need to correct the smile effect
            # dataset_names = [swir_rad_id, vnir_rad_id, 'cw_vnir', 'cw_swir']
        else:
            raise ValueError(f"'reader' must be a list of available readers: {AVAILABLE_READERS}")

        return dataset_names

    def _rgb_composite(self):
        """Create RGB composite"""
        try:
            # --- HSI2RGB method ---
            LOG.info('Use HSI2RGB method for RGB image')

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
            rgb.attrs['long_name'] = 'true color (RGB)'

            rgb = rgb.rename('rgb')
            self.scene['rgb'] = rgb
        except:
            # --- nearest method ---
            LOG.info('HSI2RGB failed. Using Nearest method for RGB image now.')

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
                                 vectorize=True,
                                 )

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
        # get the TLE info
        delta_day = timedelta(days=1)
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

    def _copy_attrs(self, new_data, new_attrs_dict):
        """Copy the radiance attrs to new DataArray

        Parameters
        ----------
        new_data : DataArray
        new_attrs_dict : dict
            dict of new attributes
            the dict should include these keys: 'long_name', 'description', 'units'
        """
        # copy attrs and add units
        new_data.attrs = self.scene['radiance'].attrs
        new_data = new_data.assign_attrs(new_attrs_dict)

        # remove useless attrs
        if 'calibration' in new_data.attrs:
            del new_data.attrs['calibration']

        return new_data

    def _scale_units(self, units):
        """Scale units from ppm"""
        if units == 'ppm':
            scale = 1
        elif units == 'ppm m':
            scale = 1/1.25e-4
        elif units == 'ppb':
            scale = 1e3
        elif units == 'umol m-2':
            scale = 1000/2900*1e6  # ppm -> umol m-2
        else:
            raise ValueError(f'We do not support converting ppm to {units}.')

        return scale

    def load(self, drop_waterbands=True):
        """Load data into xarray Dataset using Satpy.

        Parameters
        ----------
        drop_waterbands : bool
            whether to drop bands affected by water. Default: True.
        """
        # load available datasets
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

        # merge 2D VNIR and SWIR central wavelength into one DataArray (PRISMA)
        #   output dims: (bands, x)
        if all(x in self.available_dataset_names for x in ['cw_vnir', 'cw_swir']):
            scn['central_wavelengths'] = xr.concat([scn['cw_vnir'].rename({'bands_vnir': 'bands', 'fwhm_vnir': 'fwhm'}),
                                                    scn['cw_swir'].rename({'bands_swir': 'bands', 'fwhm_swir': 'fwhm'})
                                                    ],
                                                   'bands')
            scn['central_wavelengths'] = scn['central_wavelengths'].drop_duplicates(dim='bands')
            # sort bands and remove zero values
            scn['central_wavelengths'] = scn['central_wavelengths'].sortby('bands').sel(bands=slice(1e-7, None))

        # drop duplicated bands and sort it
        #   this is the case for PRISMA
        scn['radiance'] = scn['radiance'].drop_duplicates(dim='bands')
        scn['radiance'] = scn['radiance'].sortby('bands').sel(bands=slice(1e-7, None))

        # get attrs
        self.start_time = scn['radiance'].attrs['start_time']
        self.platform_name = scn['radiance'].attrs['platform_name']
        self.area = scn['radiance'].attrs['area']

        # make sure the mean "sza" and "vza" are set as attrs of `scn['radiance']`
        #   we need these for radianceCalc later
        loaded_names = [x['name'] for x in scn.keys()]
        if 'sza' not in scn['radiance'].attrs:
            scn['radiance'].attrs['sza'] = scn['sza'].mean().load().item()

        if 'vza' not in scn['radiance'].attrs:
            if 'vza' in loaded_names:
                scn['radiance'].attrs['vza'] = scn['vza'].mean().load().item()
            else:
                # vza is not saved in the PRISMA L1 product
                vaa, vza = self._calc_sensor_angle()
                scn['radiance'].attrs['vza'] = vza
                scn['radiance'].attrs['vaa'] = vaa

        if drop_waterbands:
            # drop water vapor bands
            bands = scn['radiance']['bands']
            water_mask = ((1358 < bands) & (bands < 1453)) | ((1814 < bands) & (bands < 1961))
            scn['radiance'] = scn['radiance'].where(~water_mask, drop=True)
            if 'central_wavelengths' in loaded_names:
                scn['central_wavelengths'] = scn['central_wavelengths'].where(~water_mask, drop=True)

        # load wind data and set flag for determining whether it is loaded successfully
        try:
            wind = Wind(scn)
            scn['u10'] = wind.u10
            scn['v10'] = wind.v10
            scn['sp'] = wind.sp
            self.wind = True
        except Exception as e:
            LOG.warning(e)
            LOG.warning("It seems we can't find any wind data for the date. Please check.")
            self.wind = False

        # get the radiance at 2100 nm which is useful to check albedo effects
        scn['radiance_2100'] = scn['radiance'].sel(
            bands=2100, method='nearest').rename('radiance_2100').expand_dims('bands')
        scn['radiance_2100'].attrs['long_name'] = 'TOA radiance at 2100 nm'
        scn['radiance_2100'].attrs['description'] = 'TOA radiance at 2100 nm'

        # drop useless coords of radiance_2100
        coords = list(scn['radiance_2100'].coords)
        if len(coords) > 0:
            scn['radiance_2100'] = scn['radiance_2100'].drop_vars(coords)

        # save into scene and generate RGB composite
        self.scene = scn
        LOG.info('Generating RGB')
        self._rgb_composite()

    def retrieve(self, wvl_intervals=None, species='ch4',
                 algo='smf', mode='column', rad_dist='normal',
                 land_mask=True, land_mask_source='OSM',
                 cluster=False, plume_mask=None):
        """Retrieve trace gas enhancements.

        Parameters
        ----------
        wvl_intervals: list
            The wavelength range (nm) used in matched filter. It can be one list or nested list.
            e.g. ``[2110, 2450]`` or ``[[1600, 1750], [2110, 2450]]``.
            Deafult: ``[2110, 2450]`` for ch4 and ``[1930, 2200]`` for co2.
        species : str
            The species ("ch4", "co2") to be retrieved.
            Default: "ch4".
        algo : str
            The matched filter algorithm, currently supporting only one algorithm:
            simple matched filter (smf). This is the original matched filter algorithm.
        mode : str
            The mode ("column" or "scene") to apply matched filter.
            Default: "column". Be careful of noise if you apply the matched filter for the whole scene.
        rad_dist : str
            The assumed rads distribution ("normal" or "lognormal")
            Default: "normal".
        land_mask : bool
            Whether apply the matched filter to continental and oceanic pixels seperately.
            Default: True.
        land_mask_source : str
            The data source of land mask ("OSM", "GSHHS" or "Natural Earth").
            Default: OSM.
        cluster : bool
            Whether apply the pixel classification.
            Default: False.
        plume_mask : :class:`numpy.ndarray`
            2D manual mask. 0: neglected pixels, 1: valid pixels.
            Default: ``None``.
        """
        rad_source = self.species_setting[species]['rad_source']
        if wvl_intervals is None:
            wvl_intervals = self.species_setting[species]['wavelength']

        if rad_source not in ['model', 'lut']:
            raise ValueError(f"The rad_source in the config.yaml file should be 'model' or 'lut'. {rad_source} is not supported.")

        units = self.species_setting[species]['units']
        unit_scale = self._scale_units(units)

        if (rad_source == 'lut') and (species not in ['ch4', 'co2']):
            raise ValueError(f"Please input a correct species name (ch4 or co2). {species} is not supported by LUT.")

        mf = MatchedFilter(self.scene, wvl_intervals, species, mode, rad_dist, rad_source, land_mask, land_mask_source, cluster, plume_mask)
        segmentation = mf.segmentation
        enhancement = getattr(mf, algo)()

        # load the retrieval results
        if enhancement.chunks is not None:
            enhancement.load()

        # set units
        enhancement *= unit_scale

        # copy attrs
        segmentation_attrs = {'standard_name': 'segmentation',
                              'long_name': 'pixel segmentation',
                              'units': 1,
                              'description': segmentation.attrs['description'],
                              }

        enhancement_attrs = {'standard_name': f"{self.species_setting[species]['name']}_enhancement",
                             'long_name': f"{self.species_setting[species]['name']}_enhancement",
                             'units': units,
                             'description': f"{self.species_setting[species]['name']} enhancement derived by the {wvl_intervals[0]}~{wvl_intervals[1]} nm window",
                             }

        enhancement = self._copy_attrs(enhancement, enhancement_attrs)
        segmentation = self._copy_attrs(segmentation, segmentation_attrs)

        # copy to scene
        self.scene['segmentation'] = segmentation.rename('segmentation')
        self.scene[species] = enhancement.rename(species)

        # add the MF info
        if 'mf' in algo:
            self.scene[species].attrs['matched_filter'] = f'{rad_dist} matched filter'

    def terrain_corr(self, varname='rgb', rpcs=None, gcps=None, gcp_crs=None):
        """Apply orthorectification using :class:`hypergas.orthorectification.Ortho`.

        Parameters
        ----------
        varname : str
            The variable to be orthorectified.
        rpcs:
            The Ground Control Points (gcps) or Rational Polynomial Coefficients (rpcs).
            If `rpcs` is None, we look for glt_x/glt_y data automatically.

        Returns
        -------
        da_ortho : :class:`~xarray.DataArray`
            The orthorectified data.
        """

        da_ortho = Ortho(self.scene, varname, rpcs=rpcs, gcps=gcps, gcp_crs=gcp_crs).apply_ortho()

        return da_ortho

    def denoise(self, varname='ch4', method='calibrated_tv_filter', weight=None):
        """Denoise the input data using :class:`hypergas.denoise.Denoise`.

        Parameters
        ----------
        varname : str
            The variable to be denoised.
        method : str
            The denoising method: "tv_filter" and "calibrated_tv_filter" (default).
        weight : int
            The weight for denoise_tv_chambolle.
            It would be neglected if method is "calibrated_tv_filter".
            If ``weight`` is ``None`` (default) and ``method`` is "tv_filter",
            the denoise_tv_chambolle will use the default value (0.1) which is too low for hyperspectral noisy gas field.

        Returns
        -------
        da_denoise : :class:`~xarray.DataArray`
            Denoised data.
        """
        da_denoise = Denoise(self.scene, varname, method=method, weight=weight).smooth()

        return da_denoise

    def plume_mask(self, varname='ch4_comb_denoise', n_min_threshold=5, sigma_threshold=1):
        """Create a priori plume masks using :class:`hypergas.a_priori_mask.Mask`.

        Parameters
        ----------
        scn : Satpy Scene
            Scene including one variable named ``segmentation`` which is calculated by :meth:`hypergas.landmask.Land_mask`.
            segmentation (:class:`~xarray.DataArray`): 0: ocean, >0: land
        varname : str
            The variable used to create plume mask. (Recommend: ``<gas>_comb_denoise``)
        n_min_threshold : int
            The minimum number of pixels per threshold for detecting features. (Default: 5).
        sigma_threshold : int
            Gaussian filter sigma for smoothing field.
            Default: 1. Because the ``<gas>_comb_denoise`` field is already smoothed, 1 should be high enough.
        """
        thresholds, features, da_plume_mask = Mask(self.scene, varname, n_min_threshold=n_min_threshold, sigma_threshold=sigma_threshold).get_feature_mask()

        return da_plume_mask
