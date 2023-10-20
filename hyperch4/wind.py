#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Calculate u10 and v10 for the scene."""

import os

import numpy as np
import xarray as xr
import yaml


class Wind():
    """Calculate u10 and v10 from reanalysis wind data."""

    def __init__(self, scn):
        # load settings
        _dirname = os.path.dirname(__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)

        self.era5_dir = os.path.join(_dirname, settings['era5_dir'])
        self.geosfp_dir = os.path.join(_dirname, settings['geosfp_dir'])

        # get the obs time
        self.scene = scn
        self.obs_time = scn['radiance'].attrs['start_time']

        # calculate the center location
        self._calc_center()

        # calculate the wind
        self.load_data()

    def _calc_center(self):
        """Calculate the scene center location."""
        lons, lats = self.scene['radiance'].attrs['area'].get_lonlats()
        self.center_lon = lons.mean()
        self.center_lat = lats.mean()

    def load_data(self):
        """Load wind data."""
        # get the wind data
        u10_era5, v10_era5 = self.load_era5()
        u10_geosfp, v10_geosfp = self.load_geosfp()

        # combine them into DataArrays
        u10 = xr.DataArray(np.array([u10_era5, u10_geosfp]),
                           dims='source',
                           coords={'source': ['ERA5', 'GEOS-FP']},
                           name='u10',
                           )
        v10 = xr.DataArray(np.array([v10_era5, v10_geosfp]),
                           dims='source',
                           coords={'source': ['ERA5', 'GEOS-FP']},
                           name='v10',
                           )

        # set attrs
        u10.attrs['long_name'] = '10 metre U wind component'
        v10.attrs['long_name'] = '10 metre V wind component'
        u10.attrs['units'] = 'm s-1'
        v10.attrs['units'] = 'm s-1'

        self.u10 = u10
        self.v10 = v10

    def load_era5(self):
        """Load ERA5 wind data."""
        wind_file = os.path.join(self.era5_dir, self.obs_time.strftime('%Y/sl_%Y%m%d.grib'))

        # read the nearest ERA5 wind data
        ds_era5 = xr.open_dataset(wind_file, engine='cfgrib', indexpath='')
        ds_era5.coords['longitude'] = (ds_era5.coords['longitude'] + 180) % 360 - 180

        # interpolate data to the scene center
        u10 = ds_era5['u10'].interp(time=self.obs_time.strftime('%Y-%m-%d %H:%M'),
                                    longitude=self.center_lon,
                                    latitude=self.center_lat,
                                    method='nearest')

        v10 = ds_era5['v10'].interp(time=self.obs_time.strftime('%Y-%m-%d %H:%M'),
                                    longitude=self.center_lon,
                                    latitude=self.center_lat,
                                    method='nearest')

        return u10.item(), v10.item()

    def load_geosfp(self):
        """Load GEOS-FP wind data."""
        # read GEOS-FP by hour name
        geosfp_name = 'GEOS.fp.asm.tavg1_2d_slv_Nx.' + self.obs_time.strftime('%Y%m%d') \
            + '_' + '{:02d}{:02d}'.format(self.obs_time.hour, 30) + '.V01.nc4'
        wind_file = os.path.join(self.geosfp_dir, self.obs_time.strftime('%Y/%m/%d'), geosfp_name)
        ds_geosfp = xr.open_dataset(wind_file).isel(time=0)

        # interpolate data to the scene center
        u10 = ds_geosfp['U10M'].interp(lon=self.center_lon,
                                       lat=self.center_lat,
                                       method='nearest')

        v10 = ds_geosfp['V10M'].interp(lon=self.center_lon,
                                       lat=self.center_lat,
                                       method='nearest')

        return u10.item(), v10.item()
