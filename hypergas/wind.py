#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Calculate u10 and v10 for the scene."""

import logging
import os

import pandas as pd
import xarray as xr
import yaml
from xarray import DataArray

LOG = logging.getLogger(__name__)


class Wind():
    """Calculate u10 and v10 from reanalysis wind data."""

    def __init__(self, scn):
        LOG.info('Reading wind data ...')
        # load settings
        _dirname = os.path.dirname(__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)['data']

        self.era5_dir = os.path.join(_dirname, settings['era5_dir'])
        self.geosfp_dir = os.path.join(_dirname, settings['geosfp_dir'])

        # get the obs time
        self.scene = scn
        self.obs_time = scn['radiance'].attrs['start_time']

        # calculate the lons and lats
        self.lons, self.lats = self.scene['radiance'].attrs['area'].get_lonlats()
        if not isinstance(self.lons, DataArray):
            self.lons = DataArray(self.lons, dims=("y", "x"))
            self.lats = DataArray(self.lats, dims=("y", "x"))

        # calculate the wind
        self.load_data()

    def load_data(self):
        """Load wind data."""
        # get the wind data
        u10_era5, v10_era5 = self.load_era5()
        u10_geosfp, v10_geosfp, sp_geosfp = self.load_geosfp()

        # combine them into DataArrays
        new_dim = pd.Index(['ERA5', 'GEOS-FP'], name='source')
        u10 = xr.concat([u10_era5, u10_geosfp], new_dim).rename('u10')
        v10 = xr.concat([v10_era5, v10_geosfp], new_dim).rename('v10')
        sp = sp_geosfp.rename('sp')

        # copy shared attrs
        all_attrs = self.scene['radiance'].attrs
        attrs_share = {key: all_attrs[key] for key in ['area', 'sensor', 'geotransform', 'spatial_ref', 'filename']
                       if key in all_attrs}
        u10.attrs = attrs_share
        v10.attrs = attrs_share
        sp.attrs = attrs_share

        u10.attrs['long_name'] = '10 metre U wind component'
        v10.attrs['long_name'] = '10 metre V wind component'
        sp.attrs['long_name'] = 'surface pressure'

        u10.attrs['units'] = 'm s-1'
        v10.attrs['units'] = 'm s-1'
        sp.attrs['units'] = 'Pa'

        self.u10 = u10
        self.v10 = v10
        self.sp = sp

    def load_era5(self):
        """Load local ERA5 wind data."""
        wind_file = os.path.join(self.era5_dir, self.obs_time.strftime('%Y/sl_%Y%m%d.grib'))

        # read the nearest ERA5 wind data
        ds_era5 = xr.open_dataset(wind_file, engine='cfgrib', indexpath='')
        ds_era5.coords['longitude'] = (ds_era5.coords['longitude'] + 180) % 360 - 180

        # interpolate data to the 2d scene
        u10 = ds_era5['u10'].interp(time=self.obs_time.strftime('%Y-%m-%d %H:%M'),
                                    longitude=self.lons,
                                    latitude=self.lats,
                                    )

        v10 = ds_era5['v10'].interp(time=self.obs_time.strftime('%Y-%m-%d %H:%M'),
                                    longitude=self.lons,
                                    latitude=self.lats,
                                    )

        # remove coords
        u10 = u10.reset_coords(drop=True)
        v10 = v10.reset_coords(drop=True)

        return u10, v10

    def load_geosfp(self):
        """Load local GEOS-FP wind data."""
        # read GEOS-FP by hour name
        geosfp_name = 'GEOS.fp.asm.tavg1_2d_slv_Nx.' + self.obs_time.strftime('%Y%m%d') \
            + '_' + '{:02d}{:02d}'.format(self.obs_time.hour, 30) + '.V01.nc4'
        wind_file = os.path.join(self.geosfp_dir, self.obs_time.strftime('%Y/%m/%d'), geosfp_name)
        ds_geosfp = xr.open_dataset(wind_file).isel(time=0)

        # interpolate data to the 2d scene
        u10 = ds_geosfp['U10M'].interp(lon=self.lons,
                                       lat=self.lats,
                                       )

        v10 = ds_geosfp['V10M'].interp(lon=self.lons,
                                       lat=self.lats,
                                       )

        sp = ds_geosfp['PS'].interp(lon=self.lons,
                                    lat=self.lats,
                                    )

        # remove coords
        u10 = u10.reset_coords(drop=True)
        v10 = v10.reset_coords(drop=True)
        sp = sp.reset_coords(drop=True)

        return u10, v10, sp
