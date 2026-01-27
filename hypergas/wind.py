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
from typing import Mapping

import requests
import numpy as np
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
        u10_openmeteo, v10_openmeteo, sp_openmeteo = self.load_openmeteo()

        # combine them into DataArrays
        new_dim = pd.Index(['ERA5', 'GEOS-FP', 'Open-Meteo'], name='source')
        u10 = xr.concat([u10_era5, u10_geosfp, u10_openmeteo], new_dim).rename('u10')
        v10 = xr.concat([v10_era5, v10_geosfp, v10_openmeteo], new_dim).rename('v10')

        # the SRON server does not save the pressure data
        sp_dim = pd.Index(['GEOS-FP', 'Open-Meteo'], name='source')
        sp = xr.concat([sp_geosfp, sp_openmeteo], sp_dim).rename('sp')

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

    @staticmethod
    def _interp_angle(
        nrows: int,
        ncols: int,
        corners: Mapping[str, float],
    ) -> np.ndarray:
        """
        Bilinear interpolation with center constraint.

        Parameters
        ----------
        nrows, ncols : int
            Output dimensions
        corners : mapping
            Corner angles with keys: UL, UR, LL, LR, C

        Returns
        -------
        angle : np.ndarray (nrows, ncols)
            Interpolated angle field [degrees]
        """
        required = {"UL", "UR", "LL", "LR", "C"}
        missing = required - corners.keys()
        if missing:
            raise ValueError(f"Missing corner keys: {missing}")

        ul = corners["UL"]
        ur = corners["UR"]
        ll = corners["LL"]
        lr = corners["LR"]
        center = corners["C"]

        # Normalized coordinates
        y = np.linspace(0.0, 1.0, nrows, dtype=np.float64)
        x = np.linspace(0.0, 1.0, ncols, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        # Bilinear interpolation from corners
        angle = (
            ul * (1 - xx) * (1 - yy)
            + ur * xx * (1 - yy)
            + ll * (1 - xx) * yy
            + lr * xx * yy
        )

        # Apply center constraint
        center_bilin = 0.25 * (ul + ur + ll + lr)
        delta = center - center_bilin

        # Weight: zero at edges, one at center
        w = 4.0 * xx * (1 - xx) * yy * (1 - yy)

        return angle + delta * w

    def load_openmeteo(self):
        """
        Fetch OpenMeteo data at domain corners and center, then interpolate to 2D grid.

        Returns
        -------
        u10, v10, sp : np.ndarray
            2D fields of U-wind, V-wind (m/s), and surface pressure (Pa)
        """
        nrows, ncols = self.lats.shape

        # Define corner and center positions
        positions = {
            'UL': (self.lats[0, 0], self.lons[0, 0]),           # Upper-left
            'UR': (self.lats[0, -1], self.lons[0, -1]),         # Upper-right
            'LL': (self.lats[-1, 0], self.lons[-1, 0]),         # Lower-left
            'LR': (self.lats[-1, -1], self.lons[-1, -1]),       # Lower-right
            'C': (self.lats[nrows//2, ncols//2], self.lons[nrows//2, ncols//2])  # Center
        }

        # Fetch data at each position
        wspd_dict = {}
        wdir_dict = {}
        sp_dict = {}

        for key, (lat, lon) in positions.items():
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": lat.item(),
                "longitude": lon.item(),
                "start_date": self.obs_time.strftime('%Y-%m-%d'),
                "end_date": self.obs_time.strftime('%Y-%m-%d'),
                "hourly": ["wind_speed_10m", "wind_direction_10m", "surface_pressure"],
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            }

            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Extract hourly data
                hourly = data['hourly']
                ds_hourly = xr.Dataset(
                    {
                        'wspd': (['time'], hourly['wind_speed_10m']),
                        'wdir': (['time'], hourly['wind_direction_10m']),
                        'pressure': (['time'], hourly['surface_pressure']),
                    },
                    coords={'time': pd.to_datetime(hourly['time'])}
                )

                # Interpolate to exact observation time
                ds_interp = ds_hourly.interp(time=self.obs_time)

                wspd_dict[key] = float(ds_interp['wspd'].values)
                wdir_dict[key] = float(ds_interp['wdir'].values)
                sp_dict[key] = float(ds_interp['pressure'].values) * 100  # hPa to Pa

            except Exception as e:
                raise RuntimeError(f"Failed to fetch data for {key}: {e}")

        # Interpolate wind direction and speed separately
        wdir_2d = self._interp_angle(nrows, ncols, wdir_dict)
        wspd_2d = self._interp_angle(nrows, ncols, wspd_dict)
        sp_2d = self._interp_angle(nrows, ncols, sp_dict)

        # Convert wind speed and direction to U and V components
        rad = np.deg2rad(wdir_2d)
        u10 = -wspd_2d * np.sin(rad)
        v10 = -wspd_2d * np.cos(rad)

        u10 = xr.DataArray(u10, dims=['y', 'x'])
        v10 = xr.DataArray(v10, dims=['y', 'x'])
        sp_2d = xr.DataArray(sp_2d, dims=['y', 'x'])

        return u10, v10, sp_2d
