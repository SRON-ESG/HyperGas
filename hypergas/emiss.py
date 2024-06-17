#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Calculate trace gas emission rate."""

import logging
import os
import random

import pandas as pd
import xarray as xr
from geopy.geocoders import Nominatim
from hypergas.plume_utils import (a_priori_mask_data, calc_emiss,
                                  calc_emiss_fetch)

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# constants
mass = {'ch4': 16.04e-3, 'co2': 44.01e-3}  # molar mass [kg/mol]
mass_dry_air = 28.964e-3  # molas mass dry air [kg/mol]
grav = 9.8  # gravity (m s-2)

'''
instrument data
    pixel_res: meters
    alpha: IME alphas
'''
emit_info = {
    'platform': 'EMIT', 'instrument': 'emi', 'provider': 'NASA-JPL', 'pixel_res': 60,
    'alpha_area': {'alpha1': 0., 'alpha2': 0.71, 'alpha3': 0.45},
    'alpha_point': {'alpha1': 0., 'alpha2': 0.31, 'alpha3': 0.50},
}

enmap_info = {
    'platform': 'EnMAP', 'instrument': 'hsi', 'provider': 'DLR', 'pixel_res': 30,
    'alpha_area': {'alpha1': 0., 'alpha2': 0.81, 'alpha3': 0.38},
    'alpha_point': {'alpha1': 0., 'alpha2': 0.44, 'alpha3': 0.40},
}
prisma_info = {
    'platform': 'PRISMA', 'instrument': 'hsi', 'provider': 'ASI', 'pixel_res': 30,
    'alpha_area': {'alpha1': 0., 'alpha2': 0.82, 'alpha3': 0.38},
    'alpha_point': {'alpha1': 0., 'alpha2': 0.44, 'alpha3': 0.40},
}
sensor_info = {'EMIT': emit_info, 'EnMAP': enmap_info, 'PRISMA': prisma_info}

# set global attrs for exported NetCDF file
AUTHOR = 'Xin Zhang'
EMAIL = 'xin.zhang@sron.nl; xinzhang1215@gmail.com'
INSTITUTION = 'SRON Netherlands Institute for Space Research'


class Emiss():
    """The Emiss class"""

    def __init__(self, ds, gas, plume_name):
        """Initialize Denoise.

        Args:
            ds (Dataset):
                The level2 or level3 product
            gas (str):
                The gas name (lowercase)
            plume_name (str):
                The plume index name (plume0, plume1, ....)
        """
        self.ds = ds
        self.gas = gas
        self.filename = ds.encoding['source']
        self.l1_filename = ds.attrs['filename']
        basename = os.path.basename(self.filename)
        if 'EMIT' in basename:
            self.sensor = 'EMIT'
        elif 'ENMAP' in basename:
            self.sensor = 'EnMAP'
        elif 'PRS' in basename:
            self.sensor = 'PRISMA'
        else:
            raise ValueError(f'{self.filename} is not supported.')
        self.plume_name = plume_name

    def mask_data(self, longitude, latitude,
                  wind_source='ERA5', land_only=True,
                  land_mask_source='GSHHS', only_plume=True,
                  azimuth_diff_max=30):
        """
        Create plume mask
        """
        l2b_html_filename = self.filename.replace('.nc', '.html')
        self.longitude = longitude
        self.latitude = latitude
        self.wind_source = wind_source
        self.land_only = land_only
        self.land_mask_source = land_mask_source
        self.azimuth_diff_max = azimuth_diff_max

        # select connected plume masks
        self.mask, lon_mask, lat_mask, self.plume_html_filename = a_priori_mask_data(l2b_html_filename, self.ds, self.gas,
                                                                                     self.longitude, self.latitude,
                                                                                     self.plume_name, self.wind_source,
                                                                                     self.land_only, self.land_mask_source,
                                                                                     only_plume, self.azimuth_diff_max)

    def export_plume_nc(self,):
        """
        Export plume data to L3 NetCDF file
        """
        # mask data
        gas_mask = self.ds[self.gas].where(self.mask)

        # calculate mean wind and surface pressure in the plume if they are existed
        if all(key in self.ds.keys() for key in ['u10', 'v10', 'sp']):
            u10 = self.ds['u10'].where(self.mask).mean(dim=['y', 'x'])
            v10 = self.ds['v10'].where(self.mask).mean(dim=['y', 'x'])
            sp = self.ds['sp'].where(self.mask).mean(dim=['y', 'x'])

            # keep attrs
            u10.attrs = self.ds['u10'].attrs
            v10.attrs = self.ds['v10'].attrs
            sp.attrs = self.ds['sp'].attrs
            array_list = [gas_mask, u10, v10, sp]
        else:
            array_list = [gas_mask]

        # save useful number for attrs
        sza = self.ds[self.gas].attrs['sza']
        vza = self.ds[self.gas].attrs['vza']
        start_time = self.ds[self.gas].attrs['start_time']

        # export masked data (plume)
        self.plume_nc_filename = self.filename.replace('.nc', f'_{self.plume_name}.nc').replace('L2', 'L3')

        # merge data
        ds_merge = xr.merge(array_list)

        # add crs info
        if self.ds.rio.crs:
            ds_merge.rio.write_crs(self.ds.rio.crs, inplace=True)

        # clear attrs
        ds_merge.attrs = ''

        # set global attributes
        header_attrs = {'author': AUTHOR,
                        'email': EMAIL,
                        'institution': INSTITUTION,
                        'filename': self.l1_filename,
                        'start_time': start_time,
                        'sza': sza,
                        'vza': vza,
                        'plume_longitude': self.longitude,
                        'plume_latitude': self.latitude,
                        }
        ds_merge.attrs = header_attrs

        LOG.info(f'Exported to {self.plume_nc_filename}')
        ds_merge.to_netcdf(self.plume_nc_filename)

    def estimate(self, ipcc_sector, wspd_manual=None, land_only=True):
        """
        Calculate the gas emission rate
        """
        info = sensor_info[self.sensor]
        pixel_res = info['pixel_res']

        if ipcc_sector == 'Solid Waste (6A)':
            alpha = info['alpha_area']
            alpha_replace = info['alpha_point']
        else:
            alpha = info['alpha_point']
            alpha_replace = info['alpha_area']

        wind_speed, wdir, wind_speed_all, wdir_all, wind_source_all, l_eff, u_eff, IME, Q, Q_err, \
            err_random, err_wind, err_calib = calc_emiss(self.gas, self.plume_nc_filename, self.plume_name,
                                                         alpha_replace,
                                                         pixel_res=pixel_res,
                                                         alpha=alpha,
                                                         wind_source=self.wind_source,
                                                         wspd=wspd_manual,
                                                         land_only=self.land_only,
                                                         land_mask_source=self.land_mask_source,
                                                         )

        # calculate emissions using the IME-fetch method with U10
        Q_fetch, Q_fetch_err, err_ime_fetch, err_wind_fetch \
            = calc_emiss_fetch(self.gas, self.plume_nc_filename,
                               longitude=self.longitude, latitude=self.latitude,
                               pixel_res=pixel_res,
                               wind_source=self.wind_source,
                               wspd=wind_speed
                               )

        # calculate plume bounds
        with xr.open_dataset(self.plume_nc_filename) as ds:
            plume_mask = ~ds[self.gas].isnull()
            lon_mask = ds['longitude'].where(plume_mask, drop=True)
            lat_mask = ds['latitude'].where(plume_mask, drop=True)
            t_overpass = pd.to_datetime(ds[self.gas].attrs['start_time'])

        bounds = [lon_mask.min().item(), lat_mask.min().item(),
                  lon_mask.max().item(), lat_mask.max().item()]

        # get the location attrs
        try:
            geolocator = Nominatim(user_agent='hypergas'+str(random.randint(1, 100)))
            location = geolocator.reverse(
                f'{self.latitude}, {self.longitude}', exactly_one=True, language='en')
            address = location.raw['address']
        except Exception as e:
            LOG.info('Can not access openstreetmap. Leave location info to empty.')
            address = {}

        # save ime results
        name = os.path.basename(os.path.dirname(self.filename)).replace('_', ' ')
        ime_results = {'plume_id': f"{info['instrument']}-{t_overpass.strftime('%Y%m%dt%H%M%S')}-{self.plume_name}",
                       'plume_latitude': self.latitude,
                       'plume_longitude': self.longitude,
                       'datetime': t_overpass.strftime('%Y-%m-%dT%H:%M:%S%z'),
                       'country': address.get('country', ''),
                       'state': address.get('state', ''),
                       'city': address.get('city', ''),
                       'name': name,
                       'ipcc_sector': ipcc_sector,
                       'gas': self.gas.upper(),
                       'plume_bounds': [bounds],
                       'instrument': info['instrument'],
                       'platform': info['platform'],
                       'provider': info['provider'],
                       'emission': Q,
                       'emission_uncertainty': Q_err,
                       'emission_uncertainty_random': err_random,
                       'emission_uncertainty_wind': err_wind,
                       'emission_uncertainty_calibration': err_calib,
                       'emission_fetch': Q_fetch,
                       'emission_fetch_uncertainty': Q_fetch_err,
                       'emission_fetch_uncertainty_ime': err_ime_fetch,
                       'emission_fetch_uncertainty_wind': err_wind_fetch,
                       'wind_speed': wind_speed,
                       'wind_direction': wdir,
                       'wind_source': self.wind_source,
                       'ime': IME,
                       'ueff_ime': u_eff,
                       'leff_ime': l_eff,
                       'alpha1': alpha['alpha1'],
                       'alpha2': alpha['alpha2'],
                       'alpha3': alpha['alpha3'],
                       'wind_speed_all': [wind_speed_all],
                       'wind_direction_all': [wdir_all],
                       'wind_source_all': [wind_source_all],
                       'azimuth_diff_max': self.azimuth_diff_max,
                       'land_only': self.land_only,
                       'land_mask_source': self.land_mask_source,

                       }

        # convert to DataFrame and export data as csv file
        df = pd.DataFrame(data=ime_results, index=[0])
        savename = self.plume_nc_filename.replace('.nc', '.csv')
        LOG.info(f'Exported estimates to {savename}')
        df.to_csv(savename, index=False)
