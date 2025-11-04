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

import hypergas
from hypergas.ime_csf import IME_CSF
from hypergas.plume_utils import a_priori_mask_data, cm_mask_data

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)


class Emiss():
    """The Emiss class."""

    def __init__(self, ds, gas, plume_name):
        """Initialize Denoise.

        Parameters
        ----------
        ds : :class:`~xarray.Dataset`
            The level2 or level3 product.
        gas : str
            Gas name (lowercase, e.g., “ch4”).
        plume_name : str
            The plume index name ("plume0", "plume1", ....).
        """
        self.ds = ds
        self.gas = gas
        self.plume_name = plume_name
        self.filename = ds.encoding['source']
        self.l1_filename = ds.attrs['filename']

        # get the crs
        if self.ds.rio.crs:
            self.crs = self.ds.rio.crs
        else:
            self.crs = None

        # get the L3 plume NetCDF filename
        basename = os.path.basename(self.filename)
        if 'L2' in basename:
            self.plume_nc_filename = self.filename.replace('.nc', f'_{self.plume_name}.nc').replace('L2', 'L3')
        elif 'L3' in basename:
            self.plume_nc_filename = self.filename
            # we need to close the file because it will be read again in IME_CSF class
            self.ds.close()
        else:
            raise ValueError(f'{self.filename} is not supported. Please input the L2 or L3 nc filename.')

        if 'EMIT' in basename:
            self.sensor = 'EMIT'
        elif 'ENMAP' in basename:
            self.sensor = 'EnMAP'
        elif 'PRS' in basename:
            self.sensor = 'PRISMA'
        else:
            raise ValueError(f'{self.filename} is not supported.')

    def mask_data(self, longitude, latitude,
                  wind_source='ERA5', land_only=True,
                  land_mask_source='OSM', only_plume=True,
                  azimuth_diff_max=30,
                  dist_max=180,
                  ):
        """
        Create plume mask using :class:`hypergas.plume_utils.a_priori_mask_data`.
        """
        l2b_html_filename = self.filename.replace('.nc', '.html')
        self.longitude = longitude
        self.latitude = latitude
        self.wind_source = wind_source
        self.land_only = land_only
        self.land_mask_source = land_mask_source
        self.azimuth_diff_max = azimuth_diff_max
        self.dist_max = dist_max

        # select connected plume masks
        self.mask, lon_mask, lat_mask, \
            self.longitude, self.latitude, self.plume_html_filename = a_priori_mask_data(self.ds, self.gas,
                                                                                         self.longitude, self.latitude,
                                                                                         self.plume_name, self.wind_source,
                                                                                         only_plume, self.azimuth_diff_max, self.dist_max,
                                                                                         l2b_html_filename
                                                                                         )

        self.cm_mask, self.cm_threshold = cm_mask_data(self.ds, self.gas, self.longitude, self.latitude)

    def export_plume_nc(self,):
        """
        Export plume data to L3 NetCDF file with header attributes.
        """
        # mask data
        gas_mask = self.ds[self.gas].where(self.mask)
        with xr.set_options(keep_attrs=True):
            gas_cm_mask = self.ds[self.gas].where(self.cm_mask).rename(f'{self.gas}_cm')
            # remove background
            # gas_cm_mask -= self.cm_threshold
        gas_cm_mask.attrs['description'] = gas_cm_mask.attrs['description'] + ' masked by the Carbon Mapper v2 method'

        # calculate mean wind and surface pressure in the plume if they are existed
        if all(key in self.ds.keys() for key in ['u10', 'v10', 'sp']):
            u10 = self.ds['u10'].where(self.mask).mean(dim=['y', 'x'])
            v10 = self.ds['v10'].where(self.mask).mean(dim=['y', 'x'])
            sp = self.ds['sp'].where(self.mask).mean(dim=['y', 'x'])

            # keep attrs
            u10.attrs = self.ds['u10'].attrs
            v10.attrs = self.ds['v10'].attrs
            sp.attrs = self.ds['sp'].attrs
            array_list = [gas_mask, gas_cm_mask, u10, v10, sp]
        else:
            array_list = [gas_mask, gas_cm_mask]

        # save useful number for attrs
        sza = self.ds[self.gas].attrs['sza']
        vza = self.ds[self.gas].attrs['vza']
        start_time = self.ds[self.gas].attrs['start_time']

        # export masked data (plume)
        # merge data
        ds_merge = xr.merge(array_list)

        # add crs info
        if self.crs is not None:
            ds_merge.rio.write_crs(self.crs, inplace=True)

        # clear attrs
        ds_merge.attrs = ''

        # set global attributes
        header_attrs = {'version': hypergas.__name__+'_'+hypergas.__version__,
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

    def estimate(self, ipcc_sector, sp_manual=None, wspd_manual=None, land_only=True, name=None):
        """
        Calculate the gas emission rate using :class:`hypergas.ime_csf.IME_CSF`.
        """
        # init IME_CSF class
        ime_csf = IME_CSF(sensor=self.sensor, longitude_source=self.longitude, latitude_source=self.latitude,
                          plume_nc_filename=self.plume_nc_filename, plume_name=self.plume_name,
                          ipcc_sector=ipcc_sector, gas=self.gas, wind_source=self.wind_source,
                          sp_manual=sp_manual, wspd_manual=wspd_manual,
                          land_only=self.land_only, land_mask_source=self.land_mask_source)

        # calculate emission rates
        surface_pressure, wind_speed, wdir, wind_speed_all, wdir_all, wind_source_all, \
            l_ime, l_eff, u_eff, IME, Q, Q_err, err_random, err_wind, err_calib, \
            Q_fetch, Q_fetch_err, err_ime_fetch, err_wind_fetch, \
            IME_cm, l_cm, Q_cm, \
            ds_csf, n_csf, l_csf, u_eff_csf, Q_csf, Q_csf_err, err_random_csf, err_wind_csf, err_calib_csf = ime_csf.calc_emiss()

        # export csf data
        if ds_csf is not None:
            csf_filename = self.plume_nc_filename.replace('.nc', '_csf.nc')
            LOG.info(f'Exported CSF lines to {csf_filename}')
            ds_csf.to_netcdf(csf_filename)

        # get info
        info = ime_csf.sensor_info
        alpha = ime_csf.alpha
        beta = ime_csf.beta

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

        # set name by folder name if it is not specified
        if name is None:
            name = os.path.basename(os.path.dirname(self.filename)).replace('_', ' ')

        # save results
        results = {'plume_id': f"{info['instrument']}-{t_overpass.strftime('%Y%m%dt%H%M%S')}-{self.plume_name}",
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
                   'emission_cm': Q_cm,
                   'emission_fetch': Q_fetch,
                   'emission_fetch_uncertainty': Q_fetch_err,
                   'emission_fetch_uncertainty_ime': err_ime_fetch,
                   'emission_fetch_uncertainty_wind': err_wind_fetch,
                   'emission_csf': Q_csf,
                   'emission_csf_uncertainty': Q_csf_err,
                   'emission_csf_uncertainty_random': err_random_csf,
                   'emission_csf_uncertainty_wind': err_wind_csf,
                   'emission_csf_uncertainty_calibration': err_calib_csf,
                   'surface_pressure': surface_pressure,
                   'wind_source': self.wind_source,
                   'wind_speed': wind_speed,
                   'wind_direction': wdir,
                   'ime': IME,
                   'l_ime': l_ime,
                   'ueff_ime': u_eff,
                   'leff_ime': l_eff,
                   'ime_cm': IME_cm,
                   'l_cm': l_cm,
                   'ueff_csf': u_eff_csf,
                   'n_csf': n_csf,
                   'l_csf': l_csf,
                   'alpha1': alpha['alpha1'],
                   'alpha2': alpha['alpha2'],
                   'alpha3': alpha['alpha3'],
                   'beta1': beta['beta1'],
                   'beta2': beta['beta2'],
                   'wind_speed_all': [wind_speed_all],
                   'wind_direction_all': [wdir_all],
                   'wind_source_all': [wind_source_all],
                   'azimuth_diff_max': self.azimuth_diff_max,
                   'dist_max': self.dist_max,
                   'land_only': self.land_only,
                   'land_mask_source': self.land_mask_source,
                   'version': hypergas.__name__+'_'+hypergas.__version__,
                   }

        # convert to DataFrame and export data as csv file
        df = pd.DataFrame(data=results, index=[0])
        savename = self.plume_nc_filename.replace('.nc', '.csv')
        LOG.info(f'Exported estimates to {savename}')
        df.to_csv(savename, index=False)
