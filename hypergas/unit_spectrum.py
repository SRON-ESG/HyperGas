#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Calculate unit spectrum for matched filter."""

import logging
import os
from datetime import datetime
from math import cos, pi, radians

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .lut_interp import spline_5deg_lookup

LOG = logging.getLogger(__name__)

MODE = {0: 'tropical', 1: 'midlatitudesummer', 2: 'midlatitudewinter',
        3: 'subarcticsummer', 4: 'subarcticwinter', 5: 'standard'}


class Unit_spec():
    """Calculate the unit spectrum."""

    def __init__(self, radiance, wvl_sensor, wvl_min, wvl_max,
                 species='ch4', rad_source='model'):
        """Initialize unit_spec class.

        Parameters
        ----------

        radiance : :class:`~xarray.DataArray`
            3D radiance dataarray (bands, y, x).
        wvl_sensor : :class:`~xarray.DataArray`
            1D ['bands'] or 2D['bands','x'].
            The central wavelengths (nm) of sensor.
        wvl_min : float
            The lower limit of wavelength (nm) for matched filter.
        wvl_max : float
            The upper limit of wavelength (nm) for matched filter.
        species : str
            The species to be retrieved: 'ch4' or 'co2'.
            Default: 'ch4'.
        rad_source : str
            The data ('model' or 'lut') used for calculating rads or transmissions.
            Default: 'model'.
            References:
                - model: `Gloudemans et al. (2008) <https://doi.org/10.5194/acp-8-3999-2008>`_
                - lut: only supporting ch4 and co2; `Foote et al. (2021) <https://hive.utah.edu/concern/datasets/9w0323039>`_.
        """
        # load settings
        _dirname = os.path.dirname(__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)
            data_setting = settings['data']
            species_setting = settings['species']

        self.absorption_dir = os.path.join(_dirname, data_setting['absorption_dir'])
        self.irradiance_dir = os.path.join(_dirname, data_setting['irradiance_dir'])
        self.modtran_dir = os.path.join(_dirname, data_setting['modtran_dir'])
        self.rad_source = rad_source

        # load variables from the "radiance" DataArray
        self.radiance = radiance
        self.wvl_sensor = wvl_sensor
        self.fwhm_sensor = radiance['fwhm']
        self.sza = radiance.attrs['sza']
        self.vza = radiance.attrs['vza']
        self.wvl_min = wvl_min
        self.wvl_max = wvl_max

        # create an array of concentrations
        #   you can modify it, but please keep the first one as zero
        #   the unit should be ppm for rad_source=='model' or 'ppm m' for rad_source=='lut'
        self.conc = np.array(species_setting[species]['concentrations'])

        self.species = species.upper()

        # read ref data
        date_time = radiance.attrs['start_time']
        self.doy = (date_time - datetime(date_time.year, 1, 1)).days + 1
        self.lat = radiance.attrs['area'].get_lonlats()[1].mean()
        self._read_refdata()

    def _model(self):
        """ Determine atmospheric model

            0 - Tropical
            1 - Mid-Latitude Summer
            2 - Mid-Latitude Winter
            3 - Sub-Arctic Summer
            4 - Sub-Arctic Winter
            5 - US Standard Atmosphere
        """
        # Determine season
        if self.doy < 121 or self.doy > 274:
            if self.lat < 0:
                summer = True
            else:
                summer = False
        else:
            if self.lat < 0:
                summer = False
            else:
                summer = True

        # Determine model
        if abs(self.lat) <= 15:
            model = 0
        elif abs(self.lat) >= 60:
            if summer:
                model = 3
            else:
                model = 4
        else:
            if summer:
                model = 1
            else:
                model = 2

        # fixme: NO2 only supports US Standard Atmosphere
        if self.species == 'NO2':
            model = 5

        self.model = model

    def _read_refdata(self):
        '''Read reference data'''
        # determine the model name
        self._model()

        # read atmosphere profile
        atm_filename = f'atmosphere_{MODE[self.model]}.dat'
        LOG.info(f'Read atm file: {atm_filename}')
        df_atm = pd.read_csv(os.path.join(self.absorption_dir, atm_filename), comment='#', header=None, sep='\t')

        if len(df_atm.columns) == 10:
            col_names = ['thickness', 'pressure', 'temperature', 'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2']
        if len(df_atm.columns) == 11:
            # US standard atmosphere profile
            col_names = ['thickness', 'pressure', 'temperature', 'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO2']
        # !!! test
        # if len(df_atm.columns) == 12:
        #    col_names = ['thickness', 'pressure', 'temperature', 'H2O', 'CO2', 'O3', 'N2O', 'CO', 'CH4', 'O2', 'NO2', 'O4']

        df_atm.columns = col_names

        # fixme: set NO2 to 0 if the column is not existed.
        if 'NO2' not in col_names:
            df_atm['NO2'] = 0
            # !!! test
            # df_atm['O4'] = 0

        # read solar irradiance data
        E_filename = 'solar_irradiance_0400-2600nm_highres_sparse.nc'
        Edata = xr.open_dataset(os.path.join(self.irradiance_dir, E_filename))['irradiance']

        # convert units to W m-2 um-1
        Edata *= 1e3

        # read abs data
        abs_filename = f'absorption_cs_ALL_{MODE[self.model]}.nc'
        LOG.debug(f'Reading the absorption file: {abs_filename}')
        ds_abs = xr.open_dataset(os.path.join(self.absorption_dir, abs_filename))

        # save into class
        self.atm = df_atm
        self.solar_irradiance = Edata
        self.abs = ds_abs

    def _radianceCalc(self, del_omega, albedo=0.15, return_type='transmission'):
        """Function to calculate spectral radiance over selected band range
        based on trace gas del_omega (mol/m2) added to the first layer of atmosphere

        Parameters
        ----------

        del_omega : float
            Trace gas column enhancement [mol/m2].
        albedo : float
            Albedo. This is cancelled out in the matched filter.
        return_type : str
            Returned data type: 'transmission' or 'radiance'.

        Returns
        -------
        Wavelength : array
            The wavelength range (nm) in the solar_irradiance data.

        Transmission or spectral radiance: array
            If return_type is 'transmission', Transmission is returned.
            If return_type is 'radiance', spectral radiance [1/(s*cm^2*sr*nm) is returned.
        """
        # column number density [cm^-2]
        # we need to assign the copied data, otherwise it will be overwrited in each loop
        nH2O = self.atm['H2O'].copy().values
        nCO2 = self.atm['CO2'].copy().values
        nO3 = self.atm['O3'].copy().values
        nN2O = self.atm['N2O'].copy().values
        nCO = self.atm['CO'].copy().values
        nCH4 = self.atm['CH4'].copy().values
        nNO2 = self.atm['NO2'].copy().values
        # nO4 = self.atm['O4'].copy().values

        # add species by "d_omega" [mol m-2] to the first layer
        nH2O[0] = nH2O[0] + del_omega['H2O'] * 6.023e+23 / 10000
        nCO2[0] = nCO2[0] + del_omega['CO2'] * 6.023e+23 / 10000
        nO3[0] = nO3[0] + del_omega['O3'] * 6.023e+23 / 10000
        nN2O[0] = nN2O[0] + del_omega['N2O'] * 6.023e+23 / 10000
        nCO[0] = nCO[0] + del_omega['CO'] * 6.023e+23 / 10000
        nCH4[0] = nCH4[0] + del_omega['CH4'] * 6.023e+23 / 10000
        nNO2[0] = nNO2[0] + del_omega['NO2'] * 6.023e+23 / 10000
        LOG.debug(f"RadianceCalc omega2: {nCH4[0]}")

        # crop solar irradiance and absorption data to the same wavelength range
        #   the numeric issue could lead to one or two offset
        #   so it is better to use the nearest method to get the same data with small tolerance
        wavelength = self.abs['abs_H2O'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).coords['wavelength'].data
        Edata = self.solar_irradiance
        Edata_subset = Edata.sel(wavelength=wavelength, tolerance=1e-5, method='nearest')
        Edata_subset = Edata_subset.data

        sigma_H2O_subset = self.abs['abs_H2O'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).data
        sigma_CO2_subset = self.abs['abs_CO2'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).data
        sigma_N2O_subset = self.abs['abs_N2O'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).data
        sigma_CO_subset = self.abs['abs_CO'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).data
        sigma_CH4_subset = self.abs['abs_CH4'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).data

        optd_H2O = np.matmul(sigma_H2O_subset, nH2O)
        optd_CO2 = np.matmul(sigma_CO2_subset, nCO2)
        optd_N2O = np.matmul(sigma_N2O_subset, nN2O)
        optd_CO = np.matmul(sigma_CO_subset, nCO)
        optd_CH4 = np.matmul(sigma_CH4_subset, nCH4)

        if 'NO2' in list(self.abs.keys()):
            # fixme: only calculate optd when the NO2 profile is available (standard model)
            sigma_NO2_subset = self.abs['abs_NO2'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).data
            optd_NO2 = np.matmul(sigma_NO2_subset, nNO2)
            sigma_O3_subset = self.abs['abs_O3'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).data
            optd_O3 = np.matmul(sigma_O3_subset, nO3)
            # !!! test
            # sigma_O4_subset = self.abs['abs_O4'].sel(wavelength=slice(self.wvl_min, self.wvl_max)).data
            # optd_O4 = np.matmul(sigma_O4_subset, nO4)
        else:
            optd_NO2 = 0
            optd_O3 = 0
            # !!! test
            # optd_O4 = 0

        tau_vert = optd_H2O + optd_CO2 + optd_O3 + optd_N2O + optd_CO + optd_CH4 + optd_NO2  # + optd_O4

        def f_young(za):
            za = radians(za)
            # air mass defined in Young (1994): https://encyclopedia.pub/entry/28536
            f = (1.002432*(cos(za))**2 + 0.148386*cos(za) + 0.0096467)\
                / ((cos(za))**3 + 0.149864*(cos(za))**2 + 0.0102963*cos(za) + 0.000303978)
            return f

        # amf: air mass factor
        amf = f_young(self.sza) + f_young(self.vza)

        # transmission = exp(-tau*amf), Ref: Eq.2 of Jongaramrungruang (2021)
        tau_lambda = amf * tau_vert
        transmission = np.exp(-tau_lambda)

        if return_type == 'transmission':
            return wavelength, transmission
        elif return_type == 'radiance':
            # irradiance to radiance
            self.albedo = albedo
            consTerm = albedo * cos(radians(self.sza)) / pi
            L_lambda = consTerm * np.exp(-tau_lambda) * Edata_subset
            return wavelength, L_lambda
        else:
            raise ValueError(f"Unrecognized return_type: {return_type}. It should be 'transmission' or 'radiance'")

    def _convolve(self, wvl_sensor, fwhm_sensor, wvl_lut, rad_lut):
        '''Convert high-resolution spectral radiance to low-resolution signal
        and create the unit methane absorption spectrum

        References:
            https://github.com/markusfoote/mag1c/blob/8b9ceae186f4e125bc9f628db82f41bce4c6011f/mag1c/mag1c.py#L229-L243
            https://github.com/Prikaziuk/retrieval_rtmo/blob/master/src/%2Bhelpers/create_sensor_from_fwhm.m
        '''
        # convert to numpy array
        if not isinstance(wvl_sensor, np.ndarray):
            wvl_sensor = wvl_sensor.data
        if not isinstance(fwhm_sensor, np.ndarray):
            fwhm_sensor = fwhm_sensor.data

        sigma = fwhm_sensor / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Evaluate normal distribution explicitly
        var = sigma ** 2
        denom = (2 * np.pi * var) ** 0.5
        numer = np.exp(-(wvl_lut[:, None] - wvl_sensor[None, :])**2 / (2*var))
        response = numer / denom

        # Normalize each gaussian response to sum to 1.
        response = np.divide(response, response.sum(
            axis=0), where=response.sum(axis=0) > 0, out=response)

        # implement resampling as matrix multiply
        resampled = rad_lut.dot(response)

        return resampled

    def convolve_rads(self):
        """Calculate the convolved sensor-reaching rads or transmissions.

        Returns
        -------
        convolved rads : :class:`~xarray.DataArray`
            2D (conc*wvl) or 3D (bands*conc*wvl) radiances or transmissions for ``conc`` defined in the ``config.yaml`` file.
        """

        # set the enhancement of multiple gases
        #   xch4 is converted from ppb to mol/m2 by divideing by 2900
        delta_omega = {'H2O': 0, 'CO2': 0, 'O3': 0, 'N2O': 0, 'CO': 0, 'CH4': 0, 'NO2': 0}
        delta_omega.update({self.species: self.conc*1000/2900})

        # calculate the transmission or sensor-reaching radiance with these gases
        # reference
        self.wvl_lut, self.rad_lut = self._radianceCalc({x: 0 for x in delta_omega}, return_type='radiance')

        # create array for saving radiance data
        rads_omega = np.zeros((len(self.conc), len(self.wvl_lut)))

        # iterate each conc and calculate the tau
        for i, omega in enumerate(delta_omega[self.species]):
            tmp_omega = delta_omega.copy()
            tmp_omega.update({self.species: omega})
            _, rad = self._radianceCalc(tmp_omega)
            rads_omega[i, :] = rad

        if len(self.wvl_sensor.dims) == 1:
            # dims: conc, bands
            resampled = self._convolve(self.wvl_sensor, self.fwhm_sensor, self.wvl_lut, rads_omega)
        elif len(self.wvl_sensor.dims) == 2:
            # dims: x, conc, bands
            self.wvl_sensor.load()
            resampled = xr.apply_ufunc(self._convolve,
                                       self.wvl_sensor,
                                       kwargs={'fwhm_sensor': self.fwhm_sensor,
                                               'wvl_lut': self.wvl_lut,
                                               'rad_lut': rads_omega},
                                       exclude_dims=set(('bands',)),
                                       input_core_dims=[['bands', ]],
                                       output_core_dims=[['conc', 'bands']],
                                       vectorize=True,
                                       dask='parallelized',
                                       output_dtypes=['float64'],
                                       dask_gufunc_kwargs=dict(output_sizes={'x': self.wvl_sensor.sizes['x'],
                                                                             'conc': len(self.conc),
                                                                             'bands': self.wvl_sensor.sizes['bands']
                                                                             })
                                       )
        else:
            raise ValueError(f'self.wvl_sensor should be 1D or 2D. Your input is {self.wvl_sensor.sizes}')

        return resampled

    def convolve_rads_lut(self):
        """Calculate the convolved sensor-reaching rads.

        Returns
        -------

        convolved rads : :class:`~xarray.DataArray`
            2D (conc*wvl) radiances for ``conc`` defined in the ``config.yaml`` file.
        """
        # set params for LUT
        sensor_altitude = 100
        ground_elevation = 0
        order = 1  # Spline interpolation degree
        water_vapor = 1.3

        param = {'zenith': self.sza,
                 # Model uses sensor height above ground
                 'sensor': sensor_altitude - ground_elevation,
                 'ground': ground_elevation,
                 'water': water_vapor,
                 'gas': self.species.lower(),
                 'order': order
                 }

        # read LUT
        lut_file = os.path.join(self.modtran_dir, f'dataset_{self.species.lower()}_full.hdf5')
        lut = h5py.File(lut_file, 'r', rdcc_nbytes=4194304)
        self.wvl_lut = lut['wave'][:]
        grid_data = lut['modtran_data']

        # calculate the rads with 1 ppm m CH4 for `unit_spec`
        # rad_unit = spline_5deg_lookup(grid_data, conc=1, **param)

        # array for saving rads
        rads_omega = np.empty((len(self.conc), grid_data.shape[-1]))

        # iterate each conc and calculate the rads
        for i, ppmm in enumerate(self.conc):
            rads_omega[i, :] = spline_5deg_lookup(grid_data, conc=ppmm, **param)

        if len(self.wvl_sensor.dims) == 1:
            resampled = self._convolve(self.wvl_sensor, self.fwhm_sensor, self.wvl_lut, rads_omega)
        else:
            raise ValueError(f'self.wvl_sensor should be 1D for LUT. Your input is {self.wvl_sensor.sizes}')

        return resampled

    def _unit_fit(self, rads):
        # https://github.com/markusfoote/mag1c/blob/8b9ceae186f4e125bc9f628db82f41bce4c6011f/mag1c/mag1c.py#L241
        lograd = np.log(rads, out=np.zeros_like(rads), where=rads > 0)

        # calculate slope [ln(Δradiance)/ Δc]: ln(xm) = ln(xr) - kΔc
        #   Ref: Schaum (2021) and Pei (2023)
        slope, residuals, _, _ = np.linalg.lstsq(
            np.stack((np.ones_like(self.conc), self.conc)).T, lograd, rcond=None)

        K = slope[1, :]

        return K

    def fit_slope(self, scaling=None):
        """Fit the slope for conc and rads.

        Parameters
        ----------
        scaling : float
            The scaling factor to ensure numerical stability. Default: ``None``.
        """
        # calculate rads based on conc
        LOG.info(f'Convolving rads ({self.wvl_min}~{self.wvl_max} nm) ...')
        if self.rad_source == 'model':
            rads = self.convolve_rads()
        elif self.rad_source == 'lut':
            rads = self.convolve_rads_lut()
        else:
            raise ValueError(f"rad_source only supports 'model' or 'lut'). {self.rad_source} is not supported.")

        LOG.info('Convolving rads (Done)')

        if len(rads.shape) == 2:
            K = self._unit_fit(rads)
        elif len(rads.shape) == 3:
            K_list = []
            for x in range(rads.shape[0]):
                K = self._unit_fit(rads[x, ...])
                K_list.append(K)
            K = np.stack(K_list).T  # dims: bands, x

        if scaling is None:
            # auto calculation of scaling factor
            K_negative_max = K[K < 0].max()
            if K_negative_max > -1:
                scaling = round(-1/K_negative_max, 1)
            else:
                scaling = round(-K_negative_max, 1)

        return K * scaling, scaling
