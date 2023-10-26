#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Calculate unit spectrum for matched filter."""

import logging
import os
from datetime import datetime
from math import cos, pi, radians

import numpy as np
import yaml

LOG = logging.getLogger(__name__)

MODE = {0: 'tropical', 1: 'midlatitudesummer', 2: 'midlatitudewinter',
        3: 'subarcticsummer', 4: 'subarcticwinter', 5: 'standard'}


class unit_spec():
    """Calculate the unit spectrum."""

    def __init__(self, radiance, wvl_min, wvl_max, fit_unit='lognormal'):
        """Initialize unit_spec class.

        Args:
            radiance (DataArray)
            wvl_min (float):
                The lower limit of wavelength [nm] for matched filter
            wvl_max (float):
                The upper limit of wavelength [nm] for matched filter
            fit_unit (str): the method ('lognormal', 'poly', or 'linear') of fitting the relationship between rads and conc
                Default: 'lognormal'
        """
        # load settings
        _dirname = os.path.dirname(__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)

        self.absorption_dir = os.path.join(_dirname, settings['absorption_dir'])
        self.irradiance_dir = os.path.join(_dirname, settings['irradiance_dir'])

        # load variables from "radiance" DataArray
        self.wvl_lowres = radiance['bands'].data
        self.fwhm = radiance['fwhm'].data
        self.sza = radiance.attrs['sza']
        self.vza = radiance.attrs['vza']
        self.wvl_min = wvl_min
        self.wvl_max = wvl_max
        self.fit_unit = fit_unit

        # read ref data
        date_time = radiance.attrs['start_time']
        self.doy = (date_time - datetime(date_time.year, 1, 1)).days + 1
        self.lat = radiance.attrs['area'].get_lonlats()[1].mean()
        self.refdata = self._read_refdata()

        # calculate rads based on conc
        self.conc, self.rads = self.convolve_rads()

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

        self.model = model

    def _read_abs(self, species):
        '''Read the absorption file'''
        abs_filename = f'absorption_cs_{species}_ALL_{MODE[self.model]}.csv'
        LOG.debug(f'Reading the absorption file: {abs_filename}')
        return np.genfromtxt(os.path.join(self.absorption_dir,
                                          abs_filename),
                             delimiter=',')

    def _read_refdata(self):
        '''Read reference data'''
        # determine the model name
        self._model()

        # read absorption data
        sigma_H2O = self._read_abs('H2O')
        sigma_CO2 = self._read_abs('CO2')
        sigma_N2O = self._read_abs('N2O')
        sigma_CO = self._read_abs('CO')
        sigma_CH4 = self._read_abs('CH4')

        # read typical absorption data
        atm_filename = f'atmosphere_{MODE[self.model]}.dat'
        LOG.debug(f'Read atm file: {atm_filename}')
        data_abs = np.loadtxt(os.path.join(self.absorption_dir, atm_filename))

        # read solar irradiance data
        #   if you modify the wavelength and use `type=radiance` in the `_radianceCalc` function
        #   you need to update the `w[i]` there
        E_filename = 'solar_irradiance_0400-2600nm_highres_sparse.dat'
        Edata = np.loadtxt(os.path.join(self.irradiance_dir, E_filename))
        Edata[:, 0] = np.round(Edata[:, 0], 2)

        # combine into one dict
        data = {'abs': data_abs, 'sigma_H2O': sigma_H2O, 'sigma_CO2': sigma_CO2,
                'sigma_N2O': sigma_N2O, 'sigma_CO': sigma_CO, 'sigma_CH4': sigma_CH4,
                'solar_irradiance': Edata
                }

        return data

    def _radianceCalc(self, del_omega, A=0.1, type='transmission'):
        """Function to calculate spectral radiance over selected band range
        based on methane del_omega (mol/m2) added to the first layer of atmosphere

        Args:
            del_omega (float):
                Methane column enhancement [mol/m2]
            A (float):
                albedo. This is cancelled out.
            type (str):
                returned data type: 'transmission' or 'radiance'

        Return:
            Wavelength range [nm] in the solar_irradiance data
            Transmission (if type is 'transmission') or spectral radiance [1/(s*cm^2*sr*nm), if type is 'radiance'] for the band range,
        """
        # column number density [cm^-2]
        # we need to assign the copied data, otherwise it will be overwrited in each loop
        nH2O = self.refdata['abs'][:, 3].copy()
        nCO2 = self.refdata['abs'][:, 4].copy()
        nN2O = self.refdata['abs'][:, 6].copy()
        nCO = self.refdata['abs'][:, 7].copy()
        nCH4 = self.refdata['abs'][:, 8].copy()

        # add species by "d_omega" [mol m-2] to the first layer
        nH2O[0] = nH2O[0] + del_omega['H2O'] * 6.023e+23 / 10000
        nCO2[0] = nCO2[0] + del_omega['CO2'] * 6.023e+23 / 10000
        nN2O[0] = nN2O[0] + del_omega['N2O'] * 6.023e+23 / 10000
        nCO[0] = nCO[0] + del_omega['CO'] * 6.023e+23 / 10000
        nCH4[0] = nCH4[0] + del_omega['CH4'] * 6.023e+23 / 10000
        LOG.debug(f"RadianceCalc omega2 : {nCH4[0]}")

        nLayer = nH2O.shape[0]

        wavenumber1 = np.round((1e+07)/self.wvl_min, 1)  # [cm^-1]
        wavenumber2 = np.round((1e+07)/self.wvl_max, 1)  # [cm^-1]

        Edata = self.refdata['solar_irradiance']

        # because the generated abs is from 400 to 2600 nm, we need to make sure the index is same
        Edata_subset = Edata[np.where(Edata[:, 0] < 1e7/2600)[0][-1]:np.where(Edata[:, 0] > 1e7/400)[0][0]]

        # crop to interested wavelength range
        id2 = np.where(Edata_subset[:, 0] < wavenumber1)[0][-1]
        id1 = np.where(Edata_subset[:, 0] > wavenumber2)[0][0]
        w = Edata_subset[id1:id2, 0]
        wavelength = 1e+07/w  # [nm]

        sigma_H2O_subset = self.refdata['sigma_H2O'][id1:id2]
        sigma_CO2_subset = self.refdata['sigma_CO2'][id1:id2]
        sigma_N2O_subset = self.refdata['sigma_N2O'][id1:id2]
        sigma_CO_subset = self.refdata['sigma_CO'][id1:id2]
        sigma_CH4_subset = self.refdata['sigma_CH4'][id1:id2]

        optd_H2O = np.matmul(sigma_H2O_subset, nH2O[0:nLayer])
        optd_CO2 = np.matmul(sigma_CO2_subset, nCO2[0:nLayer])
        optd_N2O = np.matmul(sigma_N2O_subset, nN2O[0:nLayer])
        optd_CO = np.matmul(sigma_CO_subset, nCO[0:nLayer])
        optd_CH4 = np.matmul(sigma_CH4_subset, nCH4[0:nLayer])

        tau_vert = optd_H2O + optd_CO2 + optd_N2O + optd_CO + optd_CH4

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

        if type == 'transmission':
            return wavelength, transmission
        elif type == 'radiance':
            E_lambda = np.zeros([w.shape[0]])
            w = np.round(w, 2)

            for i in range(w.shape[0]):
                # 400 -- 2600 nm
                if w[i] <= 1e7/2600:
                    w[i] = 1e7/2600
                elif w[i] >= 1e7/400:
                    w[i] = 1e7/400
                index = int(np.where(Edata_subset[:, 0] == w[i])[0])
                E_lambda[i] = Edata_subset[index, 1]

            # irradiance to radiance
            consTerm = A * cos(radians(self.sza)) / pi
            L_lambda = consTerm * np.exp(-1*tau_lambda) * E_lambda
            return wavelength, L_lambda
        else:
            raise ValueError(f"Unrecognized type: {type}. It should be 'transmission' or 'radiance'")

    def _convolve(self, rads):
        '''Convert high-resolution spectral radiance to low-resolution signal
        and create the unit methane absorption spectrum

        References:
            https://github.com/markusfoote/mag1c/blob/8b9ceae186f4e125bc9f628db82f41bce4c6011f/mag1c/mag1c.py#L229-L243
            https://github.com/Prikaziuk/retrieval_rtmo/blob/master/src/%2Bhelpers/create_sensor_from_fwhm.m
        '''
        sigma = self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Evaluate normal distribution explicitly
        var = sigma ** 2
        denom = (2 * np.pi * var) ** 0.5
        numer = np.exp(-(self.wvl_highres[:, None] - self.wvl_lowres[None, :])**2 / (2*var))
        # numer = np.exp(-(np.asarray(self.wvl_highres)[:, None] - self.wvl_lowres[None, :])**2 / (2*var))
        response = numer / denom

        # Normalize each gaussian response to sum to 1.
        response = np.divide(response, response.sum(
            axis=0), where=response.sum(axis=0) > 0, out=response)

        # implement resampling as matrix multiply
        resampled = rads.dot(response)

        return resampled

    def convolve_rads(self):
        """Calculate the convolved sensor-reaching rads or transmissions.

        Return
            conc (1d array): the manually set concentrations
            rads (2d array, [conc*wvl]): radiances or transmissions for `conc`
        """
        # create an array of CH4 concentrations
        #   you can modify it, but please keep the first one as zero
        conc = np.array([0, 100, 200, 400, 800, 1600, 3200, 6400])  # ppb

        # set the enhancement of multiple gases
        #   xch4 is converted from ppb to mol/m2 by divideing by 2900
        delta_omega = {'H2O': 0, 'CO2': 0, 'N2O': 0, 'CO': 0, 'CH4': conc/2900}

        # calculate the transmission or sensor-reaching radiance with these gases
        # reference
        w0, rad0 = self._radianceCalc({x: 0 for x in delta_omega})

        # calculate the rads with 1 ppm CH4 for `unit_spec`
        tmp_omega = delta_omega.copy()
        tmp_omega.update({'CH4': 1000/2900})
        self.wvl_highres, rad_unit = self._radianceCalc(tmp_omega)
        self.rad_unit = self._convolve(rad_unit)

        # create array for saving radiance data
        rads = np.zeros((len(conc), len(w0)))

        # iterate each conc and calculate the tau
        for i, omega in enumerate(delta_omega['CH4']):
            tmp_omega = delta_omega.copy()
            tmp_omega.update({'CH4': omega})
            wvl_highres, rad = self._radianceCalc(tmp_omega)
            rads[i, :] = rad

        resampled = self._convolve(rads)

        return conc, resampled

    def fit_slope(self):
        """Fit the slope for conc and rads"""
        if self.fit_unit == 'lognormal':
            lograd = np.log(self.rads, out=np.zeros_like(self.rads), where=self.rads > 0)

            # calculate slope [ln(Δradiance)/ Δc]: ln(xm) = ln(xr) - kΔc
            #   Ref: Schaum (2021) and Pei (2023)
            slope, residuals, _, _ = np.linalg.lstsq(
                np.stack((np.ones_like(self.conc), self.conc)).T, lograd, rcond=None)

            K = slope[1, :]

        elif self.fit_unit == 'poly':
            # Degree of the polynomial: y = ax^2 + bx + c
            n_pol_jac = 2
            jac_gas = np.zeros((n_pol_jac+1, self.rads.shape[1]))
            delta_rad = self.rads / self.rads[0, :]  # Equivalent to L1/L0
            # Change in methane concentration = Delta_XCH4 = XCH4(L1)-XCH4(L0), the first conc is 0
            delta_mr = self.conc

            for i in range(self.rads.shape[1]):
                # Function that relates L1/L0 and Delta_XCH4: Second order Polynomial
                jac_gas[:, i] = np.polyfit(delta_mr, delta_rad[:, i], n_pol_jac)

            # get the derivate with 1 ppm for the unit spectrum
            unit_conc = 1000  # ppb
            K = 2 * jac_gas[0, :] * unit_conc + jac_gas[1, :]  # Derivative of the Second order Poynomial

        elif self.fit_unit == 'linear':
            unit_conc = 1000  # ppb
            # first-order Taylor expansion: xm = xr(1-kΔc)
            K = (self.rad_unit-self.rads[0, :]) / unit_conc / (self.rads[0, :] + 1e-12)

        # ensuring numerical stability
        SCALING = 1e5

        return K * SCALING
