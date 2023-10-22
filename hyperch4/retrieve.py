#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Retrieve methane enhancements using hyperspectral satellite data."""

import sys

import spectral.algorithms as algo
import numpy as np
import xarray as xr
from spectral.algorithms.detectors import matched_filter

from .unit_spectrum import unit_spec

# the scaling factor for alpha
#   this should be set as same as that in `unit_spectrum.py`
SCALING = 1e5


class MatchedFilter():
    """The MatchedFilter Class."""

    def __init__(self, radiance, wvl_intervals, fit_unit='lognormal'):
        """Initialize MatchedFilter.

        To apply matched filter, `radiance` must be specified::

            alpha = MatchedFilter(radiance=<DataArray>)

        Args:
            radiance (xarray DataArray):
                name: "radiance"
                dims: ['bands', 'y', 'x']
                units: mW m^-2 sr^-1 nm^-1,
                coordinates: at least 1) "wavelength" (nm) and 2) "fwhm" (nm).
            wvl_intervals (list): The wavelength range [nm] used in matched filter. It can be one list or nested list.
                e.g. [2110, 2450] or [[1600, 1750], [2110, 2450]]
                Deafult: [2110, 2450].
            fit_unit (str): The fitting method ('lognormal', 'poly', or 'linear') to calculate the unit CH4 spectrum
                            Default: 'lognormal'
        """
        # set the wavelength range for matched filter
        self.wvl_min = wvl_intervals[0]
        self.wvl_max = wvl_intervals[1]

        # subset data to selected wavelength range for matched filter
        wvl_mask = (radiance['bands'] >= self.wvl_min) & (radiance['bands'] <= self.wvl_max)
        radiance = radiance.where(wvl_mask, drop=True)

        self.radiance = radiance

        # --- test norm ---

        # calculate unit spectrum
        self.fit_unit = fit_unit
        self.K = unit_spec(self.radiance, self.wvl_min, self.wvl_max, self.fit_unit).fit_slope()

    def _printt(self, outm):
        """Refreshing print."""
        sys.stdout.write("\r" + outm)
        sys.stdout.flush()

    # def _norm(self):
    #         from sklearn.preprocessing import MinMaxScaler
    #         data = data.reshape((-1, data.shape[-1]))
    #         self.radiance = self.radiance.stack(z=('y', 'x'))
    #         scaler = MinMaxScaler()
    #         scaler.fit(self.radiance)
    #     return scaler.transform(data)

    def col_matched_filter(self, radiance, K):
        """Calculate stats of data."""
        # create nan mask
        mask = ~np.isnan(radiance).any(axis=1)
        # calculate stats
        background = algo.calc_stats(radiance, mask=mask, index=None)
        # get mean value
        mu = background.mean

        # calculate the target spectrum
        target = K * mu

        # apply matched filter
        a = matched_filter(radiance, target, background)

        return a

    def smf(self):
        """Standard/Robust matched filter

            Compute mean and covariance of set of each column and then run standard matched filter

        Returns:
            Methane enhancements (ppb)
        """

        # rows = self.radiance.sizes['y']
        # cols = self.radiance.sizes['x']
        # alpha = np.zeros((rows, 0), dtype=float)

        # # iterate across track direction
        # for ncol in range(cols):
        #     self._printt(str(ncol))
        #     # sel by col and sort dims for matched filter algo
        #     radiance_col = self.radiance.isel(x=ncol).transpose(..., 'bands')

        #     # calculate stats of data
        #     background = algo.calc_stats(radiance_col.data, mask=None, index=None)
        #     mu = background.mean

        #     # calculate the target spectrum
        #     target = self.K * mu

        #     # apply matched filter
        #     print(background.inv_cov)
        #     a = matched_filter(radiance_col.data, target, background)

        #     # concat data
        #     alpha = np.concatenate((alpha, a), axis=1)

        alpha = xr.apply_ufunc(self.col_matched_filter,
                               self.radiance.transpose(..., 'bands'),
                               kwargs={'K': self.K},
                               exclude_dims=set(('y', 'bands')),
                               input_core_dims=[['y', 'bands']],
                               output_core_dims=[['y', 'bands']],
                               vectorize=True,
                               dask='parallelized',
                               output_dtypes=[self.radiance.dtype],
                               dask_gufunc_kwargs=dict(output_sizes={'y': self.radiance.sizes['y'],
                                                                     'bands': 1,
                                                                     })
                               )

        # set the dims order to the same as radiance
        alpha = alpha.transpose(*self.radiance.dims)

        return alpha*SCALING

    # def ctmf(self, segmentation=None):
    #     """Cluster-tuned matched filter

    #     This method clusters similar pixels to improve the mean and cov calculation.
    #     Kwargs:
    #         seg_method (str): The mothod of segmentation which is useful for clustering pixels to calculate mean and cov.
    #                           Note this only works with the `CTMF` method.
    #                           Default: 'kmeans'
    #         bkg_extent (str): The extent for calculating background.
    #                           Note this only works with the `CTMF` method.
    #                           Default: 'column'
    #         A (float): Albedo (optional) which will be cancelled out.
    #                    Default: 1
    #     """
