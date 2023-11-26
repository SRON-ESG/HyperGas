#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Retrieve methane enhancements using hyperspectral satellite data."""

import sys

import numpy as np
import spectral.algorithms as algo
import xarray as xr
from roaring_landmask import RoaringLandmask
from spectral.algorithms.detectors import matched_filter

from .unit_spectrum import Unit_spec

# the scaling factor for alpha
#   this should be set as same as that in `unit_spectrum.py`
SCALING = 1e5


class MatchedFilter():
    """The MatchedFilter Class."""

    def __init__(self, radiance, wvl_intervals, species='ch4',
                 fit_unit='lognormal', mode='column', land_mask=True):
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
            species (str): The species to be retrieved
                'ch4' or 'co2'
                Default: 'ch4'
            fit_unit (str): The fitting method ('lognormal', 'poly', or 'linear') to calculate the unit CH4 spectrum
                            Default: 'lognormal'
            mode (str): the mode ("column" or "scene") to apply matched filter.
                        Default: 'column'.
        """
        # set the wavelength range for matched filter
        self.wvl_min = wvl_intervals[0]
        self.wvl_max = wvl_intervals[1]

        # subset data to selected wavelength range for matched filter
        wvl_mask = (radiance['bands'] >= self.wvl_min) & (radiance['bands'] <= self.wvl_max)
        radiance = radiance.where(wvl_mask, drop=True)

        self.radiance = radiance
        self.mode = mode

        # calculate unit spectrum
        self.species = species
        self.fit_unit = fit_unit

        self.K = Unit_spec(self.radiance, self.wvl_min, self.wvl_max, self.species, self.fit_unit).fit_slope()

        # calculate the land/ocean segmentation
        self.land_mask = land_mask

        if land_mask:
            self.land_segmentation()
        else:
            # set all pixels as the same type
            self.segmentation = xr.DataArray(np.zeros((self.radiance.sizes['y'],
                                                      self.radiance.sizes['x'])),
                                             dims=['y', 'x'])

    def _printt(self, outm):
        """Refreshing print."""
        sys.stdout.write("\r" + outm)
        sys.stdout.flush()

    def land_segmentation(self):
        """Create the segmentation for land and ocean types"""
        # 0: ocean, 1: land
        roaring = RoaringLandmask.new()
        lons, lats = self.radiance.attrs['area'].get_lonlats()
        landmask = roaring.contains_many(lons.ravel().astype('float64'), lats.ravel().astype('float64')).reshape(lons.shape)
        # save to DataArray
        self.segmentation = xr.DataArray(landmask, dims=['y', 'x'])

    # def _norm(self):
    #         from sklearn.preprocessing import MinMaxScaler
    #         data = data.reshape((-1, data.shape[-1]))
    #         self.radiance = self.radiance.stack(z=('y', 'x'))
    #         scaler = MinMaxScaler()
    #         scaler.fit(self.radiance)
    #     return scaler.transform(data)


    def col_matched_filter(self, radiance, segmentation, K):
        """Calculate stats of data."""
        if self.mode == 'column':
            # create empty alpha with shape: [nrows('y'), 1]
            alpha = np.full((radiance.shape[0], 1), fill_value=np.nan, dtype=float)

            # iterate unique label to apply the matched filter
            for label in np.unique(segmentation):
                # create nan*label mask
                segmentation_mask = segmentation == label
                mask = ~np.isnan(radiance).any(axis=-1)
                mask = mask*segmentation_mask

                # calculate the background stats if there're many valid values
                if mask.sum() > 1:
                    background = algo.calc_stats(radiance, mask=mask, index=None, allow_nan=True)

                    # get mean value
                    mu = background.mean

                    # calculate the target spectrum
                    target = K * mu

                    # apply the matched filter
                    a = matched_filter(radiance, target, background)

                    # concat data
                    alpha[:, 0][mask] = a[:, 0][mask]

        elif self.mode == 'scene':
            background = self.background
            # get mean value
            mu = background.mean

            # calculate the target spectrum
            target = K * mu

            # apply matched filter
            alpha = matched_filter(radiance, target, background)

        else:
            raise ValueError(f'Wrong mode: {self.mode}. It should be "column" or "scene".')

        return alpha

    def smf(self):
        """Standard/Robust matched filter

            Compute mean and covariance of set of each column and then run standard matched filter

        Returns:
            Methane enhancements (ppb)
        """
        if self.mode == 'scene':
            # calculate the background of whole scene
            radiance_scene = self.radiance.transpose(..., 'bands').values
            mask_scene = ~np.isnan(radiance_scene).any(axis=-1)
            if self.land_mask:
                # only calculate background over land
                mask = mask_scene * self.segmentation.data
            else:
                mask = mask_scene
            self.background = algo.calc_stats(radiance_scene, mask=mask, index=None)

        alpha = xr.apply_ufunc(self.col_matched_filter,
                               self.radiance.transpose(..., 'bands'),
                               self.segmentation,
                               kwargs={'K': self.K},
                               exclude_dims=set(('y', 'bands')),
                               input_core_dims=[['y', 'bands'], ['y']],
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

        # fill nan values by interpolation
        #   this usually happens for pixels with only one segmentation labels in a specific column.
        #   if data is not available for the whole row, then the row values should still be nan.
        alpha = alpha.interpolate_na(dim='y', method='linear')

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
