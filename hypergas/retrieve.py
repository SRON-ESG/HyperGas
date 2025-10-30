#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Retrieve trace gas enhancements using hyperspectral satellite data."""

import logging
import numpy as np
import spectral.algorithms as algo
import xarray as xr
from spectral.algorithms.detectors import matched_filter

from .landmask import Land_mask
from .cluster import PCA_kmeans
from .unit_spectrum import Unit_spec

LOG = logging.getLogger(__name__)


class MatchedFilter():
    """The MatchedFilter Class."""

    def __init__(self, scn, wvl_intervals, species='ch4',
                 mode='column', rad_dist='normal', rad_source='model',
                 land_mask=True, land_mask_source='OSM', cluster=False,
                 plume_mask=None, scaling=None,
                 ):
        """Initialize MatchedFilter.

        Parameters
        ----------
        scn : Satpy Scene
            including at least one variable named "radiance"
            (:class:`~xarray.Dataset` ["bands", "y", "x"], units: mW m^-2 sr^-1 nm^-1),
            and at least two coordinates: [1] "wavelength" (nm) and [2] "fwhm" (nm).
        wvl_intervals : list
            The wavelength range (nm) used in matched filter. It can be one list or nested list.
            e.g. ``[2110, 2450]`` or ``[[1600, 1750], [2110, 2450]]``.
        species : str
            The species to be retrieved. Only species defined in the ``config.yaml`` file.
            Default: "ch4".
        mode : str
            The mode ("column" or "scene") to apply matched filter.
            Default: "column". Be careful of noise if you apply the matched filter for the whole scene.
        rad_dist : str
            The assumed rads distribution ("normal" or "lognormal").
            Default: "normal".
        rad_source : str
            The data ('model' or 'lut') used for calculating rads or transmissions.
            Default: "model".
        land_mask : bool
            Whether apply the matched filter to continental and oceanic pixels seperately.
            Default: True.
        land_mask_source : str
            The data source of land mask ("OSM", "GSHHS", or "Natural Earth").
            Default: "OSM".
        cluster : bool
            Whether apply the pixel classification.
            Default: False.
        plume_mask : :class:`numpy.ndarray`
            2D array, 0: neglected pixels, 1: plume pixels.
            Default: None.
        scaling : float
            The scaling factor for alpha to ensure numerical stability.
        """
        # set the wavelength range for matched filter
        self.wvl_min = wvl_intervals[0]
        self.wvl_max = wvl_intervals[1]

        # subset data to selected wavelength range for matched filter
        radiance = scn['radiance']
        wvl_mask = (radiance['bands'] >= self.wvl_min) & (radiance['bands'] <= self.wvl_max)
        radiance = radiance.where(wvl_mask, drop=True)

        # add to the class
        self.radiance = radiance
        self.mode = mode
        self.rad_dist = rad_dist
        self.rad_source = rad_source
        self.species = species

        # calculate unit spectrum
        loaded_names = [x['name'] for x in scn.keys()]
        if 'central_wavelengths' in loaded_names:
            # calculate K by column because we have central wavelength per column
            central_wavelengths = scn['central_wavelengths'].where(wvl_mask, drop=True)
            K, self.scaling = Unit_spec(self.radiance, central_wavelengths,
                          self.wvl_min, self.wvl_max, self.species, self.rad_source).fit_slope(scaling=scaling)
            self.K = xr.DataArray(K, dims=['bands', 'x'],
                                  coords={'bands': self.radiance.coords['bands']})
        else:
            # calculate K using the default dim named 'bands'
            K, self.scaling = Unit_spec(self.radiance, self.radiance.coords['bands'],
                          self.wvl_min, self.wvl_max, self.species, self.rad_source).fit_slope(scaling=scaling)
            self.K = xr.DataArray(K, dims='bands', coords={'bands': self.radiance.coords['bands']})

        # calculate the land/ocean segmentation
        self.land_mask = land_mask

        if cluster:
            segmentation = PCA_kmeans(self.radiance)
        elif land_mask:
            # get lon and lat from area attrs
            lons, lats = self.radiance.attrs['area'].get_lonlats()
            # create land mask from 10-m Natural Earth data
            segmentation = Land_mask(lons, lats, land_mask_source)
        else:
            # set all pixels as the same type
            segmentation = xr.DataArray(np.ones((self.radiance.sizes['y'],
                                                 self.radiance.sizes['x'])),
                                        dims=['y', 'x'],
                                        )
            segmentation.attrs['description'] = ''

        # save to class
        self.segmentation = segmentation

        # save plume mask
        if plume_mask is None:
            # set all pixels as non-plume
            self.plume_mask = xr.DataArray(np.ones((self.radiance.sizes['y'],
                                                    self.radiance.sizes['x'])),
                                           dims=['y', 'x'],
                                           )
        else:
            self.plume_mask = plume_mask

    # def _norm(self):
    #         from sklearn.preprocessing import MinMaxScaler
    #         data = data.reshape((-1, data.shape[-1]))
    #         self.radiance = self.radiance.stack(z=('y', 'x'))
    #         scaler = MinMaxScaler()
    #         scaler.fit(self.radiance)
    #     return scaler.transform(data)

    def col_matched_filter(self, radiance, segmentation, plume_mask, K):
        """Apply the matched filter by column.

        Parameters
        ----------
        radiance : :class:`~xarray.DataArray`
            The radiance DataArray for one column.
        segmentation : :class:`~xarray.DataArray`, same shape as ``radiance``
            The segmentation of pixels (e.g., land and water mask).
        plume_mask : :class:`~xarray.DataArray`, same shape as ``radiance``
            Since the matched filter assumes plume signals are sparse (i.e., present in only a small fraction of pixels),
            it is better to exclude pixels within identified plume masks,
            so that background statistics are estimated only from non-plume pixels and the sparsity assumption remains valid.
        K : :class:`numpy.ndarray`
            The Jacobian K (i.e, the change of the radiance (or its logarithm) for a +1 ppm methane concentration increase).

        Returns
        -------
        The gas enhancement : :class:`~xarray.DataArray`
            Unit: ppm.
        """
        if self.mode == 'column':
            # create empty alpha with shape: [nrows('y'), 1]
            alpha = np.full((radiance.shape[0], 1), fill_value=np.nan, dtype=float)

            # iterate unique label to apply the matched filter
            for label in np.unique(segmentation):
                # create nan*label mask
                segmentation_mask = segmentation == label
                mask = ~np.isnan(radiance).any(axis=-1)
                mask = mask * segmentation_mask
                # we need to create new mask with plume instead of overwrite
                #   because we want to keep retrieval results over plume pixels
                mask_exclude_plume = mask * plume_mask.astype(bool)

                # calculate the background stats if there're many valid values
                if mask_exclude_plume.sum() > 1:
                    if self.rad_dist == 'lognormal':
                        # calculate lognormal rads
                        lograds = np.log(radiance, out=np.zeros_like(radiance), where=radiance > 0)
                        background = algo.calc_stats(lograds, mask=mask_exclude_plume, index=None, allow_nan=True)

                        # apply the matched filter
                        a = matched_filter(lograds, K, background)
                    elif self.rad_dist == 'normal':
                        # linearized MF
                        if sum(mask_exclude_plume) > 1:
                            background = algo.calc_stats(radiance, mask=mask_exclude_plume, index=None, allow_nan=True)
                        else:
                            # if all pixels are masked, we use the default mask
                            background = algo.calc_stats(radiance, mask=mask, index=None, allow_nan=True)

                        # get mean value
                        mu = background.mean

                        # calculate the target spectrum
                        target = K * mu

                        # apply the matched filter
                        a = matched_filter(radiance, target, background)
                    else:
                        raise ValueError(f"{self.rad_dist} is not supported. Please use 'normal' or 'lognormal' as rad_dist.")

                    # concat data
                    alpha[:, 0][mask] = a[:, 0][mask]
                else:
                    # assign 0 value if only one pixel is available
                    #   because denoising data with one nan pixel will cause a large nan area
                    alpha[:, 0][mask] = 0

        elif self.mode == 'scene':
            # this function is experimental
            LOG.warning('The scene MF is experimental!!!')
            background = self.background

            # get mean value
            mu = background.mean

            # calculate the target spectrum
            target = K * mu

            # apply matched filter
            alpha = matched_filter(radiance, target, background)

        else:
            raise ValueError(f'Wrong mode: {self.mode}. It should be "column" or "scene".')

        if self.rad_source == 'model':
            # ppm
            return alpha
        elif self.rad_source == 'lut':
            # ppm m -> ppm assuming a scale height of about 8km
            return alpha * 1.25e-4

    def smf(self):
        """Standard/Robust matched filter.

        Compute mean and covariance of set of each column and then run standard matched filter.

        Returns
        -------
        The gas enhancement : :class:`~xarray.DataArray`
            Unit: ppm.
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

        LOG.info('Applying matched filter ...')
        alpha = xr.apply_ufunc(self.col_matched_filter,
                               self.radiance.transpose(..., 'bands'),
                               self.segmentation,
                               self.plume_mask,
                               self.K,
                               exclude_dims=set(('y', 'bands')),
                               input_core_dims=[['y', 'bands'], ['y'], ['y'], ['bands']],
                               output_core_dims=[['y', 'bands']],
                               vectorize=True,
                               dask='parallelized',
                               output_dtypes=[self.radiance.dtype],
                               dask_gufunc_kwargs=dict(output_sizes={'y': self.radiance.sizes['y'],
                                                                     'bands': 1,
                                                                     },
                                                       allow_rechunk=True,
                                                       )
                               )

        # set the dims order to the same as radiance
        alpha = alpha.transpose(*self.radiance.dims)

        # fill nan values by interpolation
        #   this usually happens for pixels with only one segmentation labels in a specific column.
        #   if data is not available for the whole row, then the row values should still be nan.
        alpha = alpha.interpolate_na(dim='y', method='linear')

        return alpha*self.scaling

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
    #     """
