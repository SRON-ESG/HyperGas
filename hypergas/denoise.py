#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Reduce the radom noise."""

import logging

import numpy as np
import xarray as xr
from scipy.ndimage import label
from scipy.stats.mstats import trimmed_mean
from skimage.restoration import (calibrate_denoiser, denoise_invariant,
                                 denoise_tv_chambolle)

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)


class Denoise():
    """The Denoise Class."""

    def __init__(self, scene, varname, method='calibrated_tv_filter', weight=None):
        """Initialize Denoise.

        Parameters
        ----------
        scn : :class:`~satpy.Scene`
            Satpy Scene which includes a variable to be denoised.
        varname : str
            The variable to be denoised.
        method : str
            The denoising method: "tv_filter" and "calibrated_tv_filter" (default).
        weight : int
            The weight for denoise_tv_chambolle.
            It would be neglected if method is "calibrated_tv_filter".
            If the weight is ``None`` (default) and ``method`` is “tv_filter”, the denoise_tv_chambolle will use the default value (0.1) which is too low for hyperspectral noisy gas field.
        """
        self.data = scene[varname]
        self.segmentation = scene['segmentation']
        self.weight = weight
        self.method = method

    def _create_mask_from_quantiles(self, image, lower_quantile=0.01, upper_quantile=0.99, min_cluster_size=10):
        """
        Create a mask based on quantile values to exclude isolated extreme values.

        Parameters
        ----------
        image : :class:`~xarray.DataArray`
            Input 2D image.
        lower_quantile : float
            Lower quantile threshold (0-1).
        upper_quantile : float
            Upper quantile threshold (0-1).
        min_cluster_size : int
            Minimum size of connected clusters to retain (in pixels).

        Returns
        -------
        mask : :class:`~xarray.DataArray`
            Masked 2D image with isolated outliers removed.
        """
        # Compute quantile thresholds directly in xarray
        lower_thresh = image.quantile(lower_quantile)
        upper_thresh = image.quantile(upper_quantile)

        # Identify potential outliers using xarray operations
        outliers = (image < lower_thresh) | (image > upper_thresh)

        # Label connected components in the outlier mask
        labeled_clusters, num_features = label(outliers.values)  # Convert only for labeling step

        if num_features == 0:
            return image  # No outliers detected, return original image

        # Compute sizes of all clusters using np.bincount
        cluster_sizes = np.bincount(labeled_clusters.ravel())

        # Identify small clusters efficiently
        small_clusters = np.isin(labeled_clusters, np.where(cluster_sizes < min_cluster_size)[0])

        # Convert the small cluster mask back to xarray
        mask = xr.DataArray(~small_clusters, coords=image.coords, dims=image.dims)

        # Apply mask using xarray.where
        return image.where(mask)

    def _copy_attrs(self, res):
        """Copy data attributes to the denoised field"""
        # create DataArray
        res = xr.DataArray(res, coords=self.data.squeeze().coords, dims=self.data.squeeze().dims)

        # copy attrs
        res = res.rename(self.data.name+'_denoise')
        res.attrs = self.data.attrs

        description = f'denoised by the {self.method} method with weight={self.weight}'
        if 'description' in res.attrs:
            res.attrs['description'] = f"{res.attrs['description']} ({description})"

        return res

    def tv_filter(self):
        """Call TV filter"""
        noisy = self.data.squeeze().where(self.segmentation == self.seg_id)
        trim_mean = trimmed_mean(noisy.stack(z=('y', 'x')).dropna('z'), (1e-3, 1e-3))
        res = denoise_tv_chambolle(np.ma.masked_array(np.where(noisy.isnull(), trim_mean, noisy), noisy.isnull()),
                                   weight=self.weight
                                   )

        return res

    def calibrated_tv_filter(self, n_weights=50, return_loss=False):
        """
        Apply TV filter with `auto calibration <https://scikit-image.org/docs/0.25.x/auto_examples/filters/plot_j_invariant_tutorial.html>`_.

        Parameters
        ----------
        n_weights : int
            Number of weights used for auto calibration.
        return_loss : bool
            Whether return the loss results.

        Returns
        -------
        denoised_calibrated_tv : :class:`~xarray.DataArray`
            2D denoised data field using calibrated parameters.
        weights : :class:`numpy.ndarray`, optional
            1D array of weights tested for calibration. Returned only if ``return_loss == True``.
        losses_tv : :class:`numpy.ndarray`, optional
            1D array of total variation (TV) filter losses. Returned only if ``return_loss == True``.
        """
        noisy = self.data.squeeze().where(self.segmentation == self.seg_id)

        # remove highest and lowest value
        noisy_mask = self._create_mask_from_quantiles(noisy)
        m = noisy_mask.isnull()
        trim_mean = trimmed_mean(noisy_mask.stack(z=('y', 'x')).dropna('z'), (1e-3, 1e-3))
        noisy_mask = np.ma.masked_array(np.where(m, trim_mean, noisy_mask), m)
        noise_std = np.std(noisy_mask)
        weight_range = (noise_std/10, noise_std*3)
        weights = np.linspace(weight_range[0], weight_range[1], n_weights)

        parameter_ranges_tv = {'weight': weights}

        _, (parameters_tested_tv, losses_tv) = calibrate_denoiser(
            noisy_mask,
            denoise_tv_chambolle,
            denoise_parameters=parameter_ranges_tv,
            extra_output=True,
        )

        LOG.debug(f'Minimum self-supervised loss TV: {np.min(losses_tv):.3f}')

        best_parameters_tv = parameters_tested_tv[np.argmin(losses_tv)]
        LOG.debug(f'best_parameters_tv: {best_parameters_tv}')

        self.weight = np.round(best_parameters_tv['weight'], 1)

        denoised_calibrated_tv = denoise_invariant(
            np.ma.masked_array(np.where(noisy.isnull(), trim_mean, noisy), noisy.isnull()),
            denoise_tv_chambolle, denoiser_kwargs=best_parameters_tv,
        )

        if return_loss:
            return denoised_calibrated_tv, weights, losses_tv
        else:
            return denoised_calibrated_tv

    def smooth(self):
        """Smooth data by TV filter."""
        # create the empty list for denoised data
        res_list = []

        # denoising data by cluster
        for seg_id in np.unique(self.segmentation):
            LOG.info(f'Applying denoising to segmentation_id {seg_id} ...')
            self.seg_id = seg_id

            if self.method == 'tv_filter':
                res = self.tv_filter()
            elif self.method == 'calibrated_tv_filter':
                res = self.calibrated_tv_filter()
            else:
                raise ValueError(f'{self.method} is not supported yet.')

            # copy attributes
            res = self._copy_attrs(res)

            # set values only for seg_id
            res = res.where(self.segmentation == seg_id, 0)

            # append to list
            res_list.append(res)

        # aggregate all results into one DataArray
        res_sum = sum(res_list)

        # copy attributes
        res_sum = self._copy_attrs(res_sum)

        return res_sum
