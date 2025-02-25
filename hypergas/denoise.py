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

        Parameters:
        -----------
        data : xarray DataArray
            The data to to smoothed
        method : str
            The denoising method: "tv_filter" and "calibrated_tv_filter" (default)
        weight : int
            The weight for denoise_tv_chambolle.
            It would be neglected if method is 'calibrated_tv_filter'.
            If the weight is None (default), the denoise_tv_chambolle will use the default value (0.1) which is too low for HSI noisy gas field.
        """
        self.data = scene[varname]
        self.segmentation = scene['segmentation']
        self.weight = weight
        self.method = method

    def _create_mask_from_quantiles(self, image, lower_quantile=0.01, upper_quantile=0.99, min_cluster_size=10):
        """
        Create a mask based on quantile values to exclude isolated extreme values.

        Parameters:
        -----------
        image : 2D xarray DataArray
            Input image
        lower_quantile : float
            Lower quantile threshold (0-1)
        upper_quantile : float
            Upper quantile threshold (0-1)
        min_cluster_size : int
            Minimum size of connected clusters to retain (in pixels)

        Returns:
        --------
        mask : Masked 2D xarray DataArray
            Masked image with isolated outliers removed
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

    def tv_filter(self):
        """TV filter"""
        noisy = self.data.squeeze.where(self.segmentation > 0)
        trim_mean = trimmed_mean(noisy.stack(z=('y', 'x')).dropna('z'), (1e-3, 1e-3))
        res = denoise_tv_chambolle(np.ma.masked_array(np.where(noisy.isnull(), trim_mean, noisy), noisy.isnull()),
                                   weight=self.weight
                                   )

        return res

    def calibrated_tv_filter(self, n_weights=50, return_loss=False):
        """
        Apply TV filter with auto calibration


        Parameters:
        -----------
        n_weights : int
            Number of weights used for auto calibration
        return_loss: boolean
            Whether return the loss results

        Returns:
        --------
        denoised_calibrated_tv : 2D xarray DataArray
            Denoised gas field using calibrated params
        weights : 1D numpy array (if return_loss == True)
            The weights tested for calibration
        losses_tv : 1D numpy array (if return_loss == True)
            The losses of TV filter
        """
        # filter out pixels over water
        noisy = self.data.squeeze().where(self.segmentation > 0)

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
        """Smooth data."""
        if self.method == 'tv_filter':
            res = self.tv_filter()
        elif self.method == 'calibrated_tv_filter':
            res = self.calibrated_tv_filter()
        else:
            raise ValueError(f'{self.method} is not supported yet.')

        # create DataArray
        res = xr.DataArray(res, coords=self.data.squeeze().coords, dims=self.data.squeeze().dims)

        # copy attrs
        res = res.rename(self.data.name+'_denoise')
        res.attrs = self.data.attrs

        description = f'denoised by the {self.method} method with weight={self.weight}'
        if 'description' in res.attrs:
            res.attrs['description'] = f"{res.attrs['description']} ({description})"

        return res
