#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Hyperspectral image classification."""

import xarray as xr
from sklearn.decomposition import PCA
from spectral import kmeans


def PCA_kmeans(radiance, pca_dim=3, ncluster=10, max_iterations=30):
    """Apply PCA and kmeans to classify pixels.

    Parameters
    ----------
    radiance : :class:`~xarray.DataArray`
        dims: y, x, bands.
    pca_dim : int
        n_components for PCA.
    ncluster : int
        number of clusters to create.
    max_iterations : int
        max number of iterations to perform.

    Returns
    ------
    segmentation : :class:`~xarray.DataArray`
        Segmentation result with dimensions (y, x).
    """
    # initialize PCA model
    pca = PCA(n_components=pca_dim)

    # fit model to data
    radiance_stack = radiance.stack(z=['y', 'x']).transpose()
    pca.fit(radiance_stack)  # input 2D array (X, band)

    # access projections into principal components
    radiance_pca = pca.transform(radiance_stack)
    radiance_pca = radiance_pca.reshape(radiance.isel(bands=slice(0, pca_dim)).transpose(..., 'bands').shape)

    # apply kmeans
    # https://www.spectralpython.net/class_func_ref.html#spectral.kmeans
    segmentation = kmeans(radiance_pca, nclusters=ncluster, max_iterations=max_iterations, distance='L1')[0]

    # save to DataArray
    segmentation = xr.DataArray(segmentation, dims=['y', 'x']).astype('float32')
    segmentation.attrs['description'] = 'Kmeans cluster'

    return segmentation
