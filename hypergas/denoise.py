#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Reduce the radom noise."""

import numpy as np
from skimage.restoration import denoise_tv_chambolle


class Denoise():
    """The Denoise Class."""

    def __init__(self, scene, varname, weight=50, method='tv_filter'):
        """Initialize Denoise.

        Args:
            data (DataArray): the data to to smoothed
            weight (int): the weight for denoise_tv_chambolle
        """
        self.data = scene[varname]
        self.weight = weight
        self.method = method

    def _split_data(self):
        """Split data in case of NaN rows"""
        # get index of nan values
        null_idx = np.where(self.data.squeeze().isnull().all('x'))[0]

        if len(null_idx) > 0:
            # calculate the difference
            bnd_idx = np.where(np.diff(null_idx) > 1)[0]

            # get the null index bound and next value
            split_idx = np.sort(np.concatenate((null_idx[bnd_idx], null_idx[bnd_idx+1]-1)))
            # add the index-1 of first nan value
            if null_idx[0] != 0:
                split_idx = np.insert(split_idx, 0, null_idx[0]-1)
            # add the index of last nan value
            split_idx = np.append(split_idx, null_idx[-1])
            # add first index
            split_idx = np.insert(split_idx, 0, 0)
            # add last index
            split_idx = np.append(split_idx, self.data.sizes['y']-1)

            self.split_idx = np.unique(split_idx)
        else:
            self.split_idx = None

    def smooth(self):
        """Smooth data."""
        if self.method == 'tv_filter':
            self._split_data()
            if self.split_idx is None:
                res = denoise_tv_chambolle(self.data, weight=self.weight)
            else:
                res = self.data.groupby_bins('y', self.split_idx, include_lowest=True)\
                    .map(denoise_tv_chambolle, weight=self.weight)

        else:
            raise ValueError(f'{self.method} is not supported yet.')

        # copy attrs
        res = res.rename(self.data.name+'_denoise')
        res.attrs = self.data.attrs

        description = f'denoised by the {self.method} method with weight={self.weight}'
        if 'description' in res.attrs:
            res.attrs['description'] = f"{res.attrs['description']} ({description})"

        return res
