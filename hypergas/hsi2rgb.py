#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Generate RGB image using hyperspectral satellite data."""

from bisect import bisect

import os
import numpy as np
import yaml
import scipy.io as spio
from scipy.interpolate import PchipInterpolator


def Hsi2rgb(wY, HSI, ydim, xdim, d, threshold):
    """Generate RGB from HSI image data using the `HSI2RGB method <https://doi.org/10.1109/IGARSS39084.2020.9323397>`_.

    Parameters
    ----------
    wY : :class:`numpy.ndarray`
        1D Band wavelengths (nm).
    HSI : :class:`numpy.ndarray`
        2D Radiance matrix (npixels, nbands), e.g., ``da.stack(z=['y', 'x']).transpose(..., 'bands')``.
    ydim: int
        The y dimension size of image.
    xdim: int
        The x dimension size of image.
    d : int
        The number (50, 55, 65, or 75) used to determine the illuminant used, if in doubt use d65.
    thresholdRGB : bool
        ``True`` if thesholding should be done to increase contrast.

    Returns
    -------
    rgb : :class:`numpy.ndarray`
        dims: (ydim, xdim, 3), where the last dimension corresponds to the color channels (R, G, B).
    """
    # load settings
    _dirname = os.path.dirname(__file__)
    with open(os.path.join(_dirname, 'config.yaml')) as f:
        settings = yaml.safe_load(f)['data']

    # slice data to VIS
    rgb_dir = os.path.join(_dirname, settings['rgb_dir'])

    # Load reference illuminant
    D = spio.loadmat(os.path.join(rgb_dir, 'D_illuminants.mat'))
    w = D['wxyz'][:, 0]
    x = D['wxyz'][:, 1]
    y = D['wxyz'][:, 2]
    z = D['wxyz'][:, 3]
    D = D['D']

    i = {50: 2,
         55: 3,
         65: 1,
         75: 4}
    wI = D[:, 0]
    I = D[:, i[d]]

    # Interpolate to image wavelengths
    I = PchipInterpolator(wI, I, extrapolate=True)(wY)  # interp1(wI,I,wY,'pchip','extrap')';
    x = PchipInterpolator(w, x, extrapolate=True)(wY)  # interp1(w,x,wY,'pchip','extrap')';
    y = PchipInterpolator(w, y, extrapolate=True)(wY)  # interp1(w,y,wY,'pchip','extrap')';
    z = PchipInterpolator(w, z, extrapolate=True)(wY)  # interp1(w,z,wY,'pchip','extrap')';

    # Truncate at 780nm
    i = bisect(wY, 780)
    HSI = HSI[:, 0:i]/HSI.max()
    wY = wY[:i]
    I = I[:i]
    x = x[:i]
    y = y[:i]
    z = z[:i]

    # Compute k
    k = 1/np.trapz(y * I, wY)

    # Compute X,Y & Z for image
    X = k * np.trapz(HSI @ np.diag(I * x), wY, axis=1)
    Z = k * np.trapz(HSI @ np.diag(I * z), wY, axis=1)
    Y = k * np.trapz(HSI @ np.diag(I * y), wY, axis=1)

    XYZ = np.array([X, Y, Z])

    # Convert to RGB
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    sRGB = M@XYZ

    # Gamma correction
    gamma_map = sRGB > 0.0031308
    sRGB[gamma_map] = 1.055 * np.power(sRGB[gamma_map], (1. / 2.4)) - 0.055
    sRGB[np.invert(gamma_map)] = 12.92 * sRGB[np.invert(gamma_map)]
    # Note: RL, GL or BL values less than 0 or greater than 1 are clipped to 0 and 1.
    sRGB[sRGB > 1] = 1
    sRGB[sRGB < 0] = 0

    if threshold:
        for idx in range(3):
            y = sRGB[idx, :]
            a, b = np.histogram(y, 100)
            b = b[:-1] + np.diff(b)/2
            a = np.cumsum(a)/np.sum(a)
            th = b[0]
            i = a < threshold
            if i.any():
                th = b[i][-1]
            y = y-th
            y[y < 0] = 0

            a, b = np.histogram(y, 100)
            b = b[:-1] + np.diff(b)/2
            a = np.cumsum(a)/np.sum(a)
            i = a > 1-threshold
            th = b[i][0]
            y[y > th] = th
            y = y/th
            sRGB[idx, :] = y

    R = np.reshape(sRGB[0, :], [ydim, xdim])
    G = np.reshape(sRGB[1, :], [ydim, xdim])
    B = np.reshape(sRGB[2, :], [ydim, xdim])

    return np.transpose(np.array([R, G, B]), [1, 2, 0])
