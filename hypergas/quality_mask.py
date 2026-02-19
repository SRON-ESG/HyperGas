#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Create a 2D quality mask for hyperspectral satellite data."""

import logging
import numpy as np
import xarray as xr

from .unit_spectrum import Unit_spec

LOG = logging.getLogger(__name__)


class QualityMask():
    """Quality mask for hyperspectral satellite scenes.

    Computes per-pixel boolean flags for water, cloud, and cirrus
    contamination based on top-of-atmosphere (TOA) reflectance thresholds,
    and combines them into a single ``qmask`` DataArray.

    Parameters
    ----------
    scn : :class:`~satpy.Scene`
        Satpy Scene that must contain the following datasets:

        * ``'radiance'`` – spectral radiance in W m-2 sr-1 um-1,
          with dimensions ``(bands, y, x)``.
        * ``'sza'``      – solar zenith angle in degrees, shape ``(y, x)``.

    Attributes
    ----------
    scn : :class:`~satpy.Scene`
        The input scene (stored for reference).
    rad : :class:`~xarray.DataArray`
        Radiance converted to uW cm-2 sr-1 nm-1 (divided by 10).
    sza : float
        Scene-mean solar zenith angle in radians.
    rho : :class:`~xarray.DataArray`
        TOA apparent reflectance, shape ``(bands, y, x)``.
    water : :class:`~xarray.DataArray`
        Boolean water mask, shape ``(y, x)``.
    cloud : :class:`~xarray.DataArray`
        Boolean cloud mask, shape ``(y, x)``.
    cirrus : :class:`~xarray.DataArray`
        Boolean cirrus mask, shape ``(y, x)``.
    qmask : :class:`~xarray.DataArray`
        Combined quality mask with a ``quality_flag`` dimension whose
        labels are ``['water', 'cloud', 'cirrus', 'invalid']``.
    """

    def __init__(self, scn):
        # Store scene and convert radiance units:
        # W m-2 sr-1 um-1 -> uW cm-2 sr-1 nm-1  (factor = 1/10)
        self.scn = scn
        self.rad = scn['radiance'] / 10

        # Scene-mean solar zenith angle (degrees -> radians)
        self.sza = np.deg2rad(float(scn['sza'].mean()))

        # Pre-compute TOA reflectance used by all mask methods
        self._toa()

    def _toa(self):
        """Compute top-of-atmosphere (apparent) reflectance.

        The TOA reflectance is defined as:

        .. math::

            \\rho(\\lambda) =
                \\frac{\\pi \\cdot L(\\lambda)}{E_0(\\lambda) \\cdot \\cos(\\theta_s)}

        where :math:`L` is the at-sensor radiance (uW cm-2 sr-1 nm-1),
        :math:`E_0` is the extraterrestrial solar irradiance convolved to
        the sensor's spectral response function (uW cm-2 nm-1), and
        :math:`\\theta_s` is the solar zenith angle.

        The result is stored in ``self.rho`` with shape ``(bands, y, x)``.
        """
        # Build a Unit_spec object to access the solar irradiance spectrum
        unit = Unit_spec(
            self.scn['radiance'],
            self.scn['radiance'].coords['bands'],
            self.scn['radiance'].coords['bands'].min(),
            self.scn['radiance'].coords['bands'].max(),
        )
        irr = unit.solar_irradiance

        # convert to uW cm-2 nm-1
        irr /= 10
        irr.attrs['units'] = 'uW cm-2 nm-1'

        # Convolve the high-resolution solar spectrum with the sensor SRF
        irr_resampled = unit._convolve(
            unit.wvl_sensor,
            unit.fwhm_sensor,
            irr.coords['wavelength'].values,
            irr.values,
        )

        # Compute TOA reflectance: ρ = (π · L) / (E₀ · cos θ_s)
        # Transpose operations keep xarray dimension alignment correct
        # Keep rho as a lazy dask array — don't load the full cube here
        irr_resampled = xr.DataArray(
            irr_resampled,
            dims=["bands"],
            coords={"bands": self.rad["bands"]}
        )

        self.rho = (np.pi * self.rad) / (irr_resampled * np.cos(self.sza))

    def mask(self):
        """Compute all individual masks and combine into ``self.qmask``.

        Calls :meth:`water_mask`, :meth:`cloud_mask`, and
        :meth:`cirrus_mask` in sequence, then concatenates the results
        along a new ``quality_flag`` dimension.  An additional
        ``'invalid'`` flag is appended that is ``True`` wherever *any*
        of the three masks is ``True``.

        After calling this method the combined mask is available as
        ``self.qmask`` with shape ``(quality_flag, y, x)`` and
        ``quality_flag`` labels ``['water', 'cloud', 'cirrus', 'invalid']``.
        """
        LOG.info('Generating quality masks using TOA ...')

        # Load all required bands in ONE dask compute instead of 5 separate ones
        bands_needed = [450, 1000, 1250, 1380, 1650]
        rho_subset = self.rho.sel(bands=bands_needed, method='nearest').load()

        self.water_mask(rho_subset)
        self.cloud_mask(rho_subset)
        self.cirrus_mask(rho_subset)

        # Drop any scalar/spectral coordinates inherited from .sel(bands=...)
        # (e.g. 'bands', 'fwhm', 'wavelength') so that xr.concat finds a
        # consistent set of coordinates across all three masks.
        spatial_masks = [
            m.drop_vars([c for c in m.coords if c not in m.dims])
            for m in (self.water, self.cloud, self.cirrus)
        ]

        # Stack the three boolean masks along a new 'quality_flag' dimension
        qmask = xr.concat(
            spatial_masks,
            dim=xr.DataArray(
                ['water', 'cloud', 'cirrus'],
                dims='quality_flag',
                name='quality_flag',
            ),
        )

        # Derive a single 'invalid' flag: True if any individual flag is True
        any_mask = qmask.any(dim='quality_flag')
        any_mask = any_mask.expand_dims(quality_flag=['invalid'])

        # Append 'invalid' to produce the final 4-flag mask
        qmask = xr.concat([qmask, any_mask], dim='quality_flag').astype(float)
        self.qmask = qmask.rename('quality_mask')

    def water_mask(self, rho=None):
        """Identify water pixels using TOA reflectance at 1000 nm.

        A pixel is flagged as water when its near-infrared reflectance
        falls below 0.05, exploiting the strong absorption of liquid
        water beyond 900 nm.

        Sets ``self.water`` to a boolean :class:`~xarray.DataArray`
        of shape ``(y, x)``.
        """
        rho = rho if rho is not None else self.rho.load()
        rho_1000 = rho.sel(bands=1000, method='nearest')
        self.water = rho_1000 < 0.05

    def cloud_mask(self, rho=None):
        """Identify cloud pixels using multi-band TOA reflectance thresholds.

        Three independent reflectance tests are applied:

        * 450 nm  > 0.28  (high visible reflectance)
        * 1250 nm > 0.46  (high short-wave infrared reflectance)
        * 1650 nm > 0.22  (high short-wave infrared reflectance)

        A pixel is flagged as cloudy only when **all three** conditions
        are satisfied (majority vote ≥ 3 out of 3), reducing false
        positives over bright land surfaces.

        Sets ``self.cloud`` to a boolean :class:`~xarray.DataArray`
        of shape ``(y, x)``.

        References
        ----------
        Sandford et al., *AMT*, 13, 7047–7057, 2020.
        https://doi.org/10.5194/amt-13-7047-2020
        """
        rho = rho if rho is not None else self.rho.load()
        rho_450 = rho.sel(bands=450,  method='nearest')
        rho_1250 = rho.sel(bands=1250, method='nearest')
        rho_1650 = rho.sel(bands=1650, method='nearest')

        # Cast each threshold test to int so they can be summed
        self.cloud = (
            (rho_450 > 0.28).astype(int)
            + (rho_1250 > 0.46).astype(int)
            + (rho_1650 > 0.22).astype(int)
        ) >= 3

    def cirrus_mask(self, rho=None):
        """Identify cirrus cloud pixels using TOA reflectance at 1380 nm.

        The 1380 nm water-vapour absorption band is used as a cirrus
        proxy: surface-leaving radiance is almost entirely absorbed by
        atmospheric water vapour at this wavelength, so any residual
        reflectance above the threshold is attributed to high-altitude
        cirrus ice clouds.

        Sets ``self.cirrus`` to a boolean :class:`~xarray.DataArray`
        of shape ``(y, x)``.

        References
        ----------
        Gao & Goetz, *GRL*, 20(4), 301–304, 1993.
        https://doi.org/10.1029/93GL00106
        """
        rho = rho if rho is not None else self.rho.load()
        rho_1380 = rho.sel(bands=1380, method='nearest')
        self.cirrus = rho_1380 > 0.1
