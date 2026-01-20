#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Calculate 2D angles for hyperspectral satellite observations."""

import numpy as np
import xarray as xr
from typing import Mapping, Tuple

from pyorbital import orbital
from pyorbital.astronomy import sun_zenith_angle, get_alt_az


def compute_raa(saa: xr.DataArray, vaa: xr.DataArray) -> xr.DataArray:
    """
    Compute relative azimuth angle (RAA).

    RAA is the absolute difference between solar and viewing azimuth,
    constrained to [0, 180] degrees.

    Parameters
    ----------
    saa : xr.DataArray (nrows, ncols)
        Solar azimuth angle [degrees]
    vaa : xr.DataArray (nrows, ncols)
        Viewing azimuth angle [degrees]

    Returns
    -------
    raa : xr.DataArray (nrows, ncols)
        Relative azimuth angle [degrees]
    """
    raa = np.abs(vaa - saa)
    raa = np.minimum(raa, 360 - raa)

    raa_da = create_angle_dataarray(raa, 'raa')

    return raa_da


def compute_sga(sza: xr.DataArray, saa: xr.DataArray,
                vza: xr.DataArray, vaa: xr.DataArray) -> xr.DataArray:
    """
    Compute sun glint angle (SGA).

    SGA is the angle between the sun's reflection vector and the
    viewing direction, relevant for water surface reflectance.

    Parameters
    ----------
    sza : xr.DataArray (nrows, ncols)
        Solar zenith angle [degrees]
    saa : xr.DataArray (nrows, ncols)
        Solar azimuth angle [degrees]
    vza : xr.DataArray (nrows, ncols)
        Viewing zenith angle [degrees]
    vaa : xr.DataArray (nrows, ncols)
        Viewing azimuth angle [degrees]

    Returns
    -------
    sga : xr.DataArray (nrows, ncols)
        Sun glint angle [degrees]
    """
    # Convert to radians
    sza_rad = np.deg2rad(sza)
    saa_rad = np.deg2rad(saa)
    vza_rad = np.deg2rad(vza)
    vaa_rad = np.deg2rad(vaa)

    # Compute relative azimuth
    phi = saa_rad - vaa_rad

    # Glint angle formula
    cos_sga = (np.cos(vza_rad) * np.cos(sza_rad) -
               np.sin(vza_rad) * np.sin(sza_rad) * np.cos(phi))

    sga = np.rad2deg(np.arccos(np.clip(cos_sga, -1, 1)))
    sga_da = create_angle_dataarray(sga, 'sga')

    return sga_da


def create_angle_dataarray(data: np.ndarray, name: str) -> xr.DataArray:
    """
    Create an xarray.DataArray with proper dimensions and coordinates.

    Parameters
    ----------
    data : np.ndarray
        2D angle data
    name : str
        Variable name (e.g., 'sza', 'saa', 'vza', 'vaa', 'raa', 'sga')

    Returns
    -------
    da : xr.DataArray
        DataArray with dimensions and CF-compliant attributes
    """
    # Define complete metadata for each angle type
    angle_metadata = {
        'sza': {
            'standard_name': 'solar_zenith_angle',
            'long_name': 'To-sun zenith (0 to 90 degrees from zenith)',
            'units': 'degrees',
            'description': 'Angle between the local vertical and the line to the sun, measured from zenith (0 degree) to horizon (90 degree)'
        },
        'saa': {
            'standard_name': 'solar_azimuth_angle',
            'long_name': 'To-sun azimuth (0 to 360 degrees CW from N)',
            'units': 'degrees',
            'description': 'Azimuth angle of the sun measured clockwise from north (0-360 degrees)'
        },
        'vza': {
            'standard_name': 'viewing_zenith_angle',
            'long_name': 'To-sensor zenith (0 to 90 degrees from zenith)',
            'units': 'degrees',
            'description': 'Angle between the local vertical and the line to the sensor, measured from zenith (0 degree) to horizon (90 degree)'
        },
        'vaa': {
            'standard_name': 'viewing_azimuth_angle',
            'long_name': 'To-sensor azimuth (0 to 360 degrees CW from N)',
            'units': 'degrees',
            'description': 'Azimuth angle of the sensor measured clockwise from north (0-360 degrees)'
        },
        'raa': {
            'standard_name': 'relative_azimuth_angle',
            'long_name': 'Relative Azimuth Angle',
            'units': 'degrees',
            'description': 'Relative azimuth angle between sun and sensor, calculated as the absolute difference between solar and viewing azimuth angles (0-180 degrees)'
        },
        'sga': {
            'standard_name': 'sun_glint_angle',
            'long_name': 'Sun Glint Angle',
            'units': 'degrees',
            'description': 'Angle between the specular reflection direction and the sensor viewing direction, used to assess potential sun glint contamination in water observations'
        }
    }

    # Get metadata for this variable (case-insensitive lookup)
    var_key = name.lower()
    if var_key in angle_metadata:
        attrs = angle_metadata[var_key]
    else:
        # Fallback for other variables
        attrs = {
            'standard_name': name.lower(),
            'long_name': name.upper(),
            'units': 'unknown',
            'description': f'{name.upper()} variable'
        }

    da = xr.DataArray(
        data,
        dims=('y', 'x'),
        name=name,
        attrs=attrs
    )
    return da


class Angle2D:
    """
    Calculate 2D angle fields (SZA, SAA, VZA, VAA, RAA, SGA) for satellite observations.

    This class computes solar and viewing geometry angles by:
    1. Extracting corner and center points from 2D lon/lat grids
    2. Computing angles at these key points
    3. Interpolating to create full 2D fields
    """

    def __init__(
        self,
        start_time,
        end_time,
        lons: np.ndarray,
        lats: np.ndarray,
        tle1: str,
        tle2: str,
    ):
        """
        Initialize Angle2D calculator.

        Parameters
        ----------
        start_time : datetime
            Acquisition start time
        end_time : datetime
            Acquisition end time
        lons : numpy.ndarray (nrows, ncols)
            Longitude grid [degrees]
        lats : numpy.ndarray (nrows, ncols)
            Latitude grid [degrees]
        tle1 : str
            TLE line 1 for satellite orbit
        tle2 : str
            TLE line 2 for satellite orbit
        """
        # Validate input types
        if not isinstance(lons, np.ndarray):
            raise TypeError("lons must be a numpy.ndarray")
        if not isinstance(lats, np.ndarray):
            raise TypeError("lats must be a numpy.ndarray")

        # Validate shapes match
        if lons.shape != lats.shape:
            raise ValueError(f"lons and lats must have the same shape. Got {lons.shape} and {lats.shape}")

        # Validate TLE data provided
        if not tle1 or not tle2:
            raise ValueError("Both tle1 and tle2 are required for angle calculations")

        self.start_time = start_time
        self.end_time = end_time
        self.t_center = start_time + (end_time - start_time) / 2

        self.nrows, self.ncols = lons.shape

        # Extract corner and center coordinates (keep as xarray)
        self.lon_pts = self._corners_and_center(lons)
        self.lat_pts = self._corners_and_center(lats)

        # Time at each corner and center
        self.corner_times = {
            "UL": start_time,
            "UR": start_time,
            "LL": end_time,
            "LR": end_time,
            "C": self.t_center,
        }

        # Initialize orbit
        self.orbit = orbital.Orbital('instrument_name', line1=tle1, line2=tle2)

    @staticmethod
    def _corners_and_center(arr: xr.DataArray) -> dict:
        """
        Extract corner and center values from xarray DataArray.

        Parameters
        ----------
        arr : xr.DataArray
            2D array with dimensions 'y' and 'x'

        Returns
        -------
        dict
            Dictionary with keys 'UL', 'UR', 'LL', 'LR', 'C' containing scalar values
        """
        n_rows, n_cols = arr.shape

        return {
            "UL": arr[0, 0],
            "UR": arr[0, n_cols - 1],
            "LL": arr[n_rows - 1, 0],
            "LR": arr[n_rows - 1, n_cols - 1],
            "C": arr[n_rows // 2, n_cols // 2],
        }

    @staticmethod
    def _interp_angle(
        nrows: int,
        ncols: int,
        corners: Mapping[str, float],
    ) -> np.ndarray:
        """
        Bilinear interpolation with center constraint.

        Parameters
        ----------
        nrows, ncols : int
            Output dimensions
        corners : mapping
            Corner angles with keys: UL, UR, LL, LR, C

        Returns
        -------
        angle : np.ndarray (nrows, ncols)
            Interpolated angle field [degrees]
        """
        required = {"UL", "UR", "LL", "LR", "C"}
        missing = required - corners.keys()
        if missing:
            raise ValueError(f"Missing corner keys: {missing}")

        ul = corners["UL"]
        ur = corners["UR"]
        ll = corners["LL"]
        lr = corners["LR"]
        center = corners["C"]

        # Normalized coordinates
        y = np.linspace(0.0, 1.0, nrows, dtype=np.float64)
        x = np.linspace(0.0, 1.0, ncols, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        # Bilinear interpolation from corners
        angle = (
            ul * (1 - xx) * (1 - yy)
            + ur * xx * (1 - yy)
            + ll * (1 - xx) * yy
            + lr * xx * yy
        )

        # Apply center constraint
        center_bilin = 0.25 * (ul + ur + ll + lr)
        delta = center - center_bilin

        # Weight: zero at edges, one at center
        w = 4.0 * xx * (1 - xx) * yy * (1 - yy)

        return angle + delta * w

    def _compute_solar_angles(self) -> Tuple[dict, dict]:
        """Compute SZA and SAA at corner and center points."""
        sza = {}
        saa = {}

        for key, time in self.corner_times.items():
            # Sun zenith angle
            sza[key] = sun_zenith_angle(time, self.lon_pts[key], self.lat_pts[key])

            # Sun azimuth angle
            _, az = get_alt_az(time, self.lon_pts[key], self.lat_pts[key])
            # saa[key] = np.rad2deg(az)
            # test
            saa[key] = np.rad2deg(az) % 360  # Converts [-180, 180] to [0, 360]

        return sza, saa

    def _compute_viewing_angles(self) -> Tuple[dict, dict]:
        """Compute VZA and VAA at corner and center points."""
        vza = {}
        vaa = {}

        for key, time in self.corner_times.items():
            # Get sensor azimuth and elevation
            azimuth, elevation = self.orbit.get_observer_look(
                time,
                lon=np.array([self.lon_pts[key]]),
                lat=np.array([self.lat_pts[key]]),
                alt=np.array([0])
            )

            # Convert elevation to zenith angle
            vza[key] = 90 - elevation
            vaa[key] = azimuth

        return vza, vaa

    def compute_sza_saa(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute 2D solar zenith and azimuth angle fields.

        Returns
        -------
        sza : xr.DataArray (nrows, ncols)
            Solar zenith angle [degrees]
        saa : xr.DataArray (nrows, ncols)
            Solar azimuth angle [degrees]
        """
        sza, saa = self._compute_solar_angles()

        sza_2d = self._interp_angle(self.nrows, self.ncols, sza)
        saa_2d = self._interp_angle(self.nrows, self.ncols, saa)

        sza_da = create_angle_dataarray(sza_2d, 'sza')
        saa_da = create_angle_dataarray(saa_2d, 'saa')

        return sza_da, saa_da

    def compute_vza_vaa(self) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute 2D viewing zenith and azimuth angle fields.

        Returns
        -------
        vza : xr.DataArray (nrows, ncols)
            Viewing zenith angle [degrees]
        vaa : xr.DataArray (nrows, ncols)
            Viewing azimuth angle [degrees]
        """
        vza, vaa = self._compute_viewing_angles()

        vza_2d = self._interp_angle(self.nrows, self.ncols, vza)
        vaa_2d = self._interp_angle(self.nrows, self.ncols, vaa)

        vza_da = create_angle_dataarray(vza_2d, 'vza')
        vaa_da = create_angle_dataarray(vaa_2d, 'vaa')

        return vza_da, vaa_da

    def compute_all(self) -> xr.Dataset:
        """
        Compute all six angle fields and return as xarray.Dataset.

        Returns
        -------
        ds : xr.Dataset
            Dataset containing all angle fields as DataArrays:
            - 'sza': Solar zenith angle [degrees]
            - 'saa': Solar azimuth angle [degrees]
            - 'vza': Viewing zenith angle [degrees]
            - 'vaa': Viewing azimuth angle [degrees]
            - 'raa': Relative azimuth angle [degrees]
            - 'sga': Sun glint angle [degrees]

            Each DataArray has the same dimensions and coordinates as the input lon/lat.
        """
        # Compute base angles
        sza, saa = self.compute_sza_saa()
        vza, vaa = self.compute_vza_vaa()

        # Compute derived angles
        raa = compute_raa(saa, vaa)
        sga = compute_sga(sza, saa, vza, vaa)

        # Create DataArrays with proper metadata

        # Create Dataset
        ds = xr.Dataset({
            'sza': sza,
            'saa': saa,
            'vza': vza,
            'vaa': vaa,
            'raa': raa,
            'sga': sga,
        })

        return ds
