#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Download DEM data for the region observed by the hyperspectral satellite."""

import os
import logging
import rasterio
from pathlib import Path
from dem_stitcher.stitcher import stitch_dem


LOG = logging.getLogger(__name__)


class DEM():
    """
    Download and store a Digital Elevation Model (DEM) for a given bounding box.
    """

    def __init__(
        self,
        filename: str,
        bounds: list,
        buffer: float = 0.01,
        dem_name: str = "glo_30",
        dst_area_or_point: str = "Point",
        dst_ellipsoidal_height: bool = False,
    ):
        """
        Parameters
        ----------
        filename : str
            Path to the input file. The DEM will be saved as:
            <filename_root>_dem.tif
        bounds : list[float]
            Bounding box in EPSG:4326 format:
            [xmin, ymin, xmax, ymax] (longitude, latitude).
        buffer : float, optional
            Buffer (in degrees) added around the bounding box.
            Default is 0.01.
        dem_name : str, optional
            Name of the DEM dataset supported by `dem_stitcher`.
            Default is "glo_30".
        dst_area_or_point : str, optional
            Pixel referencing type:
            - "Area": pixel value represents area (upper-left corner reference, GDAL default)
            - "Point": pixel value represents the pixel center
            Default is "Point".
        dst_ellipsoidal_height : bool, optional
            If True, output heights are ellipsoidal.
            If False, heights are relative to the geoid.
            Default is False.
        """

        # ---- Validate bounds ----
        if len(bounds) != 4:
            raise ValueError("`bounds` must be [xmin, ymin, xmax, ymax].")

        xmin, ymin, xmax, ymax = bounds
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid bounds: xmin < xmax and ymin < ymax required.")

        # ---- Validate dst_area_or_point ----
        if dst_area_or_point not in {"Area", "Point"}:
            raise ValueError("`dst_area_or_point` must be 'Area' or 'Point'.")

        # ---- Prepare output path ----
        dem_path = Path(filename)
        self.file_dem = dem_path.with_name(f"{dem_path.stem}_dem.tif")

        self.dem_name = dem_name
        self.dst_area_or_point = dst_area_or_point
        self.dst_ellipsoidal_height = dst_ellipsoidal_height

        # Apply buffer to bounds
        self.bounds = [
            xmin - buffer,
            ymin - buffer,
            xmax + buffer,
            ymax + buffer,
        ]

    def download(self, overwrite: bool = False) -> Path:
        """
        Download the DEM and save it to disk.

        Parameters
        ----------
        overwrite : bool, optional
            If True, force re-download even if file exists.
            Default is False.

        Returns
        -------
        Path
            Path to the downloaded DEM file.
        """

        if self.file_dem.exists() and not overwrite:
            LOG.debug(f"Skipping download. DEM already exists: {self.file_dem}")
            return self.file_dem

        LOG.debug(
            f"Downloading DEM '{self.dem_name}' "
            f"for bounds {self.bounds}"
        )

        # Fetch DEM data
        dem_array, profile = stitch_dem(
            self.bounds,
            dem_name=self.dem_name,
            dst_ellipsoidal_height=self.dst_ellipsoidal_height,
            dst_area_or_point=self.dst_area_or_point,
        )

        # Ensure output directory exists
        self.file_dem.parent.mkdir(parents=True, exist_ok=True)

        # Write GeoTIFF
        with rasterio.open(self.file_dem, "w", **profile) as dst:
            dst.write(dem_array, 1)
            dst.update_tags(
                AREA_OR_POINT=self.dst_area_or_point,
                source=self.dem_name,
            )

        LOG.info(f"DEM saved to {self.file_dem}")

        return self.file_dem
