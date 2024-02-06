#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Create 2D landmask for hyperspectral satellite data."""

import xarray as xr
import geopandas as gpd
import cartopy.feature as cfeature


def Land_mask(lon, lat, source='GSHHS'):
    """Create the segmentation for land and ocean/lake types

        Args:
            lon (2d array): longitude of pixels
            lat (2d array): latitude of pixels
        Return:
            Land_mask (2D DataArray): 0: ocean/lake, 1: land
    """
    # load land data
    if source == 'Natural Earth':
        land_data = cfeature.NaturalEarthFeature('physical', 'land', '10m')
    elif source == 'GSHHS':
        land_data = cfeature.GSHHSFeature(scale='full')
    else:
        raise ValueError("Please input the correct land data source ('GSHHS' or 'Natural Earth'). {land_data} is not supported")

    # load data into GeoDataFrame
    land_polygons = list(land_data.geometries())
    land_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=land_polygons)
    
    # create Point GeoDataFrame
    points = gpd.GeoSeries(gpd.points_from_xy(lon.ravel(), lat.ravel()))
    points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    
    # Spatially join the points with the land polygons
    joined = gpd.sjoin(points_gdf, land_gdf, how='left', op='within')

    # Check if each point is within a land polygon
    is_within_land = joined['index_right'].notnull()
    
    # create the mask
    landmask = is_within_land.values.reshape(lon.shape).astype(float)
    
    # save to DataArray
    segmentation = xr.DataArray(landmask, dims=['y', 'x'])
    segmentation.attrs['description'] = f'{source} land mask (0: ocean/lake, 1: land)'

    return segmentation
