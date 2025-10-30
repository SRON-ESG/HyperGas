#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Create 2D landmask for hyperspectral satellite data."""

import logging
import os
import gc

import cartopy.feature as cfeature
import geopandas as gpd
import numpy as np
import xarray as xr
import yaml
from pyresample import kd_tree
from pyresample.geometry import GridDefinition, SwathDefinition

LOG = logging.getLogger(__name__)


def find_tiles(lat_min, lat_max, lon_min, lon_max):
    """
    Generate the list of OSM filenames for the tiles that cover the given bounding box.

    Parameters
    ----------
    lat_min : float
        minimum latitude of the bounding box.
    lat_max : float
        maximum latitude of the bounding box.
    lon_min : float
        minimum longitude of the bounding box.
    lon_max : float
        maximum longitude of the bounding box.

    Returns
    -------
    tiles : list
        The list of tif filenames.
    """
    lat_range = range(int(lat_min // 5 * 5), int((lat_max // 5 + 1) * 5), 5)
    lon_range = range(int(lon_min // 5 * 5), int((lon_max // 5 + 1) * 5), 5)

    tiles = []
    for lat in lat_range:
        for lon in lon_range:
            lat_prefix = 'n' if lat >= 0 else 's'
            lon_prefix = 'e' if lon >= 0 else 'w'
            tile_filename = f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}.tif"
            tiles.append(tile_filename)

    return tiles


def Land_mask(lons, lats, source='OSM'):
    """Create the segmentation for land and ocean/lake types.

    Parameters
    ----------
    lons : :class:`numpy.ndarray`
        2D longitude of pixels.
    lats : :class:`numpy.ndarray`
        2D latitude of pixels.
    source : str
        The data source of land mask (“OSM”, “GSHHS” or “Natural Earth”), Default: “OSM”.

    Returns
    -------
    Land_mask : :class:`~xarray.DataArray`
        2D array, 0: ocean/lake, 1: land
    """
    # load land data
    LOG.info(f'Creating land mask using {source} data')
    if source == 'OSM':
        # get the OSM data path
        _dirname = os.path.dirname(__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)
        osm_dir = os.path.join(_dirname, settings['data']['osm_dir'])

        # read OSM+ESA_WorldCover Watermask
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        osm_filenames = find_tiles(lat_min, lat_max, lon_min, lon_max)
        osm_paths = [os.path.join(osm_dir, fname) for fname in osm_filenames]

        # Load and crop the OSM data
        with xr.open_mfdataset(osm_paths) as ds_osm:
            da_osm = ds_osm['band_data'].isel(band=0)
            osm_crop = da_osm.sel(y=slice(lat_max, lat_min), x=slice(lon_min, lon_max))
            osm_crop.load()

        # set the resample grids
        lon_grid, lat_grid = np.meshgrid(osm_crop.x, osm_crop.y)
        swath_def = SwathDefinition(lons=lons, lats=lats)

        # resample data by nearest (10 m)
        grid_def = GridDefinition(lons=lon_grid, lats=lat_grid)
        landmask = kd_tree.resample_nearest(grid_def, osm_crop.data, swath_def, radius_of_influence=10).astype('int')
        # landmask: 0->1, 1->0
        landmask = np.where((landmask == 0) | (landmask == 1), landmask ^ 1, landmask).astype(float)

        del da_osm, osm_crop, lon_grid, lat_grid
        gc.collect()

    elif source in ['Natural Earth', 'GSHHS']:
        if source == 'Natural Earth':
            land_data = cfeature.NaturalEarthFeature('physical', 'land', '10m')
        elif source == 'GSHHS':
            land_data = cfeature.GSHHSFeature(scale='full')

        # load data into GeoDataFrame
        land_polygons = list(land_data.geometries())
        land_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=land_polygons)

        # create Point GeoDataFrame
        points = gpd.GeoSeries(gpd.points_from_xy(lons.ravel(), lats.ravel()))
        points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

        # Spatially join the points with the land polygons
        joined = gpd.sjoin(points_gdf, land_gdf, how='left', predicate='within')

        # Check if each point is within a land polygon
        is_within_land = joined['index_right'].notnull()

        # create the mask
        landmask = is_within_land.values.reshape(lons.shape).astype(float)

    else:
        raise ValueError(
            "Please input the correct land data source ('GSHHS' or 'Natural Earth'). {land_data} is not supported")

    LOG.info(f'Creating land mask using {source} data (Done)')
    # save to DataArray
    segmentation = xr.DataArray(landmask, dims=['y', 'x'])
    segmentation.attrs['description'] = f'{source} land mask (0: ocean/lake, 1: land)'

    return segmentation
