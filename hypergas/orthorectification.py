#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Apply orthorectification to data."""

import logging
import os
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from dem_stitcher.stitcher import stitch_dem
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from pyresample.geometry import AreaDefinition
from rasterio import warp
from rasterio.enums import Resampling
from rasterio.warp import transform
from rasterio.control import GroundControlPoint as GCP

LOG = logging.getLogger(__name__)


class Ortho():
    """Apply orthorectification by DEM model data."""

    def __init__(self, scene, varname, rpcs=None, gcps=None, gcp_crs=None):
        """Initialize ortho class.

        See the reprojection page of `rasterio <https://rasterio.readthedocs.io/en/stable/topics/reproject.html>`_ for details.

        Parameters
        ----------
        scene : object
            The Satpy Scene defined by hypergas.
        varname : str
            The loaded variable to be orthorectified.
        rpcs:
            The Rational Polynomial Coefficients (rpcs).
        gcps:
            The Ground Control Points (gcps).
        gcp_crs : EPSG code
            The CRS of the GCP data.
        """
        self.scene = scene
        self.varname = varname
        self.rpcs = rpcs
        self.gcps = gcps
        self.gcp_crs = gcp_crs

        self.glt_x = getattr(scene, 'glt_x', None)
        self.glt_y = getattr(scene, 'glt_y', None)

        # check whether variables are loaded
        self._check_vars()

        # get the bounds of scene
        self.lons, self.lats = scene[varname].attrs['area'].get_lonlats()
        self.bounds = [self.lons.min(), self.lats.min(), self.lons.max(), self.lats.max()]

        # get the UTM epsg code
        self._utm_epsg()

        # get the default transform info
        self._get_default_transform()

        # download the DEM data if we use the RPC method
        if self.ortho_source == 'rpc':
            self._download_dem()

    def _check_vars(self):
        """Check necessary variables for ortho."""
        # check already loaded vars
        loaded_varnames = [key['name'] for key in self.scene._datasets]

        if self.varname in loaded_varnames:
            self.data = self.scene[self.varname]
        else:
            raise ValueError(
                f'{self.varname} is not loaded. Please make sure the name is correct.')

        # check if we have rpc or glt loaded
        rpc_boolean = (any(['rpc' in name for name in loaded_varnames])) or (self.rpcs is not None)
        glt_boolean = any(['glt' in name for name in loaded_varnames])

        if rpc_boolean:
            self.ortho_source = 'rpc'
        elif glt_boolean:
            self.ortho_source = 'glt'
        elif self.gcps is not None:
            self.ortho_source = 'gcp'
        else:
            self.ortho_source = 'none'

    def _get_default_transform(self):
        # calculate the pixel resolution for the UTM projection
        src_height, src_width = self.lons.shape
        self.default_dst_transform, self.dst_width, self.dst_height = warp.calculate_default_transform(
            src_crs='EPSG:4326',
            dst_crs=self.utm_epsg,
            width=src_width,
            height=src_height,
            src_geoloc_array=(self.lons, self.lats),
        )
        self.ortho_res = abs(self.default_dst_transform.a)

    def _download_dem(self):
        """Download SRTMV3 DEM data."""
        LOG.debug('Downloading SRTMV3 using stitch_dem')

        # download DEM data and update config
        dem_path = Path(self.data.attrs['filename'])
        self.file_dem = dem_path.with_name(dem_path.stem+'_dem.tif')

        if not os.path.exists(self.file_dem):
            dst_area_or_point = 'Point'
            dst_ellipsoidal_height = False
            dem_names = ['srtm_v3', 'glo_30']

            # get the DEM data 1) SRTM V3 2) Copernicus GLO-30
            try:
                X, p = stitch_dem(self.bounds,
                                  dem_name=dem_names[0],
                                  dst_ellipsoidal_height=dst_ellipsoidal_height,
                                  dst_area_or_point=dst_area_or_point)
            except Exception as error:
                LOG.info(error)
                LOG.info('Downloading GLO-30 instead of SRTMV3')
                X, p = stitch_dem(self.bounds,
                                  dem_name=dem_names[1],
                                  dst_ellipsoidal_height=dst_ellipsoidal_height,
                                  dst_area_or_point=dst_area_or_point)

            # export to tif file
            with rasterio.open(self.file_dem, 'w', **p) as ds:
                ds.write(X, 1)
                ds.update_tags(AREA_OR_POINT=dst_area_or_point)

    def _utm_epsg(self):
        """Find the suitable UTM epsg code based on lons and lats

        Ref:
            https://pyproj4.github.io/pyproj/stable/examples.html#find-utm-crs-by-latitude-and-longitude
        """
        LOG.debug('Calculate UTM EPSG using pyproj')
        utm_crs_list = query_utm_crs_info(
            datum_name='WGS 84',
            area_of_interest=AreaOfInterest(
                west_lon_degree=self.bounds[0],
                south_lat_degree=self.bounds[1],
                east_lon_degree=self.bounds[2],
                north_lat_degree=self.bounds[3],
            ),
        )

        self.utm_epsg = CRS.from_epsg(utm_crs_list[0].code).to_epsg()

    def _assign_coords(self, data):
        """Calculate and assign the UTM coords

        Parameters
        ----------
            data (DataArray): it should has the transform info.
        """
        # assign coords from AreaDefinition
        data.coords['y'] = data.attrs['area'].projection_y_coords
        data.coords['x'] = data.attrs['area'].projection_x_coords

        # add attrs
        data.coords['y'].attrs['units'] = 'm'
        data.coords['x'].attrs['units'] = 'm'
        data.coords['y'].attrs['standard_name'] = 'projection_y_coordinate'
        data.coords['y'].attrs['standard_name'] = 'projection_x_coordinate'
        data.coords['y'].attrs['long_name'] = 'y coordinate of projection'
        data.coords['x'].attrs['long_name'] = 'x coordinate of projection'

    def _assign_area(self, da_ortho, dst_transform):
        """Assign the area attrs

        Parameters
        ----------
        da_ortho : DataArray
            The orthorectified data.
        dst_transform : Affine
            The target transform (Affine order).
        """
        if self.ortho_res is not None:
            target_area = AreaDefinition.from_ul_corner(area_id=f"{self.scene[self.varname].attrs['sensor']}_utm",
                                                        projection=f'EPSG:{self.utm_epsg}',
                                                        shape=(da_ortho.sizes['y'], da_ortho.sizes['x']),
                                                        upper_left_extent=(dst_transform[2], dst_transform[5]),
                                                        resolution=self.ortho_res
                                                        )
        else:
            raise ValueError('ortho_res dict is empty for your instrument')

        da_ortho.attrs['area'] = target_area

    def apply_ortho(self):
        """Apply orthorectification.

        - EnMAP: using the RPC data from the L1 product;
        - EMIT: using the glt table from the L1 product;
        - PRISMA: using the manual gcp points from QGIS.

        Returns
        -------
        da_ortho : :class:`~xarray.DataArray`
            The orthorectified data.
        """
        # read data and expand to 3d array with "band" dim for rioxarray
        data = self.scene[self.varname]
        data_name = data.name

        if len(data.dims) == 2:
            data = data.expand_dims(dim={'band': 1})

        # load into normal array
        if 'source' in data.dims:
            source_coord = data.coords['source'].values
        dims = data.dims
        data_sizes = data.shape

        if self.ortho_source == 'rpc':
            LOG.debug(f'Orthorectify {data_name} using rpc')
            ortho_arr, dst_transform = warp.reproject(data.values,
                                                      rpcs=self.rpcs,
                                                      src_crs='EPSG:4326',
                                                      dst_crs=f'EPSG:{self.utm_epsg}',
                                                      dst_resolution=self.ortho_res,
                                                      # src_nodata=self._raw_nodata,
                                                      dst_nodata=np.nan,
                                                      # num_threads=MAX_CORES,
                                                      resampling=Resampling.nearest,
                                                      RPC_DEM=self.file_dem,
                                                      )
        elif self.ortho_source == 'glt':
            LOG.debug(f'Orthorectify {data_name} using glt')
            # Adjust for One based Index
            #   the value is 0 if no data is available
            glt_valid_mask = (self.scene['glt_x'] != 0) & (self.scene['glt_y'] != 0)
            self.scene['glt_y'].load()
            self.scene['glt_x'].load()

            # select value and set fill_value to nan
            da_ortho = data[:, self.scene['glt_y']-1, self.scene['glt_x']-1].where(glt_valid_mask)

            # create temporary array because we perfer using AreaDefinition later
            tmp_da = da_ortho.copy()

            # write crs and transform
            #   the EMIT geotransform attrs is gdal order
            tmp_da.rio.write_transform(Affine.from_gdal(*tmp_da.attrs['geotransform']), inplace=True)
            tmp_da.rio.write_crs(CRS.from_wkt(tmp_da.attrs['spatial_ref']), inplace=True)
            tmp_da = tmp_da.rename({'ortho_y': 'y', 'ortho_x': 'x'})

            # reproject to UTM
            tmp_da = tmp_da.rio.reproject(self.utm_epsg, nodata=np.nan, resolution=self.ortho_res)
            ortho_arr = tmp_da.data
            dst_transform = tmp_da.rio.transform()

        elif self.ortho_source == 'gcp':
            LOG.info(f'Orthorectify {data_name} using gcp')
            # Step 1: Project lons/lats to UTM
            # self.lons and self.lats have shape (height, width)
            utm_x, utm_y = transform(
                'EPSG:4326', f'EPSG:{self.utm_epsg}',
                self.lons.flatten(), self.lats.flatten()
            )

            # Reshape to 2D
            utm_x = np.array(utm_x).reshape(self.lons.shape)
            utm_y = np.array(utm_y).reshape(self.lats.shape)

            # Step 2: For each GCP, find nearest pixel
            gcps_corr = []
            for sx, sy, mapx, mapy in zip(self.gcps['sourceX'], self.gcps['sourceY'], self.gcps['mapX'], self.gcps['mapY']):
                # Compute distance to all pixels
                dist = np.sqrt((utm_x - sx)**2 + (utm_y - sy)**2)
                idx = np.unravel_index(np.argmin(dist), dist.shape)
                row, col = idx  # pixel coordinates
                gcps_corr.append(GCP(row=row, col=col, x=mapx, y=mapy))

            destination = np.full((data_sizes[0], self.dst_height, self.dst_width), np.nan)
            ortho_arr, dst_transform = warp.reproject(data.values,
                                                      destination=destination,
                                                      gcps=gcps_corr,
                                                      src_crs=CRS.from_epsg(self.gcp_crs),
                                                      dst_crs=CRS.from_epsg(self.utm_epsg),
                                                      dst_transform=self.default_dst_transform,
                                                      dst_nodata=np.nan,
                                                      resampling=Resampling.nearest,
                                                      )
        else:
            LOG.info('`rpc` or `glt` is missing. Please check the accuracy of orthorectification manually.')
            destination = np.full((data_sizes[0], self.dst_height, self.dst_width), np.nan)

            ortho_arr, dst_transform = warp.reproject(data.values,
                                                      destination=destination,
                                                      src_crs=CRS.from_epsg(4326),
                                                      dst_crs=CRS.from_epsg(self.utm_epsg),
                                                      dst_transform=self.default_dst_transform,
                                                      dst_nodata=np.nan,
                                                      src_geoloc_array=np.stack((self.lons, self.lats))
                                                      )

        # create the DataArray by replacing values
        da_ortho = xr.DataArray(ortho_arr, dims=dims)

        # assign source coords for wind data if exists
        if 'source' in dims:
            da_ortho.coords['source'] = source_coord

        # copy attrs
        da_ortho = da_ortho.rename(self.scene[self.varname].name)
        da_ortho.attrs = self.scene[self.varname].attrs

        ortho_description = f'orthorectified by the {self.ortho_source} method'
        if 'description' in da_ortho.attrs:
            da_ortho.attrs['description'] = f"{da_ortho.attrs['description']} ({ortho_description})"

        # update area attrs
        LOG.debug('Assign UTM Area definition')
        self._assign_area(da_ortho, dst_transform)

        # assign coords
        LOG.debug('Assign coords to orthorectified data')
        self._assign_coords(da_ortho)

        return da_ortho
