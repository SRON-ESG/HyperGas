#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
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
from rioxarray.rioxarray import _make_coords

ORTHO_RES = {'hsi_l1b': 30,  'hyc_l1': 30, 'emit_l1b': 60}
LOG = logging.getLogger(__name__)


class Ortho():
    """Apply orthorectification by DEM model data"""

    def __init__(self, scene, varname, rpcs=None):
        """Initialize ortho class.

        Args:
            scene (object): the scene defined by hyperch4
            varname (str): the loaded var to be orthorectified
        """
        self.scene = scene
        self.varname = varname
        self.rpcs = rpcs
        self.glt_x = getattr(scene, 'glt_x', None)
        self.glt_y = getattr(scene, 'glt_y', None)

        # check whether variables are loaded
        self._check_vars()

        # get the bounds of scene
        lons, lats = scene[varname].attrs['area'].get_lonlats()
        self.bounds = [lons.min(), lats.min(), lons.max(), lats.max()]

        # get the UTM epsg code
        self._utm_epsg()

        # download the DEM data if we use the RPC method
        if self.ortho_source == 'rpc':
            self._download_dem()

    def _check_vars(self):
        """Check necessary for ortho."""
        # hardcode the ortho res
        #   it is better to set it as same as the input data
        reader = list(self.scene._readers.keys())[0]
        self.ortho_res = ORTHO_RES[reader]

        # check already loaded vars
        loaded_varnames = [key['name'] for key in self.scene._datasets]

        if self.varname in loaded_varnames:
            self.data = self.scene[self.varname]
        else:
            raise ValueError(
                f'{self.varname} is not loaded. Please make sure the name is correct.')

        # check if we have rpc or glt loaded
        rpc_boolean = any(['rpc' in name for name in loaded_varnames])
        glt_boolean = any(['glt' in name for name in loaded_varnames])
        if rpc_boolean:
            self.ortho_source = 'rpc'
        elif glt_boolean:
            self.ortho_source = 'glt'
        else:
            raise ValueError(f'Neither rpc nor glt variabes are available in loaded vars: {loaded_varnames}')

    def _download_dem(self):
        """Download SRTMV3 DEM data."""
        LOG.debug('Downloading SRTMV3 using stitch_dem')
        # download DEM data and update config
        self.file_dem = Path(self.data.attrs['filename'].replace('.', '_dem.')).with_suffix('.tif')

        if not os.path.exists(self.file_dem):
            dst_area_or_point = 'Point'
            dst_ellipsoidal_height = False
            dem_name = 'srtm_v3'

            # get the DEM data
            X, p = stitch_dem(self.bounds,
                              dem_name=dem_name,
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
        try:
            LOG.debug('Calculate UTM EPSG using rioxarray')
            self.utm_epsg = self.scene['radiance'].rio.estimate_utm_crs().to_epsg()
        except:
            LOG.debug('Calculate UTM EPSG using pyproj')
            utm_crs_list = query_utm_crs_info(
                datum_name="WGS 84",
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

        Args:
            data (DataArray): it should has been written the transform.
        """
        # make coords from transform
        coords = _make_coords(
            data,
            data.rio.transform(),
            data.rio.width,
            data.rio.height,
            force_generate=True,
        )

        # assign to coords
        data.coords['y'] = coords['y']
        data.coords['x'] = coords['x']

        # add attrs
        data.coords['y'].attrs['units'] = 'metre'
        data.coords['x'].attrs['units'] = 'metre'
        data.coords['y'].attrs['standard_name'] = 'projection_y_coordinate'
        data.coords['y'].attrs['standard_name'] = 'projection_x_coordinate'
        data.coords['y'].attrs['long_name'] = 'y coordinate of projection'
        data.coords['x'].attrs['long_name'] = 'x coordinate of projection'

    def apply_ortho(self):
        """Apply orthorectification."""
        if self.ortho_source == 'rpc':
            LOG.debug('Reproject data using rpc')
            ortho_arr, dst_transform = warp.reproject(self.scene[self.varname].data,
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

            # create the DataArray by replacing values
            da_ortho = xr.DataArray(ortho_arr, dims=['bands', 'y', 'x'])

            # write crs and transform
            da_ortho.rio.write_transform(dst_transform, inplace=True)
            da_ortho.rio.write_crs(self.utm_epsg, inplace=True)

        elif self.ortho_source == 'glt':
            LOG.debug('Reproject data using glt')
            # the value is 0 if no data is available
            glt_valid_mask = (self.scene['glt_x'] != 0) & (self.scene['glt_y'] != 0)
            self.scene['glt_y'].load()
            self.scene['glt_x'].load()
            da_ortho = self.scene[self.varname][:, self.scene['glt_y']-1, self.scene['glt_x']-1].where(glt_valid_mask)

            # write crs and transform
            da_ortho.rio.write_transform(Affine.from_gdal(*da_ortho.attrs['geotransform']), inplace=True)
            da_ortho.rio.write_crs(CRS.from_wkt(da_ortho.attrs['spatial_ref']), inplace=True)
            da_ortho = da_ortho.rename({'ortho_y': 'y', 'ortho_x': 'x'})

            # reproject to UTM
            da_ortho = da_ortho.rio.reproject(self.utm_epsg, nodata=np.nan)

        else:
            raise ValueError('Please load `rpc` or `glt` variables for ortho.')

        # assign coords
        self._assign_coords(da_ortho)

        # copy attrs
        da_ortho = da_ortho.rename(self.scene[self.varname].name)
        da_ortho.attrs = self.scene[self.varname].attrs
        ortho_description = f'orthorectified by the {self.ortho_source} method'
        if 'description' in da_ortho.attrs:
            da_ortho.attrs['description'] = f"{da_ortho.attrs['description']} ({ortho_description})"

        # write area attrs
        area_def = AreaDefinition(area_id='area',
                                  description=f'Area of data with EPSG:{self.utm_epsg}',
                                  proj_id='area',
                                  projection=da_ortho.rio.crs,
                                  width=da_ortho.rio.width,
                                  height=da_ortho.rio.height,
                                  area_extent=da_ortho.rio.bounds()
                                  )
        da_ortho.attrs['area'] = area_def

        # # set fill value to nan
        # if '_FillValue' in da_ortho.attrs:
        #     da_ortho = da_ortho.where(da_ortho != da_ortho.attrs['_FillValue'])
        #     del da_ortho.attrs['_FillValue']

        return da_ortho
