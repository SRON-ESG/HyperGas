#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Plot orthorectified data and overlay images on folium maps"""

import base64
import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import folium
import geojson
import numpy as np
from cartopy.crs import epsg as ccrs_from_epsg
from folium.features import DivIcon
from folium.plugins import (Draw, FeatureGroupSubGroup, Fullscreen, Geocoder,
                            MousePosition)
from geoarray import GeoArray
from py_tools_ds.geo.coord_trafo import reproject_shapelyGeometry
from pyresample.utils import load_cf_area

LOG = logging.getLogger(__name__)


class Map():
    """Plot data on folium maps"""

    def __init__(self, dataset, varnames, center_map=None):
        """Initialize Map class.

        Args:
            dataset (xarray Dataset)
            varnames (list): the list of varnames to be plotted
            center_map (list): map center, [latitude, longitude]

        Usage:
            basic:
                m = Map(ds, ['rgb', 'ch4'])
                m.initialize()
                m.plot()
                m.export()

            full_params:
                m = Map(ds, ['rgb', 'ch4'])
                m.initialize()
                m.plot(show_layers=[False, True], opacities=[0.9, 0.7])
                m.export('full_params.html')

            Multiple datasets on one map:
                m = Map(ds1, ['rgb', 'ch4'])
                m.initialize()
                m.plot()
                m.ds = ds2
                m.varnames = varnames2
                m.plot()
                m.export()
        """
        self.ds = dataset
        self.varnames = varnames

        # check all variabels are loaded
        self._check_vars()

        # calculate the map center
        if center_map is None:
            self._calc_center()
        else:
            self.center_map = center_map

    def _check_vars(self):
        """Check variables are already loaded."""
        loaded_varnames = list(self.ds.data_vars)
        if not all([var in loaded_varnames for var in self.varnames]):
            raise ValueError(
                f'{self.varnames} are not all loaded. Please make sure the name is correct and call terrain_corr() after loading it.')

    def _calc_center(self):
        """Calculate center lon and lat based on geotransform info."""
        center_lon = self.ds.coords['longitude'].mean()
        center_lat = self.ds.coords['latitude'].mean()
        self.center_map = [center_lat, center_lon]

    def _get_cartopy_crs_from_epsg(self, epsg_code):
        if epsg_code:
            try:
                return ccrs_from_epsg(epsg_code)
            except ValueError:
                if epsg_code == 4326:
                    return ccrs.PlateCarree()
                else:
                    raise NotImplementedError('The show_map() method currently does not support the given '
                                              'projection.')
        else:
            raise ValueError(f'Expected a valid EPSG code. Got {epsg_code}.')

    def initialize(self):
        """Set the basic folium map background"""
        m = folium.Map(location=self.center_map, zoom_start=12, tiles=None, control_scale=True)

        openstreet_tile = folium.TileLayer('OpenStreetMap')

        esri_tile = folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
        )

        # add tile
        m.add_child(esri_tile)
        m.add_child(openstreet_tile)

        # add full screen
        Fullscreen(
            position='topleft',
            title='Expand me',
            title_cancel='Exit me',
            force_separate_button=True,
        ).add_to(m)

        # add draw menu
        output_geojson = str(
            Path(os.path.basename(self.ds[self.varnames[0]].attrs['filename']).replace('RAD', '')).with_suffix('.geojson'))
        Draw(export=True, position='topleft', filename=output_geojson).add_to(m)

        # add mouse position
        MousePosition(position='bottomleft').add_to(m)

        # add geocoder search
        Geocoder(position='topleft', collapsed=True).add_to(m)

        self.map = m

    def plot(self, out_epsg=3857, show_layers=None, opacities=None):
        """Plot data and export to png files

        Args:
            out_epsg (int): EPSG code of the output projection (3857 is the proj of folium Map)
            show_layers (boolean list): Whether the layers will be shown on opening (the length should be as same as varnames)
            opacities (float list): the opacities of layer (the length should be as same as varnames)
        """
        # check the length of self.
        if show_layers is None:
            self.show_layers = [True]*len(self.varnames)
        else:
            self.show_layers = show_layers
        if len(self.show_layers) != len(self.varnames):
            raise ValueError(
                f"self.'s length ({len(self.show_layers)}) should be as same as varnames's length ({len(self.varnames)})")

        if opacities is None:
            self.opacities = [0.7]*len(self.varnames)
        else:
            self.opacities = opacities
        if len(self.opacities) != len(self.varnames):
            raise ValueError(
                f"opacities's length ({len(self.opacities)}) should be as same as varnames's length ({len(self.varnames)})")

        for varname in self.varnames:
            # load data
            da_ortho = self.ds[varname]

            # get bands info which is useful for plotting automatically
            if 'bands' in da_ortho.dims:
                # e.g. RGB
                band_array = np.arange(da_ortho.sizes['bands'])
                input_data = da_ortho.fillna(0).transpose(..., 'bands').values
            else:
                band_array = 0
                input_data = da_ortho.fillna(0).values

            # fill nan values by 0 to plot data in transparent
            area, _ = load_cf_area(self.ds)

            # gdal order transform
            # (c, a, b, f, d, e)
            # https://rasterio.readthedocs.io/en/latest/topics/migrating-to-v1.html#affine-affine-vs-gdal-style-geotransforms
            geotransform = (area.upper_left_extent[0], area.pixel_size_x, 0,
                            area.upper_left_extent[1], 0, -area.pixel_size_y)
            ga = GeoArray(input_data,
                          geotransform=geotransform,
                          projection=area.crs.to_wkt(),
                          q=True,
                          nodata=0)

            # set cmap
            if 'rgb' not in varname:
                # it should be ch4 (ppb)
                cmap = 'plasma'
                vmin = 0
                vmax = 600
            else:
                cmap = None
                vmin = None
                vmax = None

            # call show_map function with original resolution
            fig, ax = ga.show_map(vmin=vmin, vmax=vmax, cmap=cmap,
                                  band=band_array, res_factor=1, out_epsg=out_epsg,
                                  return_map=True, draw_gridlines=False)

            # # --- back up --- #
            # # --- matplotlib vesion --- #
            # # the output image has strange shape ...
            # if varname == 'rgb':
            #     # create RGBA data
            #     import xarray as xr
            #     da_ortho = xr.concat([da_ortho, (~da_ortho.isnull().all(dim='bands'))], dim='bands').transpose(..., 'bands')
            #     cmap = None
            #     vmin = None
            #     vmax = None
            # else:
            #     # it should be ch4 (ppb)
            #     cmap = 'plasma'
            #     vmin = 0
            #     vmax = 600

            # # load the area from Dataset or NetCDF file
            # #   Note that you do not need to decode_coords='all' when you open the NetCDF file
            # area, _ = load_cf_area(self.ds)

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(subplot_kw=dict(projection=self._get_cartopy_crs_from_epsg(out_epsg)))

            # # set extent
            # input_crs = area.to_cartopy_crs()
            # ax.set_extent(input_crs.bounds, crs=input_crs)

            # # plot data
            # plt.imshow(da_ortho, cmap=cmap, vmin=vmin, vmax=vmax,
            #            transform=input_crs, extent=input_crs.bounds,
            #            origin='upper', regrid_shape=3000,
            #            )
            # # --- backup --- #

            # turn off axis
            ax.axis('off')

            # set png filename
            #   hard code for renaming EMIT RAD filename
            output_png = Path(da_ortho.attrs['filename'].replace(
                '.', f'_{varname}.').replace('RAD', '')).with_suffix('.png')

            # delete pads and remove edges
            fig.savefig(output_png, bbox_inches='tight', pad_inches=0.0, edgecolor=None, transparent=True, dpi=1000)

        # calculate the bounds
        #   we need to use bounds for image overlay on folium map
        extent_4326 = ax.get_extent(crs=ccrs.PlateCarree())

        self.img_bounds = [[extent_4326[2], extent_4326[0]], [extent_4326[3], extent_4326[1]]]

        # get the swath polygon
        lonlatPoly = reproject_shapelyGeometry(ga.footprint_poly, ga.prj, 4326)
        self.gjs = geojson.Feature(geometry=lonlatPoly, properties={})
        # # --- backup --- #
        # # pyresample version
        # import pyproj
        # from pyresample.gradient import get_polygon
        # lonlatPoly = get_polygon(pyproj.Proj('EPSG:4326'), area)
        # self.gjs = geojson.Feature(geometry=lonlatPoly, properties={})

        # overlay images on foilum map
        self.plot_folium()

    def plot_wind(self, source='ERA5', position='bottomright'):
        """plot the wind as html element

        Args:
            source (str): ERA5 or GEOS-FP
            position (str)
        """
        # read data
        u10 = self.ds['u10'].sel(source=source).item()
        v10 = self.ds['v10'].sel(source=source).item()

        # calculate wspd and wdir
        wspd = np.sqrt(u10**2 + v10**2)
        wdir = (270 - np.rad2deg(np.arctan2(v10, u10))) % 360

        # read arrow image
        _dirname = os.path.dirname(__file__)
        arrow_img = os.path.join(_dirname, 'imgs', 'arrow.png')

        with open(arrow_img, "rb") as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

        html = f"""
        <img src="data:image/png;base64,{image_base64}" style="transform:rotate({wdir+180}deg);">
        <h1 style="color:white;">{wspd:.1f} m/s</h1>
        """

        # add marker with icon
        icon = DivIcon(html=html)
        gplot = FeatureGroupSubGroup(self.fg, f'{self.time_str} | {source} Wind', show=False)
        self.map.add_child(gplot)
        gplot.add_child(folium.Marker(self.center_map, icon=icon, draggable=True))

    def plot_folium(self):
        """Overlay plotted png images on folium map"""
        # get the time string
        self.time_str = self.ds[self.varnames[0]].attrs['start_time']
        sensor_name = self.ds[self.varnames[0]].attrs['sensor']

        # add swath poly
        style = {'fillColor': '#00000000', 'color': 'dodgerblue'}
        folium.GeoJson(self.gjs,
                       control=False,
                       tooltip=self.time_str,
                       zoom_on_click=False,
                       style_function=lambda x: style,
                       # highlight_function= lambda feat: {'fillColor': 'blue'},
                       ).add_to(self.map)

        # add the group which controls all subgroups (varnames)
        self.fg = folium.FeatureGroup(name=f'{self.time_str} group ({sensor_name})')
        self.map.add_child(self.fg)

        for index, varname in enumerate(self.varnames):
            layer_name = f'{self.time_str} | {varname}'
            gplot = FeatureGroupSubGroup(self.fg, layer_name, show=self.show_layers[index])
            self.map.add_child(gplot)

            output_png = Path(self.ds[varname].attrs['filename'].replace(
                '.', f'_{varname}.').replace('RAD', '')).with_suffix('.png')
            raster = folium.raster_layers.ImageOverlay(image=str(output_png),
                                                       opacity=self.opacities[index],
                                                       bounds=self.img_bounds,
                                                       name=layer_name,
                                                       )
            raster.add_to(gplot)

        # plot wind
        self.plot_wind(source='ERA5')
        self.plot_wind(source='GEOS-FP')

    def export(self, savename=None):
        """Export plotted folium map to html file"""
        layer_control = folium.LayerControl(collapsed=False, position='topleft', draggable=True)
        self.map.add_child(layer_control)

        if savename is None:
            savename = str(Path(self.ds[self.varnames[0]].attrs['filename'].replace('RAD', '')).with_suffix('.html'))

        LOG.info(
            f'Export folium map to {savename}')
        self.map.save(savename)
