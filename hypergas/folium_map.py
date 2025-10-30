#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Plot orthorectified data and overlay images on folium maps."""

import base64
import gc
import logging
import os
from pathlib import Path

import cartopy.crs as ccrs
import folium
import geojson
import matplotlib.pyplot as plt
import numpy as np
from cartopy.crs import epsg as ccrs_from_epsg
from folium.features import DivIcon
from folium.plugins import (Draw, FeatureGroupSubGroup, Fullscreen, Geocoder,
                            MousePosition)
from pyresample.geometry import SwathDefinition
from scipy.spatial import ConvexHull
from shapely import geometry

LOG = logging.getLogger(__name__)


class Map():
    """Plot data on folium maps."""

    def __init__(self, dataset, varnames, center_map=None):
        """Initialize Map class.

        Parameters
        ----------
        dataset : :class:`~xarray.Dataset`
            The xarray dataset which contains gas fields and geolocations (longitude and latitude).
        varnames : list
            The list of varnames to be plotted.
        center_map : list
            The map center: [latitude, longitude].

        Examples
        --------
        Basic usage::
        
            m = Map(ds, ['rgb', 'ch4'])
            m.initialize()
            m.plot()
            m.export()
        
        Full parameters::
        
            m = Map(ds, ['rgb', 'ch4'])
            m.initialize()
            m.plot(show_layers=[False, True], opacities=[0.9, 0.7])
            m.export('full_params.html')
        
        Multiple datasets on one map::
        
            m = Map(ds1, ['rgb', 'ch4'])
            m.initialize()
            m.plot()
            m.ds = ds2
            m.varnames = varnames2
            m.plot()
            m.export()
        """
        self.ds = dataset
        self.filename = dataset.attrs['filename']
        self.varnames = varnames

        # check all variabels are loaded
        self._check_vars()

        # get the swath info
        self.swath = SwathDefinition(lons=self.ds.coords['longitude'], lats=self.ds.coords['latitude'])

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
        # The mean value doesn't work well
        # center_lon = self.ds.coords['longitude'].mean()
        # center_lat = self.ds.coords['latitude'].mean()

        corners = self.swath.corners

        # Extract lon and lat separately
        lons_corner = [c.lon for c in corners]
        lats_corner = [c.lat for c in corners]
        center_lon = np.rad2deg(np.mean(lons_corner))
        center_lat = np.rad2deg(np.mean(lats_corner))

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
        """Set the basic folium map background."""
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
            Path(os.path.basename(self.filename).replace('_RAD', '').replace('L1', 'L2')).with_suffix('.geojson'))
        Draw(export=True, position='topleft', filename=output_geojson).add_to(m)

        # add mouse position
        MousePosition(position='bottomleft').add_to(m)

        # add geocoder search
        Geocoder(position='topleft', collapsed=True).add_to(m)

        self.map = m

    def plot_png(self, out_epsg=3857, vmax=None, export_dir=None, pre_suffix=''):
        """Plot data and export to png files.

        Parameters
        ----------
        out_epsg : int
            EPSG code of the output projection (3857 is the proj of folium Map).
        vmax : float
            The cmap vmax for plotting species (unit is as same as species variable).
        export_dir : str
            The directory to save plotted images (Default: the same path as filename attrs).
        pre_suffix : str
            The suffix added to the png and html filename (Default: ""). 
        """

        for varname in self.varnames:
            # load data
            da_ortho = self.ds[varname]

            # plot variables
            if varname == 'rgb':
                # transpose the bands
                da_ortho = da_ortho.transpose(..., 'bands')
                cmap = None
                vmin = None
                cmap_vmax = None
            elif varname == 'radiance_2100':
                cmap = 'viridis'
                vmin = da_ortho.quantile(0.01)
                cmap_vmax = da_ortho.quantile(0.99)
            elif 'denoise' in varname:
                # because denoised data is usually smaller than original values
                #   we set the maximum value as vmax for checking plumes
                cmap = 'plasma'
                vmin = da_ortho.mean()
                cmap_vmax = da_ortho.quantile(0.99)

                # crop cmap_max to max(vmax, vmin+1)
                if vmax is not None:
                    if vmax <= cmap_vmax:
                        # set vmax as user defined value, if cmap_vmax is larger.
                        cmap_vmax = vmax
                if cmap_vmax < vmin:
                    # if the maximum value is negative, hard-code to 1
                    cmap_vmax = vmin + 1
            elif 'mask' in varname:
                # plot a priori plume mask
                cmap = 'tab20c'
                vmin = 0
                cmap_vmax = None

                # pick mask with more than 5 pixels
                da_count = da_ortho.groupby(da_ortho).count()
                da_count = da_count.where(da_count > 5).dropna(varname)
                count_unique = da_count.coords[varname].values
                da_ortho = da_ortho.where(np.isin(da_ortho, count_unique, 0))

                # remove background mask (0)
                da_ortho = da_ortho.where(da_ortho > 0).squeeze()
            else:
                # set vmax for species automatically
                #   Note that vmax of radiance and denoised variables are set based on percentile values above
                if vmax is None:
                    if 'ch4' in varname:
                        # hard-code vmax
                        if da_ortho.attrs['units'] == 'ppb':
                            cmap_vmax = 300
                        elif da_ortho.attrs['units'] == 'ppm':
                            cmap_vmax = 0.3
                        elif da_ortho.attrs['units'] == 'ppm m':
                            cmap_vmax = 0.3/1.25e-4
                        elif da_ortho.attrs['units'] == 'umol m-2':
                            cmap_vmax = 300/2900*1e6
                    elif 'co2' in varname:
                        # hard-code vmax
                        if da_ortho.attrs['units'] == 'ppb':
                            cmap_vmax = 1e4
                        elif da_ortho.attrs['units'] == 'ppm':
                            cmap_vmax = 10
                        elif da_ortho.attrs['units'] == 'ppm m':
                            cmap_vmax = 10/1.25e-4
                        elif da_ortho.attrs['units'] == 'umol m-2':
                            cmap_vmax = 1e4/2900*1e6
                    elif 'no2' in varname:
                        # hard-code vmax
                        if da_ortho.attrs['units'] == 'ppb':
                            cmap_vmax = 80*1e-6*2900
                        elif da_ortho.attrs['units'] == 'ppm':
                            cmap_vmax = 80*1e-6*2900/1000
                        elif da_ortho.attrs['units'] == 'ppm m':
                            cmap_vmax = 80*1e-6*2900/1000/1.25e-4
                        elif da_ortho.attrs['units'] == 'umol m-2':
                            cmap_vmax = 80
                    else:
                        raise ValueError(f"{varname} is not supported for auto colormap. Please check and add it here.")
                else:
                    cmap_vmax = vmax

                # set the vmax according to user's input
                #   the variable should be trace gas enhancement
                cmap = 'plasma'
                vmin = 0

                # in case the maximum value is negative
                if cmap_vmax < vmin:
                    cmap_vmax = vmin + 1

            fig, ax = plt.subplots(subplot_kw=dict(projection=self._get_cartopy_crs_from_epsg(out_epsg)))

            # because we use longitude and latitude, we need to specify the transform.
            input_crs = self._get_cartopy_crs_from_epsg(4326)
            ax.pcolormesh(self.ds.longitude, self.ds.latitude, da_ortho, vmin=vmin, vmax=cmap_vmax, cmap=cmap,
                          transform=input_crs, antialiased=True)

            # turn off axis
            ax.axis('off')

            # set png filename
            #   hard code for renaming EMIT RAD filename
            if export_dir is None:
                export_dir = os.path.dirname(self.filename)
                basename = os.path.basename(self.filename)
                output_png = Path(os.path.join(export_dir, basename.replace('.', f'_{varname}.')).replace('_RAD', '').replace('L1', 'L2')).with_suffix('.png')
            else:
                output_png = Path(os.path.join(export_dir,
                                               os.path.basename(self.filename).replace('.', f'_{varname}.')
                                               .replace('_RAD', '').replace('L1', 'L2'))).with_suffix('.png')

            # append pre-suffix
            output_png = str(output_png).replace('.png', f'{pre_suffix}.png')

            # delete pads and remove edges
            fig.savefig(output_png, bbox_inches='tight', pad_inches=0.0, edgecolor=None, transparent=True, dpi=1000)

        # calculate the bounds
        #   we need to use bounds for image overlay on folium map
        extent_4326 = ax.get_extent(crs=ccrs.PlateCarree())

        self.img_bounds = [[extent_4326[2], extent_4326[0]], [extent_4326[3], extent_4326[1]]]

        # clean vars
        del da_ortho, fig, ax
        gc.collect()

    def plot(self, out_epsg=3857, vmax=None, show_layers=None, opacities=None,
             marker=None, df_marker=None, export_dir=None, draw_polygon=True,
             pre_suffix=''):
        """Plot data, export to png files, plot folium map, and export to html files.

        Parameters
        ----------
        out_epsg : int
            EPSG code of the output projection (3857 is the proj of folium Map).
        vmax : float
            The cmap vmax for plotting species (unit is as same as species variable).
        show_layers : bool list
            Whether the layers will be shown on opening (the length should be as same as ``varnames``).
        opacities : float list
            The opacities of layer (the length should be as same as ``varnames``).
        marker : list
            The coords [lat, lon] for a yellow circle marker.
        df_marker : DataFrame
            The DataFrame (columns: latitude, longitude) for adding blue circle markers.
        export_dir : str
            The directory to save plotted images (Default: the same path as filename attrs).
        draw_polygon : bool
            Whether plot the scene boundary polygon (Default: True).
        pre_suffix : str
            The suffix added to the png and html filename (Default: ""). 
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
            self.opacities = [0.8]*len(self.varnames)
        else:
            self.opacities = opacities
        if len(self.opacities) != len(self.varnames):
            raise ValueError(
                f"opacities's length ({len(self.opacities)}) should be as same as varnames's length ({len(self.varnames)})")

        # plot png images
        self.plot_png(out_epsg=out_epsg, vmax=vmax, export_dir=export_dir, pre_suffix=pre_suffix)

        # get the swath polygon from area boundary
        hull = ConvexHull(self.swath.boundary().vertices)
        lonlatPoly = geometry.Polygon(hull.points[hull.vertices])
        self.gjs = geojson.Feature(geometry=lonlatPoly, properties={})

        # overlay images on foilum map
        self.marker = marker
        self.df_marker = df_marker
        self.export_dir = export_dir
        self.draw_polygon = draw_polygon
        self.plot_folium(pre_suffix)

        # delete vars
        del self.gjs
        gc.collect()

    def plot_wind(self, source='ERA5', position='bottomright'):
        """Plot the wind as html element.

        Parameters
        ----------
        source : str
            wind source: "ERA5" or "GEOS-FP".
        position : str
            the position to add the wind marker.
        """
        # read data
        u10 = self.ds['u10'].sel(source=source).mean().item()
        v10 = self.ds['v10'].sel(source=source).mean().item()

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

    def plot_folium(self, pre_suffix=''):
        """Overlay png images on folium map.

        Parameters
        ----------
        pre_suffix : str
            The suffix added to the png and html filename. 
        """
        # get the time string
        self.time_str = self.ds[self.varnames[0]].attrs['start_time']
        sensor_name = self.ds[self.varnames[0]].attrs['sensor']

        # add swath poly
        if self.draw_polygon:
            fg_poly = folium.FeatureGroup(name='Swath polygons')
            self.map.add_child(fg_poly)
            style = {'fillColor': '#00000000', 'color': 'dodgerblue'}
            folium.GeoJson(self.gjs,
                           control=False,
                           tooltip=self.time_str,
                           zoom_on_click=False,
                           style_function=lambda x: style,
                           # highlight_function= lambda feat: {'fillColor': 'blue'},
                           ).add_to(fg_poly)

        # add the group which controls all subgroups (varnames)
        self.fg = folium.FeatureGroup(name=f'{self.time_str} group ({sensor_name})')
        self.map.add_child(self.fg)

        for index, varname in enumerate(self.varnames):
            layer_name = f'{self.time_str} | {varname}'
            gplot = FeatureGroupSubGroup(self.fg, layer_name, show=self.show_layers[index])
            self.map.add_child(gplot)

            if self.export_dir is None:
                export_dir = os.path.dirname(self.filename)
                basename = os.path.basename(self.filename)
                output_png = Path(os.path.join(export_dir, basename.replace('.', f'_{varname}.')).replace('_RAD', '').replace('L1', 'L2')).with_suffix('.png')
            else:
                output_png = Path(os.path.join(self.export_dir,
                                               os.path.basename(self.filename)
                                               .replace('.', f'_{varname}.')
                                               .replace('_RAD', '').replace('L1', 'L2'))).with_suffix('.png')
            # append pre-suffix
            output_png = str(output_png).replace('.png', f'{pre_suffix}.png')

            # plot 2D trace gas field
            raster = folium.raster_layers.ImageOverlay(image=str(output_png),
                                                       opacity=self.opacities[index],
                                                       bounds=self.img_bounds,
                                                       name=layer_name,
                                                       )
            raster.add_to(gplot)

        # plot wind
        try:
            self.plot_wind(source='ERA5')
            self.plot_wind(source='GEOS-FP')
        except Exception as e:
            LOG.warning(e)
            LOG.warning('Wind data is not available in this file. Skip plotting wind arrows.')

        if self.marker is not None:
            # add a yellow circle marker (lat, lon)
            folium.Circle([self.marker[0], self.marker[1]], radius=100, color='yellow').add_to(self.map)

        if self.df_marker is not None:
            # loop dataframe to add CircleMarker with popup table
            for (index, row) in self.df_marker.iterrows():
                html = self.df_marker.iloc[index]\
                    .to_frame().to_html(classes="table table-striped table-hover table-condensed table-responsive")
                popup = folium.Popup(html, max_width=500)

                folium.CircleMarker(location=[row.loc['latitude'], row.loc['longitude']], radius=6,
                                    color='dodgerblue', fill_color='dodgerblue', fill_opacity=0.3,
                                    popup=popup,
                                    ).add_to(self.map)

    def export(self, savename=None, pre_suffix=''):
        """Export plotted folium map to html file.

        Parameters
        ----------
        savename : str
            The exported html filename.
        pre_suffix : str
            The suffix added to the html filename. 
        """
        layer_control = folium.LayerControl(collapsed=False, position='topleft', draggable=True)
        self.map.add_child(layer_control)

        if savename is None:
            savename = str(Path(self.filename.replace('_RAD', '').replace('L1', 'L2')).with_suffix('.html'))

        # append pre-suffix
        savename = savename.replace('.html', f'{pre_suffix}.html')

        LOG.info(
            f'Export folium map to {savename}')
        self.map.save(savename)
