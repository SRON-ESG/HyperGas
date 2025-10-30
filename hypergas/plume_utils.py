#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Some utils used for creating plume mask and gas emission rates."""

import base64
import gc
import logging
import math
import os
import warnings

import geopandas as gpd
import numpy as np
import pyresample
import xarray as xr
from branca.element import MacroElement
from jinja2 import Template
from pyresample.geometry import SwathDefinition
from scipy import ndimage
from shapely.geometry import Point, Polygon

from hypergas.folium_map import Map
from hypergas.landmask import Land_mask

LOG = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# calculate IME (kg m-2)
mass = {'ch4': 16.04e-3, 'co2': 44.01e-3}  # molar mass [kg/mol]
mass_dry_air = 28.964e-3  # molas mass dry air [kg/mol]
grav = 9.8  # gravity (m s-2)


class CustomControl(MacroElement):
    """Put any HTML on the map as a Leaflet Control.

    Adopted from https://github.com/python-visualization/folium/pull/1662.
    """

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        L.Control.CustomControl = L.Control.extend({
            onAdd: function(map) {
                let div = L.DomUtil.create('div');
                div.innerHTML = `{{ this.html }}`;
                return div;
            },
            onRemove: function(map) {
                // Nothing to do here
            }
        });
        L.control.customControl = function(opts) {
            return new L.Control.CustomControl(opts);
        }
        L.control.customControl(
            { position: "{{ this.position }}" }
        ).addTo({{ this._parent.get_name() }});
        {% endmacro %}
    """
    )

    def __init__(self, html, position="bottomleft"):
        def escape_backticks(text):
            """Escape backticks so text can be used in a JS template."""
            import re

            return re.sub(r"(?<!\\)`", r"\`", text)

        super().__init__()
        self.html = escape_backticks(html)
        self.position = position


def plot_wind(m, wdir, wspd, arrow_img='./imgs/arrow.png'):
    """Plot the wind arrow png by rotate the north arrow."""
    with open(arrow_img, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

    html = f"""
    <img src="data:image/png;base64,{image_base64}" style="transform:rotate({wdir+180}deg);">
    <h5 style="color:white;">{wspd:.1f} m/s</h5>
    """

    widget = CustomControl(html, position='bottomright')

    widget.add_to(m)


def get_wind_azimuth(u, v):
    """
    Calculate the wind azimuth angle based on the horizontal wind components.

    The function computes the azimuth (direction) of the wind vector from 
    its u (east-west) and v (north-south) components. The azimuth is measured 
    in radians and degrees, following meteorological convention.

    Parameters
    ----------
    u : float
        Zonal wind component (positive toward the east).
    v : float
        Meridional wind component (positive toward the north).

    Returns
    -------
    azim_rad : float
        Wind azimuth in radians, measured clockwise from the north.
    azim : float
        Wind azimuth in degrees, measured clockwise from the north.
    """
    if (u > 0):
        azim_rad = (np.pi)/2. - np.arctan(v/u)
    elif (u == 0.):
        if (v > 0.):
            azim_rad = 0.
        elif (v == 0.):
            azim_rad = 0.  # arbitrary
        elif (v < 0.):
            azim_rad = np.pi
    elif (u < 0.):
        azim_rad = 3*np.pi/2. + np.arctan(-v/u)

    azim = azim_rad*180./np.pi

    return azim_rad, azim


def _azimuth(point1, point2):
    '''azimuth between 2 points (interval 0 - 180)
    # https://stackoverflow.com/a/66118219/7347925
    '''
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180


def _dist(a, b):
    '''distance between points'''
    return math.hypot(b[0] - a[0], b[1] - a[1])


def azimuth_mrr(mrr):
    '''Azimuth of plume's `minimum_rotated_rectangle <https://geopandas.org/en/latest/docs/reference/api/geopandas.GeoSeries.minimum_rotated_rectangle.html>`_.'''
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])

    return az


def get_index_nearest(lons, lats, lon_target, lat_target):
    """Get the pixel index closest to the target.

    Parameters
    ----------
    lons : :class:`numpy.ndarray`
        2D longitudes of pixels.
    lats : :class:`numpy.ndarray`
        2D latitudes of pixels.
    lon_target : float
        The longitude of target.
    lat_target : float
        The latitude of target.

    Returns
    -------
    y_nearest : int
        The nearest y index.
    x_nearest : int
        The nearest x index.
    """
    # define the areas for data and source point
    area_source = SwathDefinition(lons=lons, lats=lats)
    area_target = SwathDefinition(lons=np.array([lon_target]), lats=np.array([lat_target]))

    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(
        source_geo_def=area_source, target_geo_def=area_target, radius_of_influence=50,
        neighbours=1)

    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute the 2D grid indices:
    y_target, x_target = np.unravel_index(index_array, area_source.shape)

    return y_target[0], x_target[0]


def target_inside_mask(ds, gas_mask_varname, y_target, x_target, lon_target, lat_target):
    """Move the target if it is not inside the mask.

    Parameters
    ----------
    ds : :class:`~xarray.Dataset`
        The dataset includes the gas field and geolocations.
    gas_mask_varname : str
        The mask name of gas.
    y_target : int
        The y index of target.
    x_target : int
        The x index of target.
    lon_target : float
        The longitude of target.
    lat_target : float
        The latitude of target.

    Returns
    -------
        y_target : int
        x_target : int
            If the input target index is not included in the mask, it will be updated using the mask pixel closest to the target.
    """
    if ds[gas_mask_varname][y_target, x_target] == 0:
        LOG.info('Picking the nearest mask pixel because the target is in the background.')
        lon_mask = ds['longitude'].where(ds[gas_mask_varname] > 0).data.flatten()
        lat_mask = ds['latitude'].where(ds[gas_mask_varname] > 0).data.flatten()
        lon_mask = lon_mask[~np.isnan(lon_mask)]
        lat_mask = lat_mask[~np.isnan(lat_mask)]

        # Get the closest mask pixel location
        min_index = gpd.points_from_xy(lon_mask, lat_mask).distance(Point(lon_target, lat_target)).argmin()
        y_target, x_target = get_index_nearest(ds['longitude'], ds['latitude'],
                                               lon_mask[min_index], lat_mask[min_index])

    return y_target, x_target


def plot_mask(filename, ds, gas, mask, lon_target, lat_target, pick_plume_name, only_plume=True):
    """Plot masked data and export to L3 HTML file.

    Parameters
    ----------
    filename : str
        The L2 HTML file name.
    ds : :class:`~xarray.Dataset`
        The dataset includes the gas field and RGB data.
    gas : str
        The gas name to be plotted.
    mask : :class:`numpy.ndarray`
        The plume boolean mask.
    lon_target : float
        The longitude of target.
    lat_target : float
        The latitude of target.
    pick_plume_name : str
        Which plume to be plotted (e.g., "plume0", "plume1", ...).

    Returns
    -------
    plume_html_filename : str
        The plume html filename.
    """
    # read gas data
    da_gas = ds[gas]

    # get masked plume data
    if mask.all():
        da_gas_mask = da_gas
    else:
        da_gas_mask = da_gas.where(xr.DataArray(mask, dims=list(da_gas.dims)))

    ds[f'{gas}_plume'] = da_gas_mask

    if only_plume:
        # only plot plume for quick check
        m = Map(ds, varnames=[f'{gas}_plume'], center_map=[lat_target, lon_target])
        m.initialize()
        m.plot(show_layers=[True], opacities=[0.9],
               marker=[lat_target, lon_target], export_dir=os.path.dirname(filename), draw_polygon=False)
    else:
        # plot all important data
        m = Map(ds, varnames=['rgb', gas, f'{gas}_comb', f'{gas}_comb_denoise',
                f'{gas}_plume'], center_map=[lat_target, lon_target])
        m.initialize()
        m.plot(show_layers=[False, False, False, False, True], opacities=[0.9, 0.8, 0.8, 0.8, 0.8],
               marker=[lat_target, lon_target], export_dir=os.path.dirname(filename), draw_polygon=False)

    # export to html file
    if 'plume' in os.path.basename(filename):
        if pick_plume_name == 'plume0':
            plume_html_filename = filename
        else:
            # rename the filenames if there are more than one plume in the file
            plume_html_filename = filename.replace('plume0', pick_plume_name)
    else:
        plume_html_filename = filename.replace('L2', 'L3').replace('.html', f'_{pick_plume_name}.html')

    m.export(plume_html_filename)

    return plume_html_filename


def select_connect_masks(masks, y_target, x_target, az_max=30, dist_max=180):
    """Select connected masks by dilation and limit the minimum rectangle angle difference

    Parameters
    ----------
    masks : :class:`~xarray.DataArray`
        2D a priori mask from L2 data.
    y_target : float
        yindex of source target.
    x_target : float
        xindex of source target.
    az_max : float
        maximum of azimuth of minimum rotated rectangle. (Default: 30).
    dist_max : float
        maximum of dilation distance (meter).

    Returns
    -------
    Connected plume mask : :class:`~xarray.DataArray`.
    """

    # get the source label of original mask
    mask_target = masks[y_target, x_target].item()

    # dilation mask
    struct = ndimage.generate_binary_structure(2, 2)
    dxy = abs(masks.coords['y'].diff('y')[0])
    if dxy == 1:
        # the projection is not UTM but EPSG:4326
        #   we use the 2d lat and lon array to calculate the distance
        R = 6371e3  # meters
        lat_1 = masks.coords['latitude'][0, 0]
        lat_2 = masks.coords['latitude'][0, 1]
        lon_1 = masks.coords['longitude'][0, 0]
        lon_2 = masks.coords['longitude'][0, 1]

        phi_1 = lat_1 * np.pi / 180
        phi_2 = lat_2 * np.pi / 180
        delta_phi = (lat_2 - lat_1) * np.pi / 180
        delta_lambda = (lon_2 - lon_1) * np.pi / 180

        a = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi_1) * \
            np.cos(phi_2) * np.sin(delta_lambda / 2) * np.sin(delta_lambda / 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        dxy = R * c  # meters

    niter = int(dist_max/dxy)
    if niter > 0:
        masks_dilation = masks.copy(deep=True, data=ndimage.binary_dilation(
            masks.fillna(0), iterations=niter, structure=struct))
    else:
        masks_dilation = masks.copy(deep=True, data=masks.fillna(0))

    # Label connected components in the dilated array
    labeled_array, num_features = ndimage.label(masks_dilation)
    masks_dilation = masks.copy(deep=True, data=labeled_array).where(masks.notnull())

    # get the dilation mask which contains mask including the target
    mask_dilation_target = masks_dilation[y_target, x_target].values
    mask_dilation_target = masks_dilation.where(masks_dilation == mask_dilation_target)

    # mask in the dilation mask
    masks_in_dilation = masks.where((masks > 0) & (mask_dilation_target > 0))

    # unique mask labels within the dilation mask
    connect_labels = np.unique(masks_in_dilation.data.flatten())

    # create mask polygons
    df_mask = masks.to_dataframe().reset_index()
    df_mask = df_mask[df_mask[masks.name] > 0]
    gdf_polygon = gpd.GeoDataFrame(geometry=df_mask.groupby(masks.name)
                                   .apply(lambda g: Polygon(gpd.points_from_xy(g['longitude'], g['latitude'])))
                                   )

    # calculate mrr and azimuth angle
    gdf_polygon['mrrs'] = gdf_polygon.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    gdf_polygon['az'] = gdf_polygon['mrrs'].apply(azimuth_mrr)

    # get the polygons inside the dilation mask which includes the target mask
    gdf_polygon_connect = gdf_polygon[gdf_polygon.index.isin(connect_labels)]

    if len(gdf_polygon_connect) > 1:
        # calculate polygon distance
        gdf_polygon_connect['distance'] = gdf_polygon_connect.geometry.apply(
            lambda g: gdf_polygon_connect[gdf_polygon_connect.index == mask_target]['geometry'].distance(g, align=False))

        # sort masks by distance
        gdf_polygon_connect.sort_values('distance', inplace=True)

        # calcualte differences of az
        gdf_polygon_connect.loc[:, 'az_diff'] = gdf_polygon_connect['az'].diff().abs().fillna(0)

        index_name = gdf_polygon_connect.index.name
        gdf_polygon_connect = gdf_polygon_connect.reset_index()

        # Iterate through the DataFrame to drop rows where az_diff is higher than az_max
        index = 0
        while index < len(gdf_polygon_connect) - 1:
            if (gdf_polygon_connect['az_diff'].iloc[index + 1] > az_max) and (gdf_polygon_connect['distance'].iloc[index+1] > 0):
                gdf_polygon_connect = gdf_polygon_connect.drop(index + 1)
                # drop the next row and recheck
                gdf_polygon_connect = gdf_polygon_connect.reset_index(drop=True)
                gdf_polygon_connect['az_diff'] = gdf_polygon_connect['az'].diff().abs().fillna(0)
            else:
                index += 1

        # Set the index back to the original index values
        gdf_polygon_connect = gdf_polygon_connect.set_index(index_name)

    # get final mask
    mask = masks_in_dilation.isin(gdf_polygon_connect.index)

    return mask


def a_priori_mask_data(ds, gas, lon_target, lat_target,
                       pick_plume_name, wind_source,
                       only_plume=True, az_max=30, dist_max=180,
                       filename=None
                       ):
    '''Read a priori plume masks and connect them by conditions.

    Parameters
    ----------
    ds : :class:`~xarray.Dataset`
        L2 dataset.
    gas : str
        The gas field to be masked.
    lon_target : float
        The longitude of plume source.
    lat_target : float
        The latitude of plume source.
    pick_plume_name : str
        The plume name (plume0, plume1, ....).
    wind_source : str
        "ERA5" or "GEOS-FP".
    az_max : float
        Maximum of azimuth of minimum rotated rectangle. (Default: 30).
    dist_max : float
        Maximum of dilation distance (meter).
    filename : str
        The L2 HTML filename.

    Returns
    -------
    mask : :class:`~xarray.DataArray`
        The Boolean mask of pixels.
    lon_mask : :class:`~xarray.DataArray`
        Plume longitude.
    lat_mask : :class:`~xarray.DataArray`
        Plume latitude.
    lon_target : float
        Longitude of target.
    lat_target : float
        Latitude of target.
    plume_html_filename : str
        Exported plume html filename.
    '''
    LOG.info('Selecting connected plume masks')
    # get the y/x index of the source location
    y_target, x_target = get_index_nearest(ds['longitude'], ds['latitude'], lon_target, lat_target)

    # check if target is inside the masks
    y_target, x_target = target_inside_mask(ds, f'{gas}_mask', y_target, x_target, lon_target, lat_target)

    # update target
    lon_target = ds['longitude'].isel(y=y_target, x=x_target).item()
    lat_target = ds['latitude'].isel(y=y_target, x=x_target).item()

    # select connected masks
    mask = select_connect_masks(ds[f'{gas}_mask'], y_target, x_target, az_max, dist_max)

    # get the masked lon and lat
    lon_mask = xr.DataArray(ds['longitude'], dims=['y', 'x']).where(mask).rename('longitude')
    lat_mask = xr.DataArray(ds['latitude'], dims=['y', 'x']).where(mask).rename('latitude')

    if filename is not None:
        # plot the mask (png and html)
        plume_html_filename = plot_mask(filename, ds, gas, mask, lon_target, lat_target,
                                        pick_plume_name, only_plume=only_plume)

        return mask, lon_mask, lat_mask, lon_target, lat_target, plume_html_filename
    else:
        return mask, lon_mask, lat_mask, lon_target, lat_target


def crop_to_valid_region(da, y_target, x_target, data_crop_length, pixel_res):
    """Crop a DataArray around a target location to ensure a square result, centered on the target, 
    and adjusted to fit within bounds if necessary.

    Parameters
    ----------
    da : :class:`~xarray.DataArray`
        The to be cropped DataArray.
    y_target : int
        The yindex of target.
    x_target : int
        The xindex of target.
    data_crop_length : float
        The crop radius (m).
    pixel_res : float
        The pixel resolution (m).

    Returns
    -------
        The cropping index : ymin, ymax, xmin, xmax.
    """
    crop_pixels = data_crop_length//pixel_res

    y_len = da.sizes['y']
    x_len = da.sizes['x']

    ymin = max(0, y_target - crop_pixels)
    xmin = max(0, x_target - crop_pixels)
    ymax = min(y_len, y_target + crop_pixels)
    xmax = min(x_len, x_target + crop_pixels)

    crop_y = min(ymax - y_target, y_target - ymin)
    crop_x = min(xmax - x_target, x_target - xmin)
    crop_pixels = min(crop_y, crop_x)//2
    ymin = y_target - crop_pixels
    xmin = x_target - crop_pixels
    ymax = y_target + crop_pixels
    xmax = x_target + crop_pixels

    return ymin, ymax, xmin, xmax


def cm_mask_data(ds, gas, lon_target, lat_target,
                 data_crop_length=2500, limit_crop_length=1000, limit_percentile=90):
    """Create plume mask using `Carbon Mapper's method <https://assets.carbonmapper.org/documents/L3_L4%20Algorithm%20Theoretical%20Basis%20Document_formatted_10-24-24.pdf>`_.

    Parameters
    ----------
    data_crop_length : float
        The length (m) for cropping data.
    limit_crop_length: float
        The length (m) for calculating the plume enhancement threshold.
    limit_percentile : float
        The percentile (%) for the plume enhancement threshold.
    """
    LOG.info('Creating the CM plume mask')

    # get the y/x index of the source location
    y_target, x_target = get_index_nearest(ds['longitude'], ds['latitude'], lon_target, lat_target)
    pixel_res = int(abs(ds.coords['y'].diff(dim='y').mean(dim='y')))

    # Step 1: Crop the data to data_crop_length (m) around the origin
    ymin_crop, ymax_crop, xmin_crop, xmax_crop = crop_to_valid_region(ds[gas], y_target, x_target, data_crop_length, pixel_res)
    cropped_data = ds[gas].isel(y=slice(ymin_crop, ymax_crop), x=slice(xmin_crop, xmax_crop))

    # Step 2: Set concentration threshold by <limit_percentile> percent (<limit_crop_length> m around)
    crop_origin_y, crop_origin_x = get_index_nearest(cropped_data['longitude'],
                                                     cropped_data['latitude'],
                                                     lon_target,
                                                     lat_target,
                                                     )

    ymin, ymax, xmin, xmax = crop_to_valid_region(cropped_data, crop_origin_y, crop_origin_x, limit_crop_length, pixel_res)
    small_crop = cropped_data.isel(y=slice(ymin, ymax), x=slice(xmin, xmax))
    threshold = np.nanpercentile(small_crop, limit_percentile).item()

    # Create binary mask
    mask = (cropped_data > threshold).astype(int)

    # Step 3: Group connected pixels
    labeled, num_features = ndimage.label(mask)

    # Remove small clusters (less than 5 pixels)
    for i in np.unique(labeled):
        if np.sum(labeled == i) < 5:
            labeled[labeled == i] = 0

    # Step 4: Enforce proximity metric
    x_coords, y_coords = np.meshgrid(range(cropped_data.sizes['x']), range(cropped_data.sizes['y']))
    distance = np.sqrt((y_coords-cropped_data.sizes['y']/2)**2 + (x_coords-cropped_data.sizes['x']/2)**2)  # unit: pixel

    for i in np.unique(labeled):
        if np.min(distance[labeled == i]) > 15:
            labeled[labeled == i] = 0

    # Step 5: Create final binary mask
    # 0: background, >0: labels
    final_mask = xr.full_like(ds[gas], 0)
    final_mask.isel(x=slice(xmin_crop, xmax_crop), y=slice(ymin_crop, ymax_crop))[:] = (labeled > 0).astype(int)

    # clean attrs
    final_mask = final_mask.rename('cm_mask')
    final_mask.attrs = ''
    final_mask.attrs['description'] = 'Carbon Mapper plume mask using v2 method'

    return final_mask, threshold

