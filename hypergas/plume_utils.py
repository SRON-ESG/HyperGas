#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Some utils used for creating plume mask and gas emission rates"""

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
    Adopted from https://github.com/python-visualization/folium/pull/1662

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
    """Plot wind by rotate the north arrow"""
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


def conish_2d(x, y, xc, yc, r):
    xr = (x-xc)*np.cos(r) + (y-yc)*np.sin(r)
    yr = -(x-xc)*np.sin(r) + (y-yc)*np.cos(r)

    max_xr = np.max(np.abs(xr))
    max_yr = np.max(np.abs(yr))

    t = np.arctan2(yr, xr)

    scale_dist = max(max_xr, max_yr)
    orig_shift = -scale_dist/10.
    t = np.arctan2(yr, xr-orig_shift)
    sig = scale_dist/2.

    open_ang = np.pi/3.5

    out = np.exp(-((xr-orig_shift)**2 + (yr)**2)/sig**2 - (t/open_ang)**2)

    return out


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
    '''azimuth of minimum_rotated_rectangle'''
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])

    if axis1 <= axis2:
        az = _azimuth(bbox[0], bbox[1])
    else:
        az = _azimuth(bbox[0], bbox[3])

    return az


def get_index_nearest(lons, lats, lon_target, lat_target):
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
    '''Make sure target is not inside the masks'''
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


def plume_mask(ds, gas, lon_target, lat_target, plume_varname='comb_denoise',
               wind_source='ERA5', wind_weights=True, land_only=False, land_mask_source='GSHHS',
               niter=1, quantile_value=0.98, size_median=3, sigma_guass=2):
    """Get the plume mask based on matched filter results

    Steps:
        - apply wind weights (optional)
        - apply land mask
        - set 1 to pixel values larger than quantile_value
        - apply median_filter, gaussian_filter, and dilation to get a smoothed mask
        - assign labels and get labelled region where the source point is inside
        - pick mask pixels where enhancement values are positive
        - erosion the mask and propagate again
        - fill holes of mask

    Args:
        gas (str): the gas field to be masked
        lon_target (float): source longitude for picking plume dilation mask
        lat_target (float): source latitude for picking plume dilation mask
        plume_varname (str): the variable used to create plume mask (Default:comb_denoise)
        wind_source (str): 'ERA5' or 'GEOS-FP'
        wind_weights (boolean): whether apply the wind weights to gas fields
        n_iter (int): number of iterations for dilations
        quantile_value (float): the lower quantile limit of enhancements are included in the mask
        size_median (int): size for median filter
        sigma_guass (int): sigma for guassian filter
    """
    # get lon and lat
    lon = ds['longitude']
    lat = ds['latitude']

    # get the variable for plume masking
    #   default: <gas>_comb_denoise
    if plume_varname == 'comb_denoise':
        varname = f'{gas}_comb_denoise'
    elif plume_varname == 'denoise':
        varname = f'{gas}_denoise'
    else:
        varname = gas

    da_gas = getattr(ds, varname, ds[gas])

    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute the 2D grid indices:
    y_target, x_target = get_index_nearest(ds['longitude'], ds['latitude'], lon_target, lat_target)

    if wind_weights:
        # calculate wind angle
        angle_wind_rad, angle_wind = get_wind_azimuth(ds['u10'].sel(source=wind_source).isel(y=y_target, x=x_target).item(),
                                                      ds['v10'].sel(source=wind_source).isel(y=y_target, x=x_target).item())
        weights = conish_2d(lon, lat, lon_target, lat_target, np.pi/2. - angle_wind_rad)
        da_gas_weights = da_gas * weights
    else:
        da_gas_weights = da_gas

    # set oceanic pixel values to nan
    if land_only:
        segmentation = Land_mask(lon.data, lat.data, land_mask_source)
        if segmentation.isel(y=y_target, x=x_target).values != 0:
            # because sometimes cartopy land mask is not accurate
            #  we need to make sure the source point is not on the ocean
            da_gas_weights = da_gas_weights.where(segmentation)
        else:
            LOG.warning('The source point is over ocean! Please make sure you choose the suitable cfeature.')

    # set init mask by the quantile value
    mask_buf = np.zeros(np.shape(da_gas))
    mask_buf[da_gas_weights >= da_gas_weights.quantile(quantile_value)] = 1.

    # blur the mask
    median_blurred = ndimage.median_filter(mask_buf, size_median)
    gaussmed_blurred = ndimage.gaussian_filter(median_blurred, sigma_guass)
    mask = np.zeros(np.shape(da_gas))
    mask[gaussmed_blurred > 0.05] = 1.

    # dilation
    if niter >= 1:
        mask = ndimage.binary_dilation(mask, iterations=niter)

    # assign labels by 3x3
    labeled_mask, nfeatures = ndimage.label(mask, structure=np.ones((3, 3)))

    # get the labeled mask where the target is inside
    feature_label = labeled_mask[y_target, x_target]
    mask = labeled_mask == feature_label

    # create the positive mask
    fg_mask = da_gas_weights.where(mask) > 0
    # erosion the mask and propagate again
    eroded_square = ndimage.binary_erosion(fg_mask)
    reconstruction = ndimage.binary_propagation(eroded_square, mask=fg_mask)
    # fill the negative values inside
    mask = ndimage.binary_fill_holes(reconstruction)

    # get the masked lon and lat
    lon_mask = xr.DataArray(lon, dims=['y', 'x']).where(mask).rename('longitude')
    lat_mask = xr.DataArray(lat, dims=['y', 'x']).where(mask).rename('latitude')

    return mask, lon_mask, lat_mask


def plot_mask(filename, ds, gas, mask, lon_target, lat_target, pick_plume_name, only_plume=True):
    """Plot masked data"""
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


def mask_data(filename, ds, gas, lon_target, lat_target, pick_plume_name, plume_varname,
              wind_source, wind_weights, land_only, land_mask_source,
              niter, size_median, sigma_guass, quantile_value, only_plume):
    '''Generate and plot masked plume data

    Args:
        filename (str): input filename
        ds (Dataset): L2 data
        gas (str): the gas field to be masked
        lon_target (float): The longitude of plume source
        lat_target (float): The latitude of plume source
        pick_plume_name (str): the plume name (plume0, plume1, ....)
        plume_varname (str): the variable used to create plume mask (Default:comb_denoise)
        wind_source (str): 'ERA5' or 'GEOS-FP'
        wind_weights (boolean): whether apply the wind weights to gas fields
        n_iter (int): number of iterations for dilations
        quantile_value (float): the lower quantile limit of enhancements are included in the mask
        size_median (int): size for median filter
        sigma_guass (int): sigma for guassian filter

    Return:
        mask (DataArray): Boolean mask (pixel)
        lon_mask (DataArray): plume longitude
        lat_mask (DataArray): plume latitude
        plume_html_filename (str): exported plume html filename
        '''
    # create the plume mask
    mask, lon_mask, lat_mask = plume_mask(ds, gas, lon_target, lat_target, plume_varname=plume_varname,
                                          wind_source=wind_source, wind_weights=wind_weights,
                                          land_only=land_only, land_mask_source=land_mask_source,
                                          niter=niter, size_median=size_median, sigma_guass=sigma_guass,
                                          quantile_value=quantile_value)

    # plot the mask (png and html)
    plume_html_filename = plot_mask(filename, ds, gas, mask, lon_target, lat_target,
                                    pick_plume_name, only_plume=only_plume)

    return mask, lon_mask, lat_mask, plume_html_filename


def select_connect_masks(masks, y_target, x_target, az_max=30, dist_max=180):
    '''
    Select connected masks by dilation and limit the minimum rectangle angle difference

    Args:
        masks (2D DataArray):
            a priori mask from L2 data
        y_target (float):
            yindex of source target
        x_target (float):
            xindex of source targeta
        az_max (float):
            maximum of azimuth of minimum rotated rectangle. (Default: 30)
        dist_max (float):
            maximum of dilation distance (meter)

    Return:
        plume masks (DataArray)
    '''
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


def a_priori_mask_data(filename, ds, gas, lon_target, lat_target, pick_plume_name,
                       wind_source, land_only, land_mask_source, only_plume, az_max=30, dist_max=180):
    '''Read a priori plume masks and connect them by conditions

    Args:
        filename (str):
            Input filename
        ds (Dataset):
            L2 data
        gas (str):
            The gas field to be masked
        lon_target (float):
            The longitude of plume source
        lat_target (float):
            The latitude of plume source
        pick_plume_name (str):
            The plume name (plume0, plume1, ....)
        wind_source (str):
            'ERA5' or 'GEOS-FP'
        az_max (float):
            Maximum of azimuth of minimum rotated rectangle. (Default: 30)
        dist_max (float):
            Maximum of dilation distance (meter)

    Return:
        mask (DataArray): Boolean mask (pixel)
        lon_mask (DataArray): plume longitude
        lat_mask (DataArray): plume latitude
        lon_target (float): longitude of target
        lat_target (float): latitude of target
        plume_html_filename (str): exported plume html filename
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

    # plot the mask (png and html)
    plume_html_filename = plot_mask(filename, ds, gas, mask, lon_target, lat_target,
                                    pick_plume_name, only_plume=only_plume)

    return mask, lon_mask, lat_mask, lon_target, lat_target, plume_html_filename


def cm_mask_data(ds, gas, lon_target, lat_target):
    LOG.info('Creating the CM plume mask')

    # get the y/x index of the source location
    y_target, x_target = get_index_nearest(ds['longitude'], ds['latitude'], lon_target, lat_target)
    pixel_res = int(abs(ds.coords['y'].diff(dim='y').mean(dim='y')))

    # Step 1: Crop the data to 2.5 km around the origin
    cropped_data = ds[gas].isel(
        y=slice(y_target - 2500//pixel_res, y_target + 2500//pixel_res),
        x=slice(x_target - 2500//pixel_res, x_target + 2500//pixel_res)
    )

    # Step 2: Set concentration threshold by 90th percent (1 km around)
    crop_origin_y, crop_origin_x = get_index_nearest(cropped_data['longitude'],
                                                     cropped_data['latitude'],
                                                     lon_target,
                                                     lat_target,
                                                     )

    small_crop = cropped_data.isel(
        y=slice(crop_origin_y - 1000//pixel_res, crop_origin_y + 1000//pixel_res),
        x=slice(crop_origin_x - 1000//pixel_res, crop_origin_x + 1000//pixel_res)
    )
    threshold = np.percentile(small_crop, 90)

    # Create binary mask
    mask = (cropped_data > threshold).astype(int)

    # Step 3: Group connected pixels
    labeled, num_features = ndimage.label(mask)

    # Remove small clusters (less than 5 pixels)
    for i in np.unique(labeled):
        if np.sum(labeled == i) < 5:
            labeled[labeled == i] = 0

    # Step 4: Enforce proximity metric
    x_coords, y_coords = np.meshgrid(
        range(cropped_data.sizes['x']), range(cropped_data.sizes['y']))
    distance = np.sqrt((y_coords-cropped_data.sizes['y']/2)**2 + (
        x_coords-cropped_data.sizes['x']/2)**2)  # unit: pixel

    for i in np.unique(labeled):
        if np.min(distance[labeled == i]) > 15:
            labeled[labeled == i] = 0

    # Step 5: Create final binary mask
    final_mask = (labeled > 0).astype(int)
    final_mask = xr.DataArray(final_mask, dims=['y', 'x'], coords=[
                              cropped_data.y, cropped_data.x])

    # broadcast the plume mask
    final_mask = final_mask.broadcast_like(ds[gas]).fillna(0)
    final_mask = final_mask.rename('cm_mask')
    final_mask.attrs['description'] = 'Carbon Mapper plume mask using v2 method'

    return final_mask


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:
        # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius

    return mask


def ime_radius(gas, da_gas, mask, sp, area):
    gas_mask = da_gas.where(mask)
    unit = gas_mask.attrs['units']

    if unit == 'ppb':
        delta_omega = gas_mask * 1.0e-9 * (mass[gas] / mass_dry_air) * sp / grav
    elif unit == 'ppm':
        delta_omega = gas_mask * 1.0e-6 * (mass[gas] / mass_dry_air) * sp / grav
    elif unit == 'ppm m':
        delta_omega = gas_mask * 7.16e-7
    else:
        raise ValueError(f"Unit '{unit}' is not supported yet. Please add it here.")

    IME = np.nansum(delta_omega * area)

    return IME


def calc_wind_error(wspd, IME, l_eff, alpha):
    """Calculate wind error with random distribution"""
    # Generate U10 distribution
    #   uncertainty = 50%, if wspd <= 3 m/s
    #   uncertainty = 1.5 m/s, if wspd > 3 m/s
    if wspd <= 3:
        sigma = wspd * 0.5
    else:
        sigma = 1.5

    wspd_distribution = np.random.normal(wspd, sigma, size=1000)

    # Calculate Ueff distribution
    u_eff_distribution = alpha['alpha1'] * np.log(wspd_distribution) + \
        alpha['alpha2'] + alpha['alpha3'] * wspd_distribution

    # Calculate Q distribution
    Q_distribution = u_eff_distribution * IME / l_eff

    # Calculate standard deviation of Q distribution
    wind_error = np.nanstd(Q_distribution)

    return wind_error


def calc_wind_error_fetch(wspd, ime_l_mean):
    """Calculate wind error with random distribution for IME-fetch"""
    # Generate U10 distribution
    #   uncertainty = 50%, if wspd <= 2 m/s
    #   uncertainty = 1.5 m/s, if wspd > 2 m/s
    if wspd <= 2:
        sigma = wspd * 0.5
    else:
        sigma = 1.5

    wspd_distribution = np.random.normal(wspd, sigma, size=1000)

    # Calculate Q distribution
    Q_distribution = ime_l_mean * wspd_distribution

    # Calculate standard deviation of Q distribution
    wind_error = np.std(Q_distribution)

    return wind_error


def calc_random_err(gas, da_gas, gas_mask, area, sp):
    """Calculate random error by moving plume around the whole scene"""
    # crop gas field to valid region
    gas_mask_crop = gas_mask.where(~gas_mask.isnull()).dropna(dim='y', how='all').dropna(dim='x', how='all')

    # get the shape of input data and mask
    bkgd_rows, bkgd_cols = gas_mask.shape
    mask_rows, mask_cols = gas_mask_crop.shape

    # Insert plume mask data at a random position
    IME_noplume = []

    while len(IME_noplume) <= 500:
        # Generate random row and column index to place b inside a
        row_idx = np.random.randint(0, bkgd_rows - mask_rows)
        col_idx = np.random.randint(0, bkgd_cols - mask_cols)

        if not np.any(da_gas[row_idx:row_idx+mask_rows, col_idx:col_idx+mask_cols].isnull()):
            gas_bkgd_mask = xr.zeros_like(da_gas)
            gas_bkgd_mask[row_idx:row_idx+mask_rows, col_idx:col_idx+mask_cols] = gas_mask_crop.values
            gas_bkgd_mask = gas_bkgd_mask.fillna(0)
            unit = da_gas.attrs['units']
            if unit == 'ppb':
                IME_noplume.append(da_gas.where(gas_bkgd_mask, drop=True).sum().values *
                                   1.0e-9 * (mass[gas] / mass_dry_air) * sp / grav * area)
            elif unit == 'ppm':
                IME_noplume.append(da_gas.where(gas_bkgd_mask, drop=True).sum().values *
                                   1.0e-6 * (mass[gas] / mass_dry_air) * sp / grav * area)
            elif unit == 'ppm m':
                IME_noplume.append(da_gas.where(gas_bkgd_mask, drop=True).sum().values * 7.16e-7 * area)
            else:
                raise ValueError(f"Unit '{unit}' is not supported yet. Please add it here.")

    std_value = np.array(IME_noplume).std()
    del IME_noplume
    gc.collect()

    return std_value


def calc_calibration_error(wspd, IME, u_eff, l_eff, alpha_replace):
    '''
    Calculate wind calibration error by replacing alphas
    '''
    # Calculate Ueff
    u_eff_replace = alpha_replace['alpha1'] * np.log(wspd) + alpha_replace['alpha2'] + alpha_replace['alpha3'] * wspd

    # Calculate uncertainty
    error = abs(u_eff_replace - u_eff) * IME / l_eff

    return error


def calc_emiss(gas, f_gas_mask, pick_plume_name, alpha_replace, pixel_res=30,
               alpha={'alpha1': 0., 'alpha2': 0.81, 'alpha3': 0.38},
               wind_source='ERA5', wspd=None, land_only=True, land_mask_source='GSHHS'):
    '''Calculate the emission rate (kg/h) using IME method

    Args:
        gas (str): the gas field to be masked
        f_gas_mask: The NetCDF file of mask created by `mask_data` function
        pick_plume_name (str): the plume name (plume0, plume1, ....)
        alpha_replace (dict): alphas of different source type (Point or Area), {'alpha1': ...}
        pixel_res (float): pixel resolution (meter)
        alpha (dict): The coefficients for effective wind (U_eff)
        wspd (float): overwritten wind speed
        land_only (boolean): whether calculate random error only over land

    Return:
        wspd: Mean wind speed (m/s)
        wdir: Mean wind direction (deg)
        wspd_all: List of available wind speed (m/s)
        wdir_all: List of available wind direction (deg)
        wind_source_all: List of wind source (str)
        L_eff: Effctive length (m)
        U_eff: Effective wind speed (m/s)
        Q: Emission rate (kg/h)
        Q_err: STD of Q  (kg/h)
        err_random: random error (kg/h)
        err_wind: wind error (kg/h)
        err_calib: calibration error (kg/h)

    '''
    # read file and pick valid data
    file_original = f_gas_mask.replace('L3', 'L2').replace(f'_{pick_plume_name}.nc', '.nc')
    ds_original = xr.open_dataset(file_original)
    ds = xr.open_dataset(f_gas_mask)

    # get the masked plume data
    gas_mask = ds.dropna(dim='y', how='all').dropna(dim='x', how='all')[gas]

    # area of pixel in m2
    area = pixel_res*pixel_res

    # calculate Leff using the root method in meter
    plume_pixel_num = (~gas_mask.isnull()).sum()
    l_eff = np.sqrt(plume_pixel_num * area).item()

    # calculate IME (kg)
    LOG.info('Calculating IME')
    sp = ds['sp'].mean().item()  # use the mean surface pressure (Pa)
    unit = gas_mask.attrs['units']

    if unit == 'ppb':
        delta_omega = gas_mask * 1.0e-9 * (mass[gas] / mass_dry_air) * sp / grav
    elif unit == 'ppm':
        delta_omega = gas_mask * 1.0e-6 * (mass[gas] / mass_dry_air) * sp / grav
    elif unit == 'ppm m':
        delta_omega = gas_mask * 7.16e-7
    else:
        raise ValueError(f"Unit '{unit}' is not supported yet. Please add it here.")

    IME = np.nansum(delta_omega * area)

    # get wind info
    LOG.info('Calculating wind info')
    u10 = ds['u10'].sel(source=wind_source).item()
    v10 = ds['v10'].sel(source=wind_source).item()
    if wspd is None:
        wspd = np.sqrt(u10**2 + v10**2)
    wdir = (270-np.rad2deg(np.arctan2(v10, u10))) % 360

    # check all wind products
    wind_source_all = list(ds['source'].to_numpy())
    wspd_all = np.sqrt(ds['u10']**2+ds['v10']**2)
    wdir_all = (270-np.rad2deg(np.arctan2(ds['v10'], ds['u10']))) % 360
    wspd_all = list(wspd_all.to_numpy())
    wdir_all = list(wdir_all.to_numpy())

    # effective wind speed
    u_eff = alpha['alpha1'] * np.log(wspd) + alpha['alpha2'] + alpha['alpha3'] * wspd

    # calculate the emission rate (kg/s)
    Q = (u_eff / l_eff * IME)

    # ---- uncertainty ----
    # 1. random
    LOG.info('Calculating random error')
    if land_only:
        # get lon and lat
        lon = ds_original['longitude']
        lat = ds_original['latitude']
        segmentation = Land_mask(lon.data, lat.data, source=land_mask_source)
        ds_original[gas] = ds_original[gas].where(segmentation)
        ds[gas] = ds[gas].where(segmentation)

    IME_std = calc_random_err(gas, ds_original[gas], ds[gas], area, sp)
    err_random = u_eff / l_eff * IME_std

    # 2. wind error
    LOG.info('Calculating wind error')
    err_wind = calc_wind_error(wspd, IME, l_eff, alpha)

    # 3. calibration error
    LOG.info('Calculating calibration error')
    err_calib = calc_calibration_error(wspd, IME, u_eff, l_eff, alpha_replace)

    # sum error
    Q_err = np.sqrt(err_random**2 + err_wind**2 + err_calib**2)

    ds.close()
    ds_original.close()

    return wspd, wdir, wspd_all, wdir_all, wind_source_all, l_eff, u_eff, IME, Q*3600, Q_err*3600, \
        err_random*3600, err_wind*3600, err_calib*3600  # kg/h


def calc_emiss_fetch(gas, f_gas_mask, longitude, latitude, pixel_res=30, wind_source='ERA5', wspd=None):
    """Calculate the emission rate (kg/h) using IME-fetch method

    Args:
        gas (str): the gas field to be masked
        f_gas_mask: The NetCDF file of mask created by `mask_data` function
        longitude (float): longitude of source
        latitude (float): latitude of source
        pixel_res (float): pixel resolution (meter)

    Return:
        Q: Emission rate (kg/h)
        Q_err: STD of Q  (kg/h)
        err_ime: ime/r error (kg/h)
        err_wind: wind error (kg/h)
    """
    # read file and pick valid data
    LOG.info('Reading data')
    ds = xr.open_dataset(f_gas_mask)
    sp = ds['sp'].mean().item()  # use the mean surface pressure (Pa)

    # get the masked plume data
    gas_mask = ds.dropna(dim='y', how='all').dropna(dim='x', how='all')[gas]

    # get wind info
    LOG.info('Calculating wind info')
    u10 = ds['u10'].sel(source=wind_source).item()
    v10 = ds['v10'].sel(source=wind_source).item()
    if wspd is None:
        wspd = np.sqrt(u10**2 + v10**2)

    # area of pixel in m2
    area = pixel_res*pixel_res

    # create mask centered on source point
    LOG.info('Calculating the index of source loc')
    y_target, x_target = get_index_nearest(gas_mask['longitude'], gas_mask['latitude'], longitude, latitude)

    mask = np.zeros(gas_mask.shape)
    mask[y_target, x_target] = 1

    # calculate plume height, width, and diagonal
    h = mask.shape[0]
    w = mask.shape[1]
    # r_max = np.sqrt(h**2+w**2)
    r_max = 1e3  # limit to 1 km

    # calculate IME by increasing mask radius
    LOG.info('Calculating IME')
    ime = []
    for r in np.arange(r_max):
        r += 1
        LOG.debug(f'IME {r} loop')
        mask = create_circular_mask(h, w,
                                    center=(x_target, y_target),
                                    radius=r)

        IME = ime_radius(gas, gas_mask, mask, sp, area)
        ime.append(IME)

        # no new plume pixels anymore
        if mask.all():
            LOG.info(f'Masking iteration stops at r = {r}')
            break

    # calculate emission rate
    L = (np.arange(len(ime))+1) * pixel_res
    ime_l_mean = np.mean(ime/L)
    ime_l_std = np.std(ime/L)
    Q = (ime_l_mean * wspd).item()

    # ---- uncertainty ----
    # 1. ime
    LOG.info('Calculating IME error')
    err_ime = Q * ime_l_std / ime_l_mean

    # 2. wind error
    LOG.info('Calculating wind error')
    err_wind = calc_wind_error_fetch(wspd, ime_l_mean)

    # sum error
    Q_err = np.sqrt(err_ime**2 + err_wind**2)

    ds.close()

    return Q*3600, Q_err*3600, err_ime*3600, err_wind*3600  # kg/h
