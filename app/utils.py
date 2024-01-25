#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Some utils used for calculating CH4 plume mask and emission rates"""

import base64
import os
import warnings

import numpy as np
import pandas as pd
import pyresample
import xarray as xr
from branca.element import MacroElement
from hypergas.folium_map import Map
from hypergas.landmask import Land_mask
from jinja2 import Template
from pyresample.geometry import SwathDefinition
from scipy import ndimage

warnings.filterwarnings('ignore')

# calculate IME (kg m-2)
mass = 16.04e-3  # molar mass CH4 [kg/mol]
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

    return y_target, x_target


def plume_mask(ds, lon_target, lat_target, plume_varname='ch4_comb_denoise',
               wind_source='ERA5', wind_weights=True, land_only=False,
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
        lon_target (float): source longitude for picking plume dilation mask
        lat_target (float): source latitude for picking plume dilation mask
        plume_varname (str): the variable used to create plume mask (Default:ch4_comb_denoise)
        wind_source (str): 'ERA5' or 'GEOS-FP'
        wind_weights (boolean): whether apply the wind weights to ch4 fields
        n_iter (int): number of iterations for dilations
        quantile_value (float): the lower quantile limit of enhancements are included in the mask
        size_median (int): size for median filter
        sigma_guass (int): sigma for guassian filter
    """
    # get lon and lat
    lon = ds['longitude']
    lat = ds['latitude']

    ch4 = getattr(ds, plume_varname, ds['ch4'])

    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute the 2D grid indices:
    y_target, x_target = get_index_nearest(ds['longitude'], ds['latitude'], lon_target, lat_target)

    if wind_weights:
        # calculate wind angle
        angle_wind_rad, angle_wind = get_wind_azimuth(ds['u10'].sel(source=wind_source).isel(y=y_target, x=x_target).item(),
                                                      ds['v10'].sel(source=wind_source).isel(y=y_target, x=x_target).item())
        weights = conish_2d(lon, lat, lon_target, lat_target, np.pi/2. - angle_wind_rad)
        ch4_weights = ch4 * weights
    else:
        ch4_weights = ch4

    # set oceanic pixel values to nan
    if land_only:
        segmentation = Land_mask(lon.data, lat.data)
        ch4_weights = ch4_weights.where(segmentation)

    # set init mask by the quantile value
    mask_buf = np.zeros(np.shape(ch4))
    mask_buf[ch4_weights >= ch4_weights.quantile(quantile_value)] = 1.

    # blur the mask
    median_blurred = ndimage.median_filter(mask_buf, size_median)
    gaussmed_blurred = ndimage.gaussian_filter(median_blurred, sigma_guass)
    mask = np.zeros(np.shape(ch4))
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
    fg_mask = ch4_weights.where(mask) > 0
    # erosion the mask and propagate again
    eroded_square = ndimage.binary_erosion(fg_mask)
    reconstruction = ndimage.binary_propagation(eroded_square, mask=fg_mask)
    # fill the negative values inside
    mask = ndimage.binary_fill_holes(reconstruction)

    # get the masked lon and lat
    lon_mask = xr.DataArray(lon, dims=['y', 'x']).where(mask).rename('longitude')
    lat_mask = xr.DataArray(lat, dims=['y', 'x']).where(mask).rename('latitude')

    return ds['ch4'], mask, lon_mask, lat_mask


def plot_mask(filename, ds, ch4, mask, lon_target, lat_target, pick_plume_name, only_plume=True):
    """Plot masked data"""
    # get masked plume data
    ch4_mask = ch4.where(xr.DataArray(mask, dims=list(ch4.dims)))
    ds['plume'] = ch4_mask

    if only_plume:
        # only plot plume for quick check
        m = Map(ds, varnames=['plume'], center_map=[lat_target, lon_target])
        m.initialize()
        m.plot(show_layers=[True], opacities=[0.9], vmax=300,
               marker=[lat_target, lon_target], export_dir=os.path.dirname(filename), draw_polygon=False)
    else:
        # plot all important data
        m = Map(ds, varnames=['rgb', 'ch4', 'ch4_comb', 'ch4_comb_denoise',
                'plume'], center_map=[lat_target, lon_target])
        m.initialize()
        m.plot(show_layers=[False, False, False, False, True], opacities=[0.9, 0.8, 0.8, 0.8, 0.8],  vmax=300,
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


def mask_data(filename, ds, lon_target, lat_target, pick_plume_name, plume_varname,
              wind_source, wind_weights, land_only,
              niter, size_median, sigma_guass, quantile_value, only_plume):
    '''Generate and plot masked plume data

    Args:
        filename (str): input filename
        ds (Dataset): L2 data
        lon_target (float): The longitude of plume source
        lat_target (float): The latitude of plume source
        pick_plume_name (str): the plume name (plume0, plume1, ....)
        plume_varname (str): the variable used to create plume mask (Default:ch4_comb_denoise)
        wind_source (str): 'ERA5' or 'GEOS-FP'
        wind_weights (boolean): whether apply the wind weights to ch4 fields
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
    ch4, mask, lon_mask, lat_mask = plume_mask(ds, lon_target, lat_target, plume_varname=plume_varname,
                                               wind_source=wind_source, wind_weights=wind_weights, land_only=land_only,
                                               niter=niter, size_median=size_median, sigma_guass=sigma_guass,
                                               quantile_value=quantile_value)

    # plot the mask (png and html)
    plume_html_filename = plot_mask(filename, ds, ch4, mask, lon_target, lat_target,
                                    pick_plume_name, only_plume=only_plume)

    return mask, lon_mask, lat_mask, plume_html_filename


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


def ime_radius(ch4, mask, sp, area):
    ch4_mask = ch4.where(mask)
    if ch4_mask.attrs['units'] == 'ppb':
        delta_omega = ch4_mask * 1.0e-9 * (mass / mass_dry_air) * sp / grav
    elif ch4_mask.attrs['units'] == 'ppm m':
        delta_omega = ch4_mask * 7.16e-7
    IME = np.nansum(delta_omega * area)

    return IME


def calc_wind_error(wspd, IME, l_eff,
                    alpha1, alpha2, alpha3,
                    ):
    """Calculate wind error with random distribution"""
    # Generate U10 distribution
    #   uncertainty = 50%, if wspd <= 2 m/s
    #   uncertainty = 1.5 m/s, if wspd > 2 m/s
    if wspd <= 2:
        sigma = wspd * 0.5
    else:
        sigma = 1.5

    wspd_distribution = np.random.normal(wspd, sigma, size=1000)

    # Calculate Ueff distribution
    u_eff_distribution = alpha1 * np.log(wspd) + alpha2 + alpha3 * wspd_distribution

    # Calculate Q distribution
    Q_distribution = u_eff_distribution * IME / l_eff

    # Calculate standard deviation of Q distribution
    wind_error = np.std(Q_distribution)

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


def calc_random_err(ch4, ch4_mask, area, sp):
    """Calculate random error by moving plume around the whole scene"""
    # crop ch4 to valid region
    ch4_mask_crop = ch4_mask.where(~ch4_mask.isnull()).dropna(dim='y', how='all').dropna(dim='x', how='all')

    # get the shape of input data and mask
    bkgd_rows, bkgd_cols = ch4_mask.shape
    mask_rows, mask_cols = ch4_mask_crop.shape

    # Insert plume mask data at a random position
    IME_noplume = []

    while len(IME_noplume) <= 500:
        # Generate random row and column index to place b inside a
        row_idx = np.random.randint(0, bkgd_rows - mask_rows)
        col_idx = np.random.randint(0, bkgd_cols - mask_cols)

        if not np.any(ch4[row_idx:row_idx+mask_rows, col_idx:col_idx+mask_cols].isnull()):
            ch4_bkgd_mask = xr.zeros_like(ch4)
            ch4_bkgd_mask[row_idx:row_idx+mask_rows, col_idx:col_idx+mask_cols] = ch4_mask_crop.values
            ch4_bkgd_mask = ch4_bkgd_mask.fillna(0)
            if ch4.attrs['units'] == 'ppb':
                IME_noplume.append(ch4.where(ch4_bkgd_mask, drop=True).sum().values *
                                   1.0e-9 * (mass / mass_dry_air) * sp / grav * area)
            elif ch4.attrs['units'] == 'ppm m':
                IME_noplume.append(ch4.where(ch4_bkgd_mask, drop=True).sum().values * 7.16e-7)

    return np.array(IME_noplume).std()


def calc_emiss(f_ch4_mask, pick_plume_name, pixel_res=30, alpha1=0.0, alpha2=0.66, alpha3=0.34,
               wind_source='ERA5', wspd=None, land_only=False):
    '''Calculate the emission rate (kg/h) using IME method

    Args:
        f_ch4_mask: The NetCDF file of mask created by `mask_data` function
        pick_plume_name (str): the plume name (plume0, plume1, ....)
        pixel_res (float): pixel resolution (meter)
        alpha1--3 (float): The coefficients for effective wind (U_eff)
        wspd (float): overwritten wind speed
        land_only (boolean): whether calculate random error only over land

    Return:
        wspd: Mean wind speed (m/s)
        wdir: Mean wind direction (deg)
        L_eff: Effctive length (m)
        U_eff: Effective wind speed (m/s)
        Q: Emission rate (kg/h)
        Q_err: STD of Q  (kg/h)
        err_random: random error (kg/h)
        err_wind: wind error (kg/h)
        err_shape: shape error (kg/h)

    '''
    # read file and pick valid data
    ds_original = xr.open_dataset(f_ch4_mask.replace('L3', 'L2').replace(f'_{pick_plume_name}.nc', '.nc'))
    ds = xr.open_dataset(f_ch4_mask)

    # get the masked plume data
    ch4_mask = ds.dropna(dim='y', how='all').dropna(dim='x', how='all')['ch4']

    # area of pixel in m2
    area = pixel_res*pixel_res

    # calculate Leff using the root method in meter
    plume_pixel_num = (~ch4_mask.isnull()).sum()
    l_eff = np.sqrt(plume_pixel_num * area).item()

    # calculate IME
    sp = ds['sp'].mean().item()  # use the mean surface pressure (Pa)
    if ch4_mask.attrs['units'] == 'ppb':
        delta_omega = ch4_mask * 1.0e-9 * (mass / mass_dry_air) * sp / grav
    elif ch4_mask.attrs['units'] == 'ppm m':
        delta_omega = ch4_mask * 7.16e-7
    IME = np.nansum(delta_omega * area)

    # get wind info
    u10 = ds['u10'].sel(source=wind_source).item()
    v10 = ds['v10'].sel(source=wind_source).item()
    if wspd is None:
        wspd = np.sqrt(u10**2 + v10**2)
    wdir = (270-np.rad2deg(np.arctan2(v10, u10))) % 360

    # effective wind speed
    u_eff = alpha1 * np.log(wspd) + alpha2 + alpha3 * wspd

    # calculate the emission rate (kg/s)
    Q = (u_eff / l_eff * IME)

    # ---- uncertainty ----
    # 1. random
    if land_only:
        # get lon and lat
        lon = ds_original['longitude']
        lat = ds_original['latitude']
        segmentation = Land_mask(lon.data, lat.data)
        ds_original['ch4'] = ds_original['ch4'].where(segmentation)
        ds['ch4'] = ds['ch4'].where(segmentation)

    IME_std = calc_random_err(ds_original['ch4'], ds['ch4'], area, sp)
    err_random = u_eff / l_eff * IME_std

    # 2. wind error
    err_wind = calc_wind_error(wspd, IME, l_eff, alpha1, alpha2, alpha3)

    # 3. alpha2 (shape)
    err_shape = (IME / l_eff) * (alpha2 * (0.66-0.42)/0.42)

    Q_err = np.sqrt(err_random**2 + err_wind**2 + err_shape**2)

    return wspd, wdir, l_eff, u_eff, IME, Q*3600, Q_err*3600, \
        err_random*3600, err_wind*3600, err_shape*3600  # kg/h


def calc_emiss_fetch(f_ch4_mask, pixel_res=30, wind_source='ERA5', wspd=None):
    """Calculate the emission rate (kg/h) using IME-fetch method

    Args:
        f_ch4_mask: The NetCDF file of mask created by `mask_data` function
        pixel_res (float): pixel resolution (meter)

    Return:
        Q: Emission rate (kg/h)
        Q_err: STD of Q  (kg/h)
        err_ime: ime/r error (kg/h)
        err_wind: wind error (kg/h)
    """
    # read file and pick valid data
    ds = xr.open_dataset(f_ch4_mask)
    sp = ds['sp'].mean().item()  # use the mean surface pressure (Pa)

    df = pd.read_csv(f_ch4_mask.replace('.nc', '.csv'))
    lon_target = df['plume_longitude']
    lat_target = df['plume_latitude']

    # get the masked plume data
    ch4_mask = ds.dropna(dim='y', how='all').dropna(dim='x', how='all')['ch4']

    # get wind info
    u10 = ds['u10'].sel(source=wind_source).item()
    v10 = ds['v10'].sel(source=wind_source).item()
    if wspd is None:
        wspd = np.sqrt(u10**2 + v10**2)

    # area of pixel in m2
    area = pixel_res*pixel_res

    # create mask centered on source point
    mask = np.zeros(ch4_mask.shape)
    y_target, x_target = get_index_nearest(ch4_mask['longitude'], ch4_mask['latitude'], lon_target, lat_target)
    mask[y_target, x_target] = 1

    # calculate plume height, width, and diagonal
    h = mask.shape[0]
    w = mask.shape[1]
    r_max = np.sqrt(h**2+w**2)

    # calculate IME by increasing mask radius
    ime = []
    for r in np.arange(r_max):
        r += 1
        mask = create_circular_mask(h, w,
                                    center=(x_target, y_target),
                                    radius=r)

        IME = ime_radius(ch4_mask, mask, sp, area)
        ime.append(IME)

        # no new plume pixels anymore
        if mask.all():
            print(f'Masking iteration stops at r = {r}')
            break

    # calculate emission rate
    L = (np.arange(len(ime))+1) * pixel_res
    ime_l_mean = np.mean(ime/L)
    ime_l_std = np.std(ime/L)
    Q = (ime_l_mean * wspd).item()

    # ---- uncertainty ----
    # 1. ime
    err_ime = Q * ime_l_std / ime_l_mean

    # 2. wind error
    err_wind = calc_wind_error_fetch(wspd, ime_l_mean)

    # sum error
    Q_err = np.sqrt(err_ime**2 + err_wind**2)

    return Q*3600, Q_err*3600, err_ime*3600, err_wind*3600  # kg/h
