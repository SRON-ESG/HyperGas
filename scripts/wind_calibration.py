#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Wind calibration for EMIT, EnMAP, and PRISMA."""

import gc
import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy
import tobac
import xarray as xr
from scipy import ndimage
from shapely.geometry import Polygon
from skimage.restoration import denoise_tv_chambolle

np.random.seed(100)
warnings.filterwarnings('ignore')


def _azimuth(point1, point2):
    # https://stackoverflow.com/a/66118219/7347925
    """azimuth between 2 points (interval 0 - 180)"""
    angle = np.arctan2(point2[0] - point1[0], point2[1] - point1[1])
    return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180


def _dist(a, b):
    """distance between points"""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def azimuth(mrr):
    """azimuth of minimum_rotated_rectangle"""
    if isinstance(mrr, Polygon):
        bbox = list(mrr.exterior.coords)
        axis1 = _dist(bbox[0], bbox[3])
        axis2 = _dist(bbox[0], bbox[1])

        if axis1 <= axis2:
            az = _azimuth(bbox[0], bbox[1])
        else:
            az = _azimuth(bbox[0], bbox[3])
    else:
        az = np.nan

    return az


def denoise_data(data, weight):
    data = xr.DataArray(data, dims=['y', 'x'])
    return denoise_tv_chambolle(data.expand_dims(time=1), weight=weight)


def get_feature_mask(data, sigma_threshold=1, n_min_threshold=5):
    dxy = abs(data.y.diff('y')[0])
    thresholds = [(data.mean() + 2*data.std()).values, (data.mean() + 3*data.std()).values]

    # detect features
    features = tobac.feature_detection_multithreshold(
        data,
        dxy, thresholds,
        position_threshold='extreme',
        n_min_threshold=n_min_threshold,
        sigma_threshold=sigma_threshold,
    )

    threshold_seg = [(data.mean() + 2*data.std()).values]

    masks, features_mask = tobac.segmentation_2D(
        features, data, dxy, threshold=threshold_seg, statistic={'feature_max': np.max})

    features_mask['feature'] = np.arange(1, len(features_mask)+1, 1)
    # keep masks of more than n_min_threshold pixels
    masks = masks.astype('int')
    mask_count = masks.groupby(masks).count()
    mask_count = mask_count.where(mask_count > n_min_threshold).dropna(masks.name).astype('int')
    mask_unique = mask_count.coords['segmentation_mask'].values
    masks = masks.where(np.isin(masks, mask_unique, 0))
    masks = masks.drop_vars(['time'])

    mask_count = masks.groupby(masks).count().astype('int')
    mask_unique = mask_count.coords['segmentation_mask'].values

    # Replace values in the DataArray with sequential numbers
    for index, value in enumerate(mask_unique):
        masks = xr.where(masks == value, index, masks)

    return thresholds, features_mask, masks


def select_connect_masks(masks, masks_dilation, gdf_polygon, y_target, x_target):
    # get the source label of original mask and dilation mask
    mask_target = masks[y_target, x_target].item()
    mask_dilation_target = masks_dilation[y_target, x_target].values

    # get the dilation mask which contains mask including the target
    mask_dilation_target = masks_dilation.where(masks_dilation == mask_dilation_target)

    # mask in the dilation mask
    masks_in_dilation = masks.where((masks > 0) & (mask_dilation_target > 0))

    # unique mask labels within the dilation mask
    connect_labels = np.unique(masks_in_dilation.data.flatten())

    # get the polygons inside the dilation mask which includes the target mask
    gdf_polygon_connect = gdf_polygon[gdf_polygon.index.isin(connect_labels)]

    if len(gdf_polygon_connect) > 1:
        # more thane one masks inside the dilation mask
        # calculate polygon distance
        distance = gdf_polygon_connect.geometry.apply(
            lambda g: gdf_polygon_connect[gdf_polygon_connect.index == mask_target]['geometry'].distance(g, align=False))
        if distance.empty:
            return gdf_polygon_connect, masks_in_dilation
        else:
            gdf_polygon_connect['distance'] = distance
            # sort masks by distance
            gdf_polygon_connect.sort_values('distance', inplace=True)

            # calcualte differences of az
            gdf_polygon_connect.loc[:, 'az_diff'] = gdf_polygon_connect['az'].diff().abs().fillna(0)

            # Drop rows where az_diff is higher than 20
            for i in range(len(gdf_polygon_connect) - 1):
                if i >= len(gdf_polygon_connect)-1:
                    break
                if (gdf_polygon_connect['az_diff'].iloc[i+1] > 20) and (gdf_polygon_connect['distance'].iloc[i+1] > 0):
                    gdf_polygon_connect = gdf_polygon_connect.drop(gdf_polygon_connect.index[i+1])
                    gdf_polygon_connect['az_diff'] = gdf_polygon_connect['az'].diff().abs().fillna(0)

    return gdf_polygon_connect, masks_in_dilation


def calc_wspd(emiss_rate, data, connect_masks, wspd):
    '''Calculate wind speed: U10 and Ueff'''
    Q = emiss_rate  # t/h
    dxy = data.coords['y'].diff('y')[0]
    IME = data.where(connect_masks > 0).sum() * dxy**2 / 1e6  # g m-2 * m2 --> t
    L = np.sqrt((connect_masks > 0).sum() * dxy**2)  # m
    lifetime = (IME/Q*3600).load().item()  # s
    st = (data.coords['time'] - pd.Timedelta(lifetime, 'S')).dt.strftime('%Y-%m-%d %H:%M:%S').item()
    et = data.coords['time'].dt.strftime('%Y-%m-%d %H:%M:%S').item()
    wspd_mean = wspd.sel(time=slice(st, et)).where(connect_masks > 0).mean().values
    Ueff = (Q * L / IME / 3600).values  # m/s

    return wspd_mean, Ueff


def create_mask(data, denoise_data, emiss_rate, wspd, y_target=None, x_target=None, dist=180):
    denoise_data = denoise_data.squeeze()
    denoise_data = denoise_data.expand_dims(time=1)
    thresholds, features, masks = get_feature_mask(denoise_data)

    df_mask = masks.to_dataframe().reset_index()
    df_mask = df_mask[df_mask['segmentation_mask'] > 0]
    df_mask = df_mask[df_mask['segmentation_mask'].map(df_mask['segmentation_mask'].value_counts()).gt(5)]

    gdf_polygon = gpd.GeoDataFrame(geometry=df_mask.groupby('segmentation_mask')
                                   .apply(lambda g: Polygon(gpd.points_from_xy(g['x'], g['y'])))
                                   )

    gdf_polygon['mrrs'] = gdf_polygon.geometry.apply(lambda geom: geom.minimum_rotated_rectangle)
    gdf_polygon['az'] = gdf_polygon['mrrs'].apply(azimuth)
    gdf_polygon = gdf_polygon.dropna(how='any')

    # dilation the mask
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    dxy = data.coords['y'].diff('y')[0]
    niter = int(dist/dxy)
    masks_dilation = masks.copy(deep=True, data=ndimage.binary_dilation(masks, iterations=niter,  structure=struct))
    masks_dilation = masks_dilation.where(masks.notnull())

    # Label connected components in the dilated array
    labeled_array, num_features = ndimage.label(masks_dilation)
    masks_dilation = masks.copy(deep=True, data=labeled_array)

    # the target should be around the rightmost feature location
    target_max = features.loc[features['feature_max'].idxmax()]
    if y_target is None:
        y_target = target_max['hdim_1']
    if x_target is None:
        x_target = target_max['hdim_2']

    gdf_polygon_connect, masks_in_dilation = select_connect_masks(
        masks, masks_dilation, gdf_polygon, y_target=y_target, x_target=x_target)

    connect_labels = gdf_polygon_connect.index
    connect_masks = masks_in_dilation.where(masks_in_dilation.isin(connect_labels))

    U10, Ueff = calc_wspd(emiss_rate, data, connect_masks, wspd)

    return U10, Ueff


def denoise_ch4(delta_xch4, weight=50):
    delta_xch4_denoise = xr.apply_ufunc(denoise_data,
                                        delta_xch4,
                                        kwargs={'weight': weight},
                                        input_core_dims=[['y', 'x']],
                                        output_core_dims=[['y', 'x']],
                                        dask="parallelized",
                                        output_dtypes=[delta_xch4.dtype],
                                        vectorize=True,
                                        )

    # load data
    delta_xch4_denoise = delta_xch4_denoise.load()

    return delta_xch4_denoise


def read_data(filename):
    # read full data lazily
    ds = xr.open_mfdataset(filename)
    wspd = ds['wspd']

    # subset to last 2 hours
    dt = ds.time.diff('time').dt.seconds[0]
    ds = ds.isel(time=slice(int(3600/dt)+1, None))
    emiss = ds['emission_rate'].load()
    delta_xch4 = ds['ch4'].load()

    return delta_xch4, wspd, emiss


def calibration(delta_xch4, delta_xch4_denoise, wspd, emiss, y_target=None, x_target=None):
    '''Calibrate wind speed'''
    df_list = []
    for name in delta_xch4.coords['name'].values:
        for precision in delta_xch4.coords['precision']:
            u10_list = []
            ueff_list = []
            t_list = []
            emiss_list = []
            for t in delta_xch4.coords['time']:
                print(name, t.dt.strftime('%H:%M:%S'))
                u10, ueff = create_mask(delta_xch4.sel(time=t, precision=precision, name=name),
                                        delta_xch4_denoise.sel(time=t, precision=precision, name=name),
                                        emiss.sel(time=t, name=name),
                                        wspd.sel(name=name),
                                        y_target, x_target,
                                        )
                t_list.append(t.dt.strftime('%H:%M:%S').item())
                u10_list.append(u10.item())
                ueff_list.append(ueff.item())
                emiss_list = emiss.sel(time=t, name=name).item()

            df_tmp = pd.DataFrame({'time': t_list, 'u10': u10_list, 'ueff': ueff_list, 'emission_rate': emiss_list})
            df_tmp['name'] = name
            df_tmp['precision'] = precision.item()
            df_list.append(df_tmp)

    return pd.concat(df_list)


# read products derived from WRF-LES data
delta_xch4_area, wspd_area, emiss_area = read_data('../hypergas/resources/landfill_ensemble/delta_xch4_area.nc')
delta_xch4_point, wspd_point, emiss_point = read_data('../hypergas/resources/point_ensemble/delta_xch4_point.nc')
delta_xch4_area_50m = delta_xch4_area.coarsen(y=2, x=2).mean().load()
delta_xch4_point_50m = delta_xch4_point.coarsen(y=2, x=2).mean().load()
wspd_area_50m = wspd_area.coarsen(y=2, x=2).mean().load()
wspd_point_50m = wspd_point.coarsen(y=2, x=2).mean().load()

# hard code the plume source
# this doesn't work, I added the auto one
# y_target_area = 120; x_target_area = 195
# y_target_point = 180; x_target_point = 299
# y_target_area_50m = 60; x_target_area_50m = 97
# y_target_point_50m = 90; x_target_point_50m = 149

# ---- area source ----
# denoise delta_xch4 field
# EMIT: res=60m, weight=50, EnMAP: res=30m, weight=50, PRISMA: res=30m, weight=120
print('Denoising area-source data ...')
print('Denoising data (EnMAP)')
delta_xch4_denoise_enmap = denoise_ch4(delta_xch4_area, weight=50)
print('Denoising data (PRISMA)')
delta_xch4_denoise_prisma = denoise_ch4(delta_xch4_area, weight=120)
print('Denoising data (EMIT)')
delta_xch4_denoise_emit = denoise_ch4(delta_xch4_area_50m, weight=50)

print('Calibrating wspd ...')
savedir_area = '../hypergas/resources/landfill_ensemble/'
savename = f'{savedir_area}/wspd_calib_enmap.csv'
df_enmap_area = calibration(delta_xch4_area, delta_xch4_denoise_enmap, wspd_area, emiss_area)
print(f'Exported to {savename}')
df_enmap_area.to_csv(savename, index=False)
del delta_xch4_denoise_enmap
gc.collect()

df_prisma_area = calibration(delta_xch4_area, delta_xch4_denoise_prisma, wspd_area, emiss_area)
savename = f'{savedir_area}/wspd_calib_prisma.csv'
print(f'Exported to {savename}')
df_prisma_area.to_csv(savename, index=False)
del delta_xch4_denoise_prisma
gc.collect()

df_emit_area = calibration(delta_xch4_area_50m, delta_xch4_denoise_emit, wspd_area_50m, emiss_area)
savename = f'{savedir_area}/wspd_calib_emit.csv'
print(f'Exported to {savename}')
df_emit_area.to_csv(savename, index=False)
del delta_xch4_denoise_emit
gc.collect()

# ---- point source ----
print('Denoising point-source data ...')
print('Denoising data (EnMAP)')
delta_xch4_denoise_enmap = denoise_ch4(delta_xch4_point, weight=50)
print('Denoising data (PRISMA)')
delta_xch4_denoise_prisma = denoise_ch4(delta_xch4_point, weight=120)
print('Denoising data (EMIT)')
delta_xch4_denoise_emit = denoise_ch4(delta_xch4_point_50m, weight=50)

print('Calibrating wspd ...')
savedir_point = '../hypergas/resources/point_ensemble/'
savename = f'{savedir_point}/wspd_calib_enmap.csv'
df_enmap_point = calibration(delta_xch4_point, delta_xch4_denoise_enmap, wspd_point, emiss_point)
print(f'Exported to {savename}')
df_enmap_point.to_csv(savename, index=False)
del delta_xch4_denoise_enmap
gc.collect()

df_prisma_point = calibration(delta_xch4_point, delta_xch4_denoise_prisma, wspd_point, emiss_point)
savename = f'{savedir_point}/wspd_calib_prisma.csv'
print(f'Exported to {savename}')
df_prisma_point.to_csv(savename, index=False)
del delta_xch4_denoise_prisma
gc.collect()

df_emit_point = calibration(delta_xch4_point_50m, delta_xch4_denoise_emit, wspd_point_50m, emiss_point)
savename = f'{savedir_point}/wspd_calib_emit.csv'
print(f'Exported to {savename}')
df_emit_point.to_csv(savename, index=False)
del delta_xch4_denoise_emit
gc.collect()
