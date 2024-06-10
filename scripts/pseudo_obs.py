#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Generate pseudo observations of point-source and area-source methane plumes using WRF-LES simualtion data."""

import gc
from glob import glob

import numpy as np
import xarray as xr


def preprocess(ds):
    '''
    Preprocess data into same format
    '''
    # add name dim
    ds = ds.expand_dims(name=[ds.encoding["source"].split('_')[-2]])

    # add coords
    x_coords = np.arange(0, len(ds.x)*int(ds.attrs['DX']), int(ds.attrs['DX']))
    y_coords = np.arange(0, len(ds.y)*int(ds.attrs['DY']), int(ds.attrs['DY']))
    ds = ds.assign_coords({'y': y_coords, 'x': x_coords})

    # drop lev coords
    if 'lev' in list(ds.coords):
        ds = ds.drop_vars(['lev'])

    return ds


def calc_bkgd_noise(ch4_column, bg_ch4=10, precision=[0.01, 0.03, 0.05, 0.12]):
    '''
    Calculate methane background+noise field.

    Args:
        ch4_column (g m-2):
            methane column density
        bg_ch4 (g m-2):
            background methane = 1875 ppb ~ 0.01 kg m-2 --> g m-2
        precision:
            measurement precision (EMIT=3%, EnMAP=5%, PRISMA=12%)

    Return:
        enhancement_noise (DataArray)
    '''
    # Loop through noise levels
    noisy_ch4 = []
    for noise_level in precision:
        # Add background
        ch4_img = np.ones(ch4_column.shape) * bg_ch4

        # Add noise
        noise = np.random.normal(loc=0, scale=noise_level*bg_ch4, size=ch4_column.shape)
        noisy_ch4_img = ch4_img + noise

        noisy_ch4.append(noisy_ch4_img)

    # create DataArray
    enhancement_noise = xr.DataArray(np.stack(noisy_ch4, axis=0) - bg_ch4,
                                     dims=['precision', 'name', 'time', 'y', 'x'],
                                     coords={'precision': precision})
    del noisy_ch4
    gc.collect()

    # assign attrs
    enhancement_noise = enhancement_noise.rename('methane_noisy_background')
    enhancement_noise.attrs['units'] = 'g m-2'

    return enhancement_noise


def pseudo_plume(ds, type='area'):
    '''
    Generate pseudo plume

    Args:
        ds:
            WRF-LES Dataset
        type:
            plume type ('area' or 'point')
    '''
    # random emission rate from 1 to 30 t/h
    print('Generate random emission rates')
    emiss_rate = np.random.uniform(low=1, high=30, size=(ds['ch4'].sizes['name'], ds['ch4'].sizes['time']))  # t/h

    # expand dim
    emiss_rate = np.repeat(np.repeat(emiss_rate[:, :, np.newaxis, np.newaxis], ds['ch4'].sizes['y'], axis=2), ds['ch4'].sizes['x'], axis=3)

    # save as DataArray
    emiss_rate = ds['ch4'].copy(deep=True, data=emiss_rate)
    #emiss_rate = xr.DataArray(emiss_rate,
    #                          dims=['name', 'time'],
    #                          coords={'name': emiss_rate_area.coords['name'],
    #                                  'time': emiss_rate_area.coords['time']
    #                                 }
    #                         )
    emiss_rate = emiss_rate.rename('emission_rate')
    emiss_rate.attrs['units'] = 't h-1'
    emiss_rate.attrs['description'] = 'methane emission rate'

    # convert to mol/s
    emiss_unit = emiss_rate*1e6/(16.04*3600)  # mol/s

    # area source area: 275m*275m
    if type == 'area':
        nemiss_pixels = 275*275/(ds.attrs['DX']*ds.attrs['DY'])
    elif type == 'point':
        nemiss_pixels = 1

    # calculate scale factor
    scale_factor = emiss_unit/(12*nemiss_pixels)  # default emiss: 12 mol/s/npix

    # calculate the column
    print('Calculating column density ...')
    ch4_column = ds['ch4'] * scale_factor
    ch4_column.load()

    print('Adding background and noise ...')
    delta_xch4_noise = calc_bkgd_noise(ch4_column)
    delta_xch4 = ch4_column + delta_xch4_noise

    delta_xch4 = delta_xch4.rename('ch4')
    delta_xch4.attrs['description'] = 'methane enhancement'
    delta_xch4.attrs['units'] = 'g m-2'

    wspd = np.sqrt(ds['U10']**2+ds['V10']**2).rename('wspd')
    wspd.attrs['units'] = 'm s-1'
    wspd.attrs['description'] = '10 m wind speed'

    ds_merge = xr.merge([emiss_rate, delta_xch4, wspd])
    del emiss_rate, delta_xch4, wspd, ch4_column, delta_xch4_noise
    gc.collect()

    return ds_merge


np.random.seed(100)

filenames_area = glob('../hypergas/resources/landfill_ensemble/**/*column.nc', recursive=True)
ds_area = xr.open_mfdataset(filenames_area, preprocess=preprocess, concat_dim='name', combine='nested', parallel=True)
ds_merge_area = pseudo_plume(ds_area, type='area')
savename = '../hypergas/resources/landfill_ensemble/delta_xch4_area.nc'
print(f'Exporting to {savename}')
ds_merge_area.to_netcdf(savename)

filenames_point = glob('../hypergas/resources/point_ensemble/**/*column.nc', recursive=True)
ds_point = xr.open_mfdataset(filenames_point, preprocess=preprocess, concat_dim='name', combine='nested', parallel=True)
ds_merge_point = pseudo_plume(ds_point, type='point')
savename = '../hypergas/resources/point_ensemble/delta_xch4_point.nc'
print(f'Exporting to {savename}')
ds_merge_point.to_netcdf(savename)
