#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Some interp functions copied from `mag1c <https://github.com/markusfoote/mag1c/blob/target-generation/mag1c/target_generation.py>`_, which is licensed under the BSD 3-Clause License."""


import numpy as np
import scipy

def check_param(value, min, max, name):
    if value < min or value > max:
        raise ValueError(f'The value for {name} exceeds the sampled parameter space.'
                         f'The limits are[{min}, {max}], requested {value}.')


@np.vectorize
# [0.,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
def get_5deg_zenith_angle_index(zenith_value):
    check_param(zenith_value, 0, 80, 'Zenith Angle')
    return zenith_value / 5


@np.vectorize
def get_5deg_sensor_height_index(sensor_value):  # [1, 2, 4, 10, 20, 120]
    # Only check lower bound here, atmosphere ends at 120 km so clamping there is okay.
    check_param(sensor_value, 1, np.inf, 'Sensor Height')
    # There's not really a pattern here, so just linearly interpolate between values -- piecewise linear
    if sensor_value < 1.0:
        return np.float64(0.0)
    elif sensor_value < 2.0:
        idx = sensor_value - 1.0
        return idx
    elif sensor_value < 4:
        return sensor_value / 2
    elif sensor_value < 10:
        return (sensor_value / 6) + (4.0 / 3.0)
    elif sensor_value < 20:
        return (sensor_value / 10) + 2
    elif sensor_value < 120:
        return (sensor_value / 100) + 3.8
    else:
        return 5


@np.vectorize
def get_5deg_ground_altitude_index(ground_value):  # [0, 0.5, 1.0, 2.0, 3.0]
    check_param(ground_value, 0, 3, 'Ground Altitude')
    if ground_value < 1:
        return 2 * ground_value
    else:
        return 1 + ground_value


@np.vectorize
def get_5deg_water_vapor_index(water_value):  # [0,1,2,3,4,5,6]
    check_param(water_value, 0, 6, 'Water Vapor')
    return water_value

@np.vectorize
# [0.0,1000,2000,4000,8000,16000,32000,64000]
def get_5deg_methane_index(methane_value):
    # the parameter clamps should rarely be calle because there are default concentrations, but the --concentraitons parameter exposes these
    check_param(methane_value, 0, 64000, 'Methane Concentration')
    if methane_value <= 0:
        return 0
    elif methane_value < 1000:
        return methane_value / 1000
    return np.log2(methane_value / 500)


@np.vectorize
def get_carbon_dioxide_index(coo_value):
    check_param(coo_value, 0, 1280000, 'Carbon Dioxode Concentration')
    if coo_value <= 0:
        return 0
    elif coo_value < 20000:
        return coo_value / 20000
    return np.log2(coo_value / 10000)


def get_5deg_lookup_index(zenith=0, sensor=120, ground=0, water=0, conc=0, gas='ch4'):
    if 'ch4' in gas:
        idx = np.asarray([[get_5deg_zenith_angle_index(zenith)],
                          [get_5deg_sensor_height_index(sensor)],
                          [get_5deg_ground_altitude_index(ground)],
                          [get_5deg_water_vapor_index(water)],
                          [get_5deg_methane_index(conc)]])
    elif 'co2' in gas:
        idx = np.asarray([[get_5deg_zenith_angle_index(zenith)],
                          [get_5deg_sensor_height_index(sensor)],
                          [get_5deg_ground_altitude_index(ground)],
                          [get_5deg_water_vapor_index(water)],
                          [get_carbon_dioxide_index(conc)]])
    else:
        raise ValueError('Unknown gas provided.')
    return idx


def spline_5deg_lookup(grid_data, zenith=0, sensor=120, ground=0, water=0, conc=0, gas='ch4', order=1):
    coords = get_5deg_lookup_index(
        zenith=zenith, sensor=sensor, ground=ground, water=water, conc=conc, gas=gas)
    # correct_lookup = np.asarray([scipy.ndimage.map_coordinates(
    #     im, coordinates=coords, order=order, mode='nearest') for im in np.moveaxis(grid_data, 5, 0)])
    if order == 1:
        coords_fractional_part, coords_whole_part = np.modf(coords)
        coords_near_slice = tuple((slice(int(c), int(c+2))
                                  for c in coords_whole_part))
        near_grid_data = grid_data[coords_near_slice]
        new_coord = np.concatenate((coords_fractional_part * np.ones((1, near_grid_data.shape[-1])),
                                    np.arange(near_grid_data.shape[-1])[None, :]), axis=0)
        lookup = scipy.ndimage.map_coordinates(
            near_grid_data, coordinates=new_coord, order=1, mode='nearest')
    elif order == 3:
        lookup = np.asarray([scipy.ndimage.map_coordinates(
            im, coordinates=coords_fractional_part, order=order, mode='nearest') for im in np.moveaxis(near_grid_data, 5, 0)])
    return lookup.squeeze()

