#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Fitting the polynomial centerline for the CSF method."""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point
from pyproj import Transformer
import gc


def feature(x, order=3):
    """Generate polynomial feature matrix [1, x, x^2, ..., x^order]"""
    x = x.reshape(-1, 1)
    return np.power(x, np.arange(order+1).reshape(1, -1))


def fit_polynomial_centerline(x, y, weights, x_source, y_source, order=3, alpha=0.01):
    """
    Fit a polynomial curve through plume data using principal direction.

    Parameters
    ----------
    x, y : array-like
        Coordinates of points to fit
    weights : array-like
        Weights for each point (e.g., gas concentrations)
    x_source, y_source : float
        Source location
    order : int
        Polynomial order (3 = cubic)
    alpha : float
        Regularization parameter

    Returns
    -------
    w : array
        Polynomial coefficients
    rotation_angle : float
        Rotation angle used
    x_offset, y_offset : float
        Translation offsets
    """
    # Prepare data
    X = np.asarray(x).flatten()
    Y = np.asarray(y).flatten()
    w_vals = np.asarray(weights).flatten()

    # Normalize weights
    w_vals = w_vals / np.max(w_vals)

    # Find principal direction (source to weighted centroid)
    w_norm = w_vals / np.sum(w_vals)
    x_centroid = np.sum(X * w_norm)
    y_centroid = np.sum(Y * w_norm)

    # Principal direction vector
    dx_principal = x_centroid - x_source
    dy_principal = y_centroid - y_source
    rotation_angle = np.arctan2(dy_principal, dx_principal)

    # Rotate coordinates so principal direction aligns with x-axis
    cos_theta = np.cos(-rotation_angle)
    sin_theta = np.sin(-rotation_angle)

    # Translate to source as origin, then rotate
    x_translated = X - x_source
    y_translated = Y - y_source

    x_rotated = x_translated * cos_theta - y_translated * sin_theta
    y_rotated = x_translated * sin_theta + y_translated * cos_theta

    # Now fit polynomial in rotated space: y_rotated = f(x_rotated)
    W = np.diag(w_vals)
    A = feature(x_rotated, order)

    # Weighted ridge regression
    poly_coeffs = np.linalg.pinv(
        A.T.dot(W).dot(A) + alpha * np.eye(A.shape[1])
    ).dot(A.T).dot(W).dot(y_rotated)

    return poly_coeffs, rotation_angle, x_source, y_source


def evaluate_polynomial_curve(w, rotation_angle, x_offset, y_offset, x_rotated):
    """
    Evaluate polynomial curve and transform back to original coordinates.

    Parameters
    ----------
    w : array
        Polynomial coefficients
    rotation_angle : float
        Rotation angle
    x_offset, y_offset : float
        Translation offsets (source location)
    x_rotated : array
        x coordinates in rotated space

    Returns
    -------
    x_original, y_original : arrays
        Coordinates in original space
    """
    # Evaluate polynomial in rotated space
    X_rot = feature(x_rotated.reshape(-1), len(w)-1)
    y_rotated = X_rot.dot(w)

    # Rotate back to original space
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    x_original = x_rotated * cos_theta - y_rotated * sin_theta + x_offset
    y_original = x_rotated * sin_theta + y_rotated * cos_theta + y_offset

    return x_original, y_original


def get_tangent_vector(w, x):
    """
    Calculate unit tangent vector at point x for polynomial curve in rotated space.

    Parameters
    ----------
    w : array
        Polynomial coefficients [w0, w1, w2, ..., wd]
    x : float
        x-coordinate in rotated space

    Returns
    -------
    tx, ty : float
        Unit tangent vector components in rotated space
    """
    # dy/dx = w1 + 2*w2*x + 3*w3*x^2 + ...
    derivative = sum(i * w[i] * x**(i-1) for i in range(1, len(w)))

    # Tangent vector in rotated space: (1, dy/dx)
    tx = 1.0
    ty = derivative

    # Normalize
    norm = np.sqrt(tx**2 + ty**2)
    return tx / norm, ty / norm


def compute_curve_arc_length_fine(w, x_start, x_end, n_samples=10000):
    """
    Compute arc length of polynomial curve using fine numerical integration.

    Parameters
    ----------
    w : array
        Polynomial coefficients
    x_start, x_end : float
        x-coordinate range in rotated space
    n_samples : int
        Number of samples for numerical integration

    Returns
    -------
    cumulative_arc : array
        Cumulative arc length at each x sample
    x_samples : array
        x coordinates of samples
    """
    x_samples = np.linspace(x_start, x_end, n_samples)

    # Compute derivative dy/dx at each point
    derivatives = np.array([sum(i * w[i] * x**(i-1) for i in range(1, len(w)))
                           for x in x_samples])

    # Arc length element: sqrt(1 + (dy/dx)^2) * dx
    dx = (x_end - x_start) / (n_samples - 1)
    arc_elements = np.sqrt(1 + derivatives**2) * dx

    # Cumulative arc length
    cumulative_arc = np.concatenate([[0], np.cumsum(arc_elements[:-1])])

    return cumulative_arc, x_samples


def get_x_at_arc_length(cumulative_arc, x_samples, target_arc_length):
    """
    Find x coordinate corresponding to a specific arc length along the curve.

    Parameters
    ----------
    cumulative_arc : array
        Cumulative arc lengths
    x_samples : array
        Corresponding x coordinates
    target_arc_length : float
        Desired arc length from start

    Returns
    -------
    x : float
        x coordinate at target arc length
    """
    if target_arc_length >= cumulative_arc[-1]:
        return x_samples[-1]
    if target_arc_length <= 0:
        return x_samples[0]

    # Interpolate to find exact x
    return np.interp(target_arc_length, cumulative_arc, x_samples)


def calculate_perpendicular_extent(x_data, y_data, weights, w, rotation_angle,
                                   x_offset, y_offset, x_start, x_end, n_samples=100):
    """
    Calculate the maximum perpendicular distance of plume points from the centerline.

    Parameters
    ----------
    x_data, y_data : array-like
        Coordinates of plume points in the original space.
    weights : array-like
        Weights for each point.
    w : array-like
        Polynomial coefficients defining the plume centerline in rotated space.
    rotation_angle : float
        Rotation angle of the coordinate transformation.
    x_offset, y_offset : float
        Source location offsets.
    x_start, x_end : float
        Range of x-coordinates (in rotated space) for evaluating the centerline.
    n_samples : int, optional
        Number of sample points to generate along the centerline.

    Returns
    -------
    max_perp_distance : float
        Maximum perpendicular distance from any sufficiently weighted point
        to the plume centerline.
    """

    # Generate centerline points in original space
    x_rot_samples = np.linspace(x_start, x_end, n_samples)
    x_orig, y_orig = evaluate_polynomial_curve(
        w, rotation_angle, x_offset, y_offset, x_rot_samples
    )

    # Construct LineString for centerline
    centerline = LineString(np.column_stack((x_orig, y_orig)))

    # Only consider points with weight > 10% of maximum
    weight_threshold = 0.1 * np.max(weights)
    valid_points = (weights > weight_threshold)

    # Compute perpendicular distances for valid points
    max_distance = max(
        Point(x, y).distance(centerline)
        for x, y in zip(x_data[valid_points], y_data[valid_points])
    )

    return max_distance


def create_perpendicular_lines_equal_arc(w, rotation_angle, x_offset, y_offset,
                                         x_start, x_end, n_lines, line_width):
    """
    Create perpendicular lines at equal arc-length intervals in original coordinate system.

    Parameters
    ----------
    w : array
        Polynomial coefficients
    rotation_angle : float
        Rotation angle
    x_offset, y_offset : float
        Source location
    x_start, x_end : float
        x-coordinate range in rotated space
    n_lines : int
        Number of perpendicular lines
    line_width : float
        Total width of perpendicular lines

    Returns
    -------
    csf_lines : list of LineString
        Perpendicular lines in original coordinate system
    """
    csf_lines = []
    half_width = line_width / 2.0

    # Compute cumulative arc length with fine sampling
    cumulative_arc, x_samples = compute_curve_arc_length_fine(w, x_start, x_end, n_samples=10000)
    total_arc_length = cumulative_arc[-1]

    # Generate positions at equal arc-length intervals
    if n_lines == 1:
        arc_lengths = np.array([total_arc_length / 2.0])
    else:
        arc_lengths = np.linspace(0, total_arc_length, n_lines)

    # Rotation matrices
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)

    for arc_len in arc_lengths:
        # Find x coordinate in rotated space at this arc length
        x_rot = get_x_at_arc_length(cumulative_arc, x_samples, arc_len)

        # Get tangent vector in rotated space
        tx_rot, ty_rot = get_tangent_vector(w, x_rot)

        # Perpendicular vector in rotated space (rotate tangent by 90°)
        px_rot = -ty_rot
        py_rot = tx_rot

        # Transform center point to original space
        X_rot = feature(np.array([x_rot]), len(w)-1)
        y_rot = float(X_rot.dot(w))

        x_center = x_rot * cos_theta - y_rot * sin_theta + x_offset
        y_center = x_rot * sin_theta + y_rot * cos_theta + y_offset

        # Transform perpendicular vector to original space
        px_orig = px_rot * cos_theta - py_rot * sin_theta
        py_orig = px_rot * sin_theta + py_rot * cos_theta

        # Normalize (should already be normalized, but just to be safe)
        p_norm = np.sqrt(px_orig**2 + py_orig**2)
        px_orig /= p_norm
        py_orig /= p_norm

        # Create perpendicular line endpoints
        x1 = x_center - half_width * px_orig
        y1 = y_center - half_width * py_orig
        x2 = x_center + half_width * px_orig
        y2 = y_center + half_width * py_orig

        line = LineString([[x1, y1], [x2, y2]])
        csf_lines.append(line)

    return csf_lines
