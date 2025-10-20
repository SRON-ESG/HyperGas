#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Modified CSF centerline definition from the `ddeq package <https://ddeq.readthedocs.io/en/stable/>`_."""


import numpy as np
import scipy
import xarray as xr


class Poly2D:
    def __init__(
        self, x, y, w, degree=2, x_o=0.0, y_o=0.0, x0=0.0, y0=0.0, force_source=True
    ):
        """\
        A 2D curve fitted on the point cloud given by x and y.

        Parameters
        ----------
        x,y,w: x,y coords and weights used for fitting the data

        degree: degrees if the two polynomials x(t) and y(t)

        x_o, y_o: location of source (will be added to x,y and given
                  high weight such that curves goes through source if
                  force_origin is True)

        x0, y0: origin of coordinate system
        """
        self.x = x
        self.y = y
        self.w = w
        self.degree = degree
        self.force_source = force_source

        self.x_o = x_o
        self.y_o = y_o
        self.t_o = np.nan

        self.x0 = x0
        self.y0 = y0

        # initial coefficients
        self.c = np.zeros(2 * (degree + 1))
        self.c[2] = self.x_o
        self.c[5] = self.y_o

        self._fit()

        # arc length to origin
        self.t_o = self.get_parameter(self.x_o, self.y_o)

        #
        self.tmin = 0.0
        self.tmax = np.max(self.get_parameter(self.x, self.y))
        self.interpolator = None

    def _fit(self):
        def objective(c, w, x, y):
            xt, yt = self._compute_curve(c, x, y)

            return np.concatenate([w * (xt - x), w * (yt - y)])

        # add origin
        if self.force_source:
            x = np.append(self.x, self.x_o)
            y = np.append(self.y, self.y_o)
            w = np.append(self.w, 100.0 * np.nanmax(self.w))
        else:
            x = np.append(self.x, self.x0)  # force origin of coords
            y = np.append(self.y, self.y0)
            w = np.append(self.w, 1000.0)

        # angle around origin (not used)
        # phi = np.arctan2(y - y0, x - x0)

        # curve fit
        res = scipy.optimize.leastsq(
            objective, x0=self.c, args=(w, x, y), full_output=True
        )
        self.c = res[0]
        self.cov_x = res[1]  # TODO
        ierr = res[4]

        if ierr not in [1, 2, 3, 4]:
            print("least square failed with error code: ", res)

    def get_parameter(self, x, y):
        return np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)

    def compute_tangent(self, t0, norm=False):
        v = np.array(self(t=t0, m=1))
        if norm:
            v /= np.linalg.norm(v, axis=0)
        return v

    def compute_angle(self, t=None):
        """
        Compute tangent angle for curve.
        """
        if t is None:
            t = self.t_o

        u, v = self.compute_tangent(t)
        return np.rad2deg(np.arctan2(u, v))

    def get_coefficients(self, c=None, m=0):

        if c is None:
            c = self.c

        k = c.size // 2
        cx = c[:k]
        cy = c[k:]

        if m != 0:
            cx = np.polyder(cx, m)
            cy = np.polyder(cy, m)

        return cx, cy

    def _compute_poly_curve(self, cx, cy, t):
        """
        Compute poly curve.
        """
        return np.polyval(cx, t), np.polyval(cy, t)

    def compute_normal(self, t0, x=None, y=None, t=None):

        x0, y0 = self(t=t0)

        # tangent vector
        v = self.compute_tangent(t0, norm=True)

        # rotate 90 degree
        n = np.dot([[0, -1], [1, 0]], v)

        cx = np.array([-n[0], x0])
        cy = np.array([-n[1], y0])

        if t is None:
            s = np.sign(v[0] * (y - y0) - v[1] * (x - x0))
            t = s * np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

        return self._compute_poly_curve(cx, cy, t)

    def _compute_curve(self, c, x=None, y=None, t=None, m=0):

        if t is None:
            t = self.get_parameter(x, y)
        else:
            if x is not None or y is not None:
                print("Warning: `x` and `y` will be ignored as `t` was given.")

        cx, cy = self.get_coefficients(c=c, m=m)

        x, y = self._compute_poly_curve(cx, cy, t)

        return x, y

    def __call__(self, x=None, y=None, t=None, m=0):
        return self._compute_curve(self.c, x, y, t, m)


def cubic_equation(a, b, c, d):
    """
    Find roots of cubic polynomial:
        a * x**3 + b * x**2 + c * x + d = 0
    """
    try:
        dtype = np.complex256
    except AttributeError:
        dtype = np.complex128
    a = np.asarray(a).astype(dtype)
    b = np.asarray(b).astype(dtype)
    c = np.asarray(c).astype(dtype)
    d = np.asarray(d).astype(dtype)

    d0 = b**2 - 3 * a * c
    d1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d

    C = ((d1 + np.sqrt(d1**2 - 4 * d0**3)) / 2.0) ** (1 / 3)

    xi = (-1.0 + np.sqrt(-3.0 + 0j)) / 2.0
    def s(k): return xi**k * C

    roots = [-1.0 / (3.0 * a) * (b + s(k) + d0 / s(k)) for k in range(3)]

    return np.array(roots)


def integral_sqrt_poly(x, a, b, c):
    """
    Integral over sqrt(a*x**2 + b*x + c)
    """
    s = np.sqrt(a * x**2 + b * x + c)

    A = (b + 2 * a * x) / (4 * a) * s
    B = (4 * a * c - b**2) / (8 * a ** (3 / 2))
    C = np.abs(2 * a * x + b + 2 * np.sqrt(a) * s)

    return A + B * np.log(C)


def compute_arc_length(curve, smin, smax):
    a, b = curve.get_coefficients()

    c0 = 4 * a[0] ** 2 + 4 * b[0] ** 2
    c1 = 4 * a[0] * a[1] + 4 * b[0] * b[1]
    c2 = a[1] ** 2 + b[1] ** 2

    smin = integral_sqrt_poly(smin, c0, c1, c2)
    smax = integral_sqrt_poly(smax, c0, c1, c2)

    return smax - smin


def compute_plume_coordinates(data, curve, which="centers"):
    """
    Computes along- and across-plume coordinates analytically
    if curve.degree == 2.

    Parameters
    ----------
    data : satellite data incl. x, y and plume_area
    curve : center curve
    which : process either pixel 'centers' or 'corners'.

    """
    if curve.degree != 2:
        raise ValueError("Degree of curve needs to be 2 not %d" % curve.degree)

    a, b = curve.get_coefficients()

    x_2d, y_2d = np.meshgrid(data.coords['x'], data.coords['y'])
    x_2d = data.copy(data=x_2d)
    y_2d = data.copy(data=y_2d)
    area = (~data.isnull())
    qx = x_2d.values[area]
    qy = y_2d.values[area]

    # coefficients for analytical solution
    c0 = 4 * a[0] ** 2 + 4 * b[0] ** 2
    c1 = 6 * a[0] * a[1] + 6 * b[0] * b[1]
    c2 = (
        4 * a[0] * a[2]
        - 4 * a[0] * qx
        + 2 * a[1] ** 2
        + 4 * b[0] * b[2]
        - 4 * b[0] * qy
        + 2 * b[1] ** 2
    )
    c3 = 2 * a[1] * a[2] - 2 * a[1] * qx + 2 * b[1] * b[2] - 2 * b[1] * qy

    roots = cubic_equation(c0, c1, c2, c3)
    real = np.abs(roots.imag) < 1e-6

    tmin = []
    n_no_solutions = 0
    n_multiple_solutions = 0

    for i in range(qx.size):

        n_solutions = np.sum(real[:, i])

        if n_solutions == 0:
            tmin.append(np.nan)
            n_no_solutions += 1

        elif n_solutions == 1:
            tmin.append(float(roots[:, i][real[:, i]].real))

        elif n_solutions > 1:
            # use shortest arc length (which might fail for strongly bend plumes)
            # using shortest distance fails, if curve bends back to source location
            j = np.argmin(roots[:, i].real)
            tmin.append(roots[j, i].real)

            n_multiple_solutions += 1

        else:
            raise ValueError

    if n_no_solutions > 0:
        name = " ".join(
            "%s" % v
            for v in [
                data.time.values,
                getattr(data, "orbit", "none"),
                getattr(data, "lon_eq", "none"),
            ]
        )
        print('No real solution for some points in "%s"' % name)

    if n_multiple_solutions > 0:
        pass

    tmin = np.array(tmin)
    px, py = curve(t=tmin)

    # sign of distance (negative left of curve from source)
    t = curve.get_parameter(qx, qy)
    v = curve.compute_tangent(t, norm=True)
    n = np.array([px - qx, py - qy])
    cross = np.cross(v, n, axis=0)
    sign = np.sign(cross)

    # compute distance
    distance = xr.full_like(x_2d, np.nan)
    distance.values[area] = sign * np.sqrt((px - qx) ** 2 + (py - qy) ** 2)

    # arc-length
    arc = xr.full_like(x_2d, np.nan)
    arc.values[area] = compute_arc_length(curve, curve.t_o, tmin)

    return arc, distance, tmin.min(), tmin.max()
