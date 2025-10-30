#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Get the Two Line Element (TLE) file at specific time."""

import os

import spacetrack.operators as op
import yaml
from spacetrack import SpaceTrackClient

# NORAD Catalog Numbers (https://celestrak.com/satcat/search.php)
norad_cat_id = {'ENMAP': 52159, 'PRISMA': 44072}  # use upper case for platform_name


class TLE():
    """Get the TLE list for satellite observation using `spacetrack <https://spacetrack.readthedocs.io/>`_."""

    def __init__(self, id):
        # load settings
        _dirname = os.path.dirname(__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)

        username = settings['data']['spacetrack_usename']
        password = settings['data']['spacetrack_password']

        # connect to the client
        self.client = SpaceTrackClient(identity=username, password=password)

        # get the NORAD id
        self.norad_cat_id = norad_cat_id[id.upper()]

    def get_tle(self, start_date, end_date):
        """Get the TLE content as list.

        Parameters
        ----------
        start_date : datetime
            Beginning of observation datatime.
        end_date : datetime
            End of observation datatime.

        Returns
        -------
        tles : TLE data in lines.
        """
        # create epoch
        epoch = op.inclusive_range(start_date, end_date)

        # request the tle lines
        tles = self.client.tle(norad_cat_id=self.norad_cat_id,
                               epoch=epoch,
                               format='tle',
                               orderby=['epoch']).split('\n')[:-1]

        return tles
