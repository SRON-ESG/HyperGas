#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022-2023 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Get the TLE file at specific time"""

import os

import spacetrack.operators as op
import yaml
from spacetrack import SpaceTrackClient

# NORAD Catalog Numbers (https://celestrak.com/satcat/search.php)
norad_cat_id = {'ENMAP': 52159, 'PRISMA': 44072}  # use upper case for platform_name


class TLE():
    """Get the TLE list for satellite observation."""

    def __init__(self, id):
        # load settings
        _dirname = os.path.dirname(__file__)
        with open(os.path.join(_dirname, 'config.yaml')) as f:
            settings = yaml.safe_load(f)

        username = settings['spacetrack_usename']
        password = settings['spacetrack_password']

        # connect to the client
        self.client = SpaceTrackClient(identity=username, password=password)

        # get the NORAD id
        self.norad_cat_id = norad_cat_id[id.upper()]

    def get_tle(self, start_date, end_date):
        """Get the TLE content as list

        start_date (datetime): Beginning of observation datatime
        end_date (datetime): End of observation datatime
        """
        # create epoch
        epoch = op.inclusive_range(start_date, end_date)

        # request the tle lines
        tles = self.client.tle(norad_cat_id=self.norad_cat_id,
                               epoch=epoch,
                               format='tle',
                               orderby=['epoch']).split('\n')[:-1]

        return tles
