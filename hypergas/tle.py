#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Get the Two Line Element (TLE) file at specific time."""

import os
from datetime import datetime, timedelta

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

    @staticmethod
    def parse_tle_epoch(tle_line1):
        """Parse the epoch from TLE Line 1.

        The epoch is in columns 19-32 of Line 1 in the format: YYDDD.DDDDDDDD
        where YY is the year, DDD is the day of year, and .DDDDDDDD is the fractional day.

        Parameters
        ----------
        tle_line1 : str
            The first line of the TLE.

        Returns
        -------
        epoch : datetime
            The epoch time of the TLE.
        """
        # Extract epoch string (columns 18-32, 0-indexed so 18-32)
        epoch_str = tle_line1[18:32].strip()

        # Parse year (first 2 digits)
        year_2digit = int(epoch_str[:2])
        # Convert 2-digit year to 4-digit (00-56 -> 2000-2056, 57-99 -> 1957-1999)
        year = 2000 + year_2digit if year_2digit < 57 else 1900 + year_2digit

        # Parse day of year and fractional day
        day_of_year_float = float(epoch_str[2:])
        day_of_year = int(day_of_year_float)
        fractional_day = day_of_year_float - day_of_year

        # Create datetime for Jan 1 of the year, then add days
        epoch = datetime(year, 1, 1) + timedelta(days=day_of_year - 1,
                                                 seconds=fractional_day * 86400)

        return epoch

    def get_tle(self, start_date, end_date, overpass_time=None):
        """Get the TLE content as list, optionally selecting the closest to overpass time.

        Parameters
        ----------
        start_date : datetime
            Beginning of observation datatime.
        end_date : datetime
            End of observation datatime.
        overpass_time : datetime, optional
            If provided, returns only the TLE closest to this time.
            If None, returns all TLEs in the time range.

        Returns
        -------
        tles : TLE data in lines. If overpass_time is provided, returns a list of 2 lines
               (line1, line2) for the closest TLE. Otherwise, returns all TLE lines.
        """
        # create epoch range
        epoch = op.inclusive_range(start_date, end_date)

        # request the tle lines using gp_history
        tle_data = self.client.gp_history(norad_cat_id=self.norad_cat_id,
                                          epoch=epoch,
                                          orderby=['epoch'],
                                          format='tle')

        # Split into lines and filter out empty lines
        tles = [line for line in tle_data.split('\n') if line.strip()]

        if len(tles) == 0:
            return None if overpass_time else []

        # If no overpass time specified, return all TLEs
        if overpass_time is None:
            return tles

        # Find the TLE closest to overpass_time
        min_time_diff = None
        closest_tle = None

        # Process TLEs in pairs (line1, line2)
        for i in range(0, len(tles), 2):
            if i + 1 >= len(tles):
                break

            line1 = tles[i]
            line2 = tles[i+1]

            # Parse the TLE epoch from line 1
            tle_epoch = self.parse_tle_epoch(line1)

            # Calculate time difference
            time_diff = abs((tle_epoch - overpass_time).total_seconds())

            if min_time_diff is None or time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_tle = [line1, line2]

        return closest_tle
