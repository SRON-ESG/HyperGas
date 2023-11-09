#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Plot orthorectified CH4 L2B NetCDF products and export them as png/html."""

import gc
import logging
import multiprocessing
import os
import warnings
from glob import glob
from itertools import chain
from pathlib import Path

import xarray as xr
from hyperch4.folium_map import Map

warnings.filterwarnings('ignore')

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# set filename pattern to load data automatically
PATTERNS = ['ENMAP01-____L2B*.nc', 'EMIT_L2B*.nc', 'PRS_L2_*.nc']

# the vriabels to be plotted
VARNAMES = ['rgb', 'ch4', 'ch4_comb', 'ch4_denoise', 'ch4_comb_denoise']


class L2B_plot():
    """The L2B plotting class."""

    def __init__(self, filename):
        """Init the class"""
        # set the map to None
        self.m = None
        self.filename = filename

    def _load(self):
        """Load the L2B data"""
        LOG.info(f'Reading {self.filename}')
        self.ds = xr.open_dataset(self.filename)

    def make_map(self):
        """"Make the background folium map"""
        LOG.debug(f'Create the map for {VARNAMES}')
        self.m = Map(self.ds, VARNAMES)
        self.m.initialize()

    def plot(self):
        """Plot data on folium map."""
        LOG.debug('Plot the data on map')
        # the length of `show_layers` and `opacities` should be as same as VARNAMES
        self.m.plot(show_layers=[False]*(len(VARNAMES)-1)+[True], opacities=[0.9]+[0.7]*(len(VARNAMES)-1))


def get_dirs(root_dir):
    """Get all lowest directories"""
    lowest_dirs = list()

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d[0] == '.']
        if files and not dirs:
            lowest_dirs.append(root)

    return list(sorted(lowest_dirs))


def plot_data(filelist, len_chunklist, index):
    """Read data from filelist and plot data"""
    # initialize two classes with the first file
    l2b_map_allinone = L2B_plot(filelist[0])

    for filename in filelist:
        # use the filename as savename with html suffix
        savename = str(Path(filename).with_suffix('.html'))

        # load data
        LOG.info(f'Processing {filename}')
        l2b_map = L2B_plot(filename)
        l2b_map.filename = filename
        l2b_map._load()
        l2b_map_allinone.ds = l2b_map.ds

        # 1. one map for each file
        # make a new map everytime and plot data
        LOG.info('Making map for single file')
        l2b_map.make_map()
        l2b_map.plot()
        l2b_map.m.export(savename)
        del l2b_map.ds, l2b_map.m, l2b_map
        gc.collect()

        # 2. one map for all files
        LOG.info('Making a whole map (to be exported at the end)')
        if l2b_map_allinone.m is None:
            # make the background map only once
            l2b_map_allinone.make_map()
        else:
            # if the background map is already there, just replace the dataset to be plotted
            l2b_map_allinone.m.ds = l2b_map_allinone.ds

        # plot the variables
        l2b_map_allinone.plot()
        del l2b_map_allinone.m.ds, l2b_map_allinone.ds
        gc.collect()

    # use the folder name as html name
    savename_all = os.path.join(os.path.dirname(filelist[0]), Path(filelist[0]).parent.parts[-1] + '.html')

    # add index to the html file if we have multiple chunks
    if len_chunklist > 1:
        savename_all = savename_all.replace('.html', f'_{index}.html')

    # export combined html file
    l2b_map_allinone.m.export(savename_all)
    del l2b_map_allinone.m, l2b_map_allinone
    gc.collect()


def main():
    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    # split the list into chunks in case of RAM error
    chunk = 8
    filelist_chunk = [filelist[i:i+chunk] for i in range(0, len(filelist), chunk)]
    len_chunklist = len(filelist_chunk)

    for index, filelist in enumerate(filelist_chunk):
        # I have tried to del var and release memory, but it doesn't work
        #   so I use the process trick here.
        p = multiprocessing.Pool(1)
        p.starmap(plot_data, [(filelist, len_chunklist, index)])
        p.terminate()
        p.join()


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI/'
    lowest_dirs = get_dirs(root_dir)

    for data_dir in lowest_dirs:
        LOG.info(f'Plotting data under {data_dir}')
        main()
