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
import pkgutil
import warnings
from glob import glob
from itertools import chain
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

from hypergas.folium_map import Map
from utils import get_dirs

warnings.filterwarnings('ignore')

# set the logger level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
LOG = logging.getLogger(__name__)

# set filename pattern to load data automatically
PATTERNS = ['ENMAP01-____L2B*.nc', 'EMIT_L2B*.nc', 'PRS_L2_*.nc']

# modify the settings
SPECIES = 'ch4'  # 'ch4' or 'co2'
VMAX = 300   # suggested setting: 300 for ch4 (ppb), 30 for co2 (ppm)
VARNAMES = ['rgb', 'radiance_2100', SPECIES, f'{SPECIES}_comb', f'{SPECIES}_denoise', f'{SPECIES}_comb_denoise']


class L2B_plot():
    """The L2B plotting class."""

    def __init__(self, filename, df_marker):
        """Init the class"""
        # set the map to None
        self.m = None
        self.filename = filename
        self.df_marker = df_marker

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
        self.m.plot(show_layers=[False]*(len(VARNAMES)-1)+[True],
                    opacities=[0.9]+[0.7]*(len(VARNAMES)-1),
                    df_marker=self.df_marker,
                    vmax=VMAX
                    )


def read_markers():
    # read pre-saved markers if it exists
    path_hyper = os.path.dirname(pkgutil.get_loader('hypergas').path)
    with open(os.path.join(path_hyper, 'config.yaml')) as f:
        settings = yaml.safe_load(f)

    filename_marker = os.path.join(path_hyper, settings['markers_filename'])
    if os.path.exists(filename_marker):
        df_marker = pd.read_csv(filename_marker)
    else:
        df_marker = None

    return df_marker


def plot_data(filelist, df_marker, len_chunklist, index):
    """Read data from filelist and plot data"""
    # initialize two classes with the first file
    l2b_map_allinone = L2B_plot(filelist[0], df_marker)

    for filename in filelist:
        # use the filename as savename with html suffix
        savename = str(Path(filename).with_suffix('.html'))

        # load data
        LOG.info(f'Processing {filename}')
        l2b_map = L2B_plot(filename, df_marker)
        l2b_map.filename = filename
        l2b_map._load()
        l2b_map_allinone.ds = l2b_map.ds

        # 1. one map for each file
        # make a new map everytime and plot data
        LOG.info('Making map for single file')
        l2b_map.make_map()
        l2b_map.plot()
        l2b_map.m.export(savename)

        # get scene center
        scene_center = l2b_map.m.center_map
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

        # update scene center
        l2b_map_allinone.m.center_map = scene_center

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


def main(chunk=8, skip_exist=True, plot_markers=False):
    # get the filname list
    filelist = list(chain(*[glob(os.path.join(data_dir, pattern), recursive=True) for pattern in PATTERNS]))
    filelist = list(sorted(filelist))

    if len(filelist) == 0:
        return

    # split the list into chunks in case of RAM error
    filelist_chunk = [filelist[i:i+chunk] for i in range(0, len(filelist), chunk)]
    len_chunklist = len(filelist_chunk)

    if plot_markers:
        df_marker = read_markers()
    else:
        df_marker = None

    if skip_exist:
        # get the html names which should be exported after running the plotting script
        html_files = [filename.replace('.nc', '.html') for filename in filelist]

        # check if all html files exist
        all_html_exist = all([os.path.isfile(filename) for filename in html_files])
        html_dir = os.path.dirname(filelist[0])

        if all_html_exist:
            skip_exist = True
            LOG.info(f'Skip plotting files under {html_dir} which already contains all L2 html files')
        else:
            skip_exist = False
            LOG.info(f'Replotting all L2 data under {html_dir} which does not contain all L2 html files')

    if not skip_exist:
        for index, filelist in enumerate(filelist_chunk):
            # I have tried to del var and release memory, but it doesn't work
            #   so I use the process trick here.
            p = multiprocessing.Pool(1)
            p.starmap(plot_data, [(filelist, df_marker, len_chunklist, index)])
            p.terminate()
            p.join()


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI_plume/'
    lowest_dirs = get_dirs(root_dir)

    # whether skip dir which contains exported html
    skip_exist = True

    # whether plot pre-saved markers on map
    plot_markers = False

    # the chunk of files for each html file
    #   don't set it too high if you meet RAM error
    chunk = 4

    for data_dir in lowest_dirs:
        LOG.info(f'Plotting data under {data_dir}')
        main(chunk, skip_exist, plot_markers)
