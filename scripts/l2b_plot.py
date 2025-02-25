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
from shapely import Point

import pandas as pd
import xarray as xr
import yaml

import hypergas
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

# --- settings --- #
# whether combine all htmls into one html
#   this is useful to compare plumes on different days
map_allinone = True

# set species to be plotted (3 formats)
#   'all': retrieve all supported gases
#   single gas name str, e.g. 'ch4'
#   list of gas names, e.g. ['ch4', 'co2']
species = 'all'

# colorbar vmax of species
vmax = None  # This can be None or single value (None: vmax will be set automatically for different species; single value: it will overwrite all vmax)
# --- settings --- #

# load settings
_dirname = os.path.dirname(hypergas.__file__)
with open(os.path.join(_dirname, 'config.yaml')) as f:
    settings = yaml.safe_load(f)
species_setting = settings['species']

# set variables automatically
if species == 'all':
    species_list = list(species_setting.keys())
elif type(species) is list:
    species_list = species
elif type(species) is str:
    species_list = [species]
else:
    raise ValueError(f"species should be one str or list of str. {species} is not supported")

species_vnames = []
for species in species_list:
    species_vnames.extend([species, f'{species}_comb', f'{species}_denoise', f'{species}_comb_denoise', f'{species}_mask'])

varnames = ['rgb', 'radiance_2100']
varnames.extend(species_vnames)


class L2B_plot():
    """The L2B plotting class."""

    def __init__(self, filename, df_marker):
        """Init the class"""
        # set the map to None
        self.m = None
        self.filename = filename
        self.df_marker = df_marker
        self.varnames = varnames

    def _load(self, bbox=None):
        """Load the L2B data"""
        LOG.info(f'Reading {self.filename}')
        self.ds = xr.open_dataset(self.filename)

        if bbox:
            # subset to zoom-in region
            zoom_mask = (self.ds['longitude'] > bbox[0]) & (self.ds['longitude'] < bbox[2]) \
                            & (self.ds['latitude'] > bbox[1]) & (self.ds['latitude'] < bbox[3])
            self.ds = self.ds.where(zoom_mask, drop=True)

    def make_map(self):
        """"Make the background folium map"""
        # only pick available varnames
        available_varnames = list(self.ds.data_vars)
        self.varnames = [name for name in self.varnames if name in available_varnames]

        LOG.info(f'Create the map for {self.varnames}')
        self.m = Map(self.ds, self.varnames)
        self.m.initialize()

    def plot(self, pre_suffix=''):
        """Plot data on folium map."""
        LOG.debug('Plot the data on map')
        # the length of `show_layers` and `opacities` should be as same as varnames
        self.m.plot(show_layers=[False]*(len(self.varnames)-2)+[True, False],
                    opacities=[0.9]+[0.7]*(len(self.varnames)-1),
                    df_marker=self.df_marker,
                    vmax=vmax,
                    pre_suffix=pre_suffix,
                    )


def read_markers():
    # read pre-saved markers if it exists
    path_hyper = os.path.dirname(pkgutil.get_loader('hypergas').path)
    with open(os.path.join(path_hyper, 'config.yaml')) as f:
        settings = yaml.safe_load(f)

    filename_marker = os.path.join(path_hyper, settings['data']['markers_filename'])
    if os.path.exists(filename_marker):
        df_marker = pd.read_csv(filename_marker)
    else:
        df_marker = None

    return df_marker


def get_zoom_bbox(pad, lat_zoom, lon_zoom):
    """Calculate the lon/lat boundary for zoom-in images"""
    if pad:
        if (lat_zoom != None) & (lon_zoom != None):
            point_zoom = Point(lon_zoom, lat_zoom)

            # boundary box: min x, min y, max x, max y
            bbox = point_zoom.buffer(pad).bounds

            return bbox
        else:
            LOG.warning(f'lat_zoom ({lat_zoom}) and lon_zoom ({lon_zoom}) should be numbers. Plotting the whole scene instead ...')
            return None
    else:
        return None


def plot_data(filelist, df_marker, len_chunklist, index, bbox):
    """Read data from filelist and plot data"""
    # initialize two classes with the first file
    l2b_map_allinone = L2B_plot(filelist[0], df_marker)

    for filename in filelist:
        # use the filename as savename with html suffix
        savename = str(Path(filename).with_suffix('.html'))
        if bbox:
            pre_suffix = '_zoomin'
        else:
            pre_suffix = ''

        # load data
        LOG.info(f'Processing {filename}')
        l2b_map = L2B_plot(filename, df_marker)
        l2b_map.filename = filename
        l2b_map._load(bbox)
        l2b_map_allinone.ds = l2b_map.ds

        # 1. one map for each file
        # make a new map everytime and plot data
        LOG.info('Making map for single file')
        l2b_map.make_map()
        l2b_map.plot(pre_suffix=pre_suffix)
        l2b_map.m.export(savename, pre_suffix=pre_suffix)

        # get scene center
        scene_center = l2b_map.m.center_map
        del l2b_map.ds, l2b_map.m, l2b_map
        gc.collect()

        if map_allinone:
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
    if map_allinone:
        l2b_map_allinone.m.export(savename_all, pre_suffix=pre_suffix)
        del l2b_map_allinone.m, l2b_map_allinone
        gc.collect()


def main(chunk=8, skip_exist=True, plot_markers=False, bbox=None):
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
            p.starmap(plot_data, [(filelist, df_marker, len_chunklist, index, bbox)])
            p.terminate()
            p.join()


if __name__ == '__main__':
    # root dir of hyper data
    root_dir = '/data/xinz/Hyper_TROPOMI_landfill/'
    lowest_dirs = get_dirs(root_dir)

    # whether skip dir which contains exported html
    skip_exist = True

    # whether plot pre-saved markers on map
    plot_markers = False


    # the chunk of files for each html file
    #   don't set it too high if you meet RAM error
    chunk = 3

    # pad (degree) around the specific location
    #   if it is None, we will plot the whole scene
    pad = None # e.g. 0.05
    lat_zoom = None # float
    lon_zoom = None # float
    bbox = get_zoom_bbox(pad, lat_zoom, lon_zoom)

    for data_dir in lowest_dirs:
        LOG.info(f'Plotting data under {data_dir}')
        main(chunk, skip_exist, plot_markers, bbox)
