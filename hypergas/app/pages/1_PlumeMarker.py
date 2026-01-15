#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Streamlit app for creating plume markers"""

import os
import sys
from glob import glob
import geopandas as gpd

import streamlit as st
import streamlit.components.v1 as components

sys.path.append('..')

st.set_page_config(
    page_title="PlumeMarker",
    page_icon=":round_pushpin:",
    layout="wide",
    initial_sidebar_state="expanded",
)

col1, col2 = st.columns([7, 3])

with col2:
    # --- Load data and plot it over background map --- #
    st.info('Load data and check the quickview of map and CH$_4$', icon="1️⃣")

    # set the folder path
    folderPath = st.text_input('**Enter L2 folder path:**')

    if folderPath:
        # get all html files recursively
        html_list = glob(folderPath + '/**/*L2*.html', recursive=True)
        html_list = sorted(html_list, key=lambda x: os.path.basename(x))

        # show basename in the selectbox
        filelist = [os.path.basename(file) for file in html_list]
        # we do not want to read the html of masked plume
        filelist = [file for file in filelist if 'plume' not in file]
        filename = st.selectbox("Pick L2 HTML file here:",
                                filelist,
                                index=0,
                                )

        # get the full path
        st.success(filename)
        index = filelist.index(filename)
        filename = html_list[index]

        # upload file if you are using remote server
        st.info('Add markers on the left, click Export button, and then upload the geojson file', icon="2️⃣")
        uploaded_file = st.file_uploader("Upload the corresponding geojson file to the folder path above",
                                         accept_multiple_files=False,
                                         )

        # write the jason file to server
        if uploaded_file is not None:
            with open(filename.replace('.html', '.geojson'), 'wb') as output_file:
                st.success(f'Uploaded ;) Please click the cross button in case files are messed up.')
                output_file.write(uploaded_file.getbuffer())

    else:
        filename = None
        plume_dict = None

if filename is not None:
    with col1:
        # read html file
        HtmlFile = open(filename, 'r', encoding='utf-8')
        source_code = HtmlFile.read()

        # add to map
        components.html(
            source_code, width=None, height=600, scrolling=False
        )

col3, col4 = st.columns([7, 3])

with col3:
    # --- print marker dict if geojson file exists --- #
    if filename is not None:
        gjs_filename = filename.replace('.html', '.geojson')
        if os.path.exists(gjs_filename):
            geo_df = gpd.read_file(gjs_filename)
            geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]
            plume_dict = {}
            for index, loc in enumerate(geo_df_list):
                plume_dict[f'plume{index}'] = loc
            st.warning('You have already generated markers before. Please take care if you wanna recreate markers.')
            st.write(plume_dict)
