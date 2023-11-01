#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperCH4 developers
#
# This file is part of hyperch4.
#
# hyperch4 is a library to retrieve methane from hyperspectral satellite data
"""Streamlit app for calculating ch4 emission rates"""

import os
import sys
from glob import glob

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
from geopy.geocoders import Nominatim
from utils import calc_emiss, mask_data

sys.path.append('..')

st.set_page_config(
    page_title="Emission",
    page_icon="📈",
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
        gjs_filepath_list = glob(folderPath + '/**/*L2*.geojson', recursive=True)
        gjs_filepath_list = sorted(gjs_filepath_list, key=lambda x: os.path.basename(x))
        html_filepath_list = [gjs_str.replace('.geojson', '.html') for gjs_str in gjs_filepath_list]

        # load plume html file if it exists
        html_filepath_list = [glob(filepath.replace('L2', 'L3').replace('.html', '*plume*html'))[0]
                              if len(glob(filepath.replace('L2', 'L3').replace('.html', '*plume*html'))) > 0
                              else filepath
                              for filepath in html_filepath_list]

        # show basename in the selectbox
        filelist = [os.path.basename(file) for file in html_filepath_list]
        filename = st.selectbox("Pick L2 HTML file here:",
                                filelist,
                                index=0,
                                )

        # get the full path
        st.info(filename)
        index = filelist.index(filename)
        filename = html_filepath_list[index]
        gjs_filename = gjs_filepath_list[index]

        # read the plume source info
        geo_df = gpd.read_file(gjs_filename)
        geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]
        plume_dict = {}
        for index, loc in enumerate(geo_df_list):
            plume_dict[f'plume{index}'] = loc
        st.write(plume_dict)
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
    # --- Create plume mask --- #
    with st.form("mask_form"):
        # --- Generate plume mask by submitting the center location --- #
        st.info("Create CH$_4$ plume mask from selected plume.", icon="2️⃣")
        st.warning(
            'Don\'t need to run this again, if you already have the plume HTML file.', icon="☕️")

        # form of submitting the center of plume mask
        st.markdown('**Form for creating plume mask**')

        if plume_dict is not None:
            # input of several params
            pick_plume_name = st.selectbox("Pick plume here:",
                                           list(plume_dict.keys()),
                                           index=0,
                                           )

            niter = st.number_input("Set the number of iteration for dilation",
                                    min_value=0,
                                    step=1,
                                    value=1,
                                    format='%d'
                                    )

            size_median = st.number_input("Set the size for median filter",
                                          min_value=0,
                                          step=1,
                                          value=3,
                                          format='%d'
                                          )

            sigma_guass = st.number_input("Set the sigma for guassian filter",
                                          min_value=0,
                                          step=1,
                                          value=2,
                                          format='%d'
                                          )

            quantile_value = st.number_input("Set quantile for the low limit of plume mask",
                                             min_value=0.,
                                             max_value=1.,
                                             step=0.005,
                                             value=0.99,
                                             format='%f'
                                             )

            wind_source = st.selectbox("Pick wind source:",
                                       ['ERA5', 'GEOS-FP'],
                                       index=0,
                                       )

            wind_weights = st.checkbox('Whether apply the wind weights',
                                       value=True,
                                       )

            only_plume = st.checkbox('Whether only plot ch4 plume',
                                     value=True,
                                     )

        submitted = st.form_submit_button("Submit")

        # button for removing plume files
        clean_button = st.form_submit_button("Clean all mask files (nc, html, and png)")
        if clean_button:
            if 'plume' in filename:
                mask_files = glob(filename.replace('html', 'nc'))
                mask_files.extend([filename])
                mask_files.extend([filename.replace('html', 'png')])
            else:
                mask_files = glob(filename.replace('L2', 'L3').replace('.html', '_plume*nc'))
                mask_files.extend(glob(filename.replace('L2', 'L3').replace('.html', '_plume*html')))
                mask_files.extend(glob(filename.replace('.html', '_plume*png')))

            if len(mask_files) > 0:
                for file_mask in mask_files:
                    if os.path.exists(file_mask):
                        os.remove(file_mask)

            st.success('Removed all mask files.', icon="🗑️")

        if submitted:
            with st.spinner('Wait for it...'):
                # read the plume loc
                pick_loc = plume_dict[pick_plume_name]
                st.write('You picked location (lat, lon): ', str(pick_loc))
                latitude = pick_loc[0]
                longitude = pick_loc[1]

                # read L2 data
                if 'plume' in filename:
                    ds_name = '_'.join(filename.replace('L3', 'L2').split('_')[:-1]) + '.nc'
                else:
                    ds_name = filename.replace('.html', '.nc')
                with xr.open_dataset(ds_name) as ds:
                    # create mask and plume html file
                    mask, lon_mask, lat_mask, plume_html_filename = mask_data(filename, ds, longitude, latitude, pick_plume_name,
                                                                              wind_source, wind_weights, niter,
                                                                              size_median, sigma_guass, quantile_value,
                                                                              only_plume)

                # export masked data (plume)
                if 'plume' in filename:
                    plume_nc_filename = filename.replace('.html', '.nc')
                else:
                    plume_nc_filename = filename.replace('.html', f'_{pick_plume_name}.nc').replace('L2', 'L3')
                ch4_mask = ds['ch4'].where(mask)
                xr.merge([ch4_mask, ds['u10'], ds['v10'], ds['sp']]).to_netcdf(plume_nc_filename)

            st.success(
                f'Exported to: \n \n {plume_html_filename} \n \n You can type "R" to refresh this page for checking the plume mask.', icon="✅")

with col3:
    with st.form("emiss_form"):
        # --- Create emission rate --- #
        st.info('Estimating the CH$_4$ emission rate using IME method', icon="3️⃣")

        if plume_dict is not None:
            # pick plume
            pick_plume_name = st.selectbox("Pick plume here:",
                                           list(plume_dict.keys()),
                                           index=0,
                                           )

        # input alphas for calculating U_eff
        st.success('U_eff = alpha1 * np.log(wind_speed_average) + alpha2 + alpha3 * wind_speed_average', icon='🧮')
        alpha1 = st.number_input('alpha1 for Ueff', value=0.0, format='%f')
        alpha2 = st.number_input('alpha2 for Ueff (area source: 0.66, point source: 0.42)', value=0.66, format='%f')
        alpha3 = st.number_input('alpha3 for Ueff', value=0.34, format='%f')

        wspd = st.number_input(
            'Manual wspd [m/s] (please leave this as "None", if you use the reanalysis wind data)', value=None, format='%f')

        # sitename for csv export
        sitename = st.text_input('Sitename (any name you like)', value='')

        # platform for csv output
        platform = st.selectbox('Platform', ('EnMAP', 'EMIT', 'PRISMA'))

        # set pixel resolution
        if platform == 'EMIT':
            instrument = 'emi'
            provider = 'NASA-JPL'
            pixel_res = 60  # meter

        elif platform in ['EnMAP', 'PRISMA']:
            instrument = 'hsi'
            provider = 'DLR'
            pixel_res = 30  # meter

        # source type
        source_tropomi = st.selectbox('Whether TROPOMI captures the source (0: False; 1: True)', [0, 1, None], index=1)
        source_trace = st.selectbox(
            'Whether ClimateTrace includes the source (0: False; 1: True)', [0, 1, None], index=1)

        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.spinner('Calculating emission rate ...'):
                # set output name
                if 'plume' in filename:
                    plume_nc_filename = filename.replace('.html', '.nc')
                else:
                    plume_nc_filename = filename.replace('L2', 'L3').replace('.html', f'_{pick_plume_name}.nc')

                # calculate emissions
                wspd, wdir, l_eff, u_eff, Q, Q_err,\
                    err_random, err_wind, err_shape = calc_emiss(plume_nc_filename, pick_plume_name,
                                                                 pixel_res=pixel_res,
                                                                 alpha1=alpha1,
                                                                 alpha2=alpha2,
                                                                 alpha3=alpha3,
                                                                 wind_source=wind_source,
                                                                 wspd=wspd,
                                                                 )

                # print the emission data
                st.warning(f'''The CH$_4$ emission rate is {Q:.2f} $\pm$ {Q_err:.2f} kg/h.
                               [
                               U$_{{eff}}$: {u_eff:.2f} m/s,
                               L$_{{eff}}$: {l_eff:.2f} m,
                               err_random: {err_random:.2f} kg/h,
                               err_wind: {err_wind:.2f} kg/h,
                               err_shape: {err_shape:.2f} kg/h,
                               ]
                           ''', icon="🔥")

                # calculate plume bounds
                with xr.open_dataset(plume_nc_filename) as ds:
                    lon_mask = ds['longitude']
                    lat_mask = ds['latitude']
                    t_overpass = pd.to_datetime(ds['ch4'].attrs['start_time'])

                bounds = [lon_mask.min(skipna=True).item(), lat_mask.min(skipna=True).item(),
                          lat_mask.min(skipna=True).item(), lat_mask.max(skipna=True).item()]

                # get the location attrs
                geolocator = Nominatim(user_agent='hyper')
                location = geolocator.reverse(
                    f'{plume_dict[pick_plume_name][0]}, {plume_dict[pick_plume_name][1]}', exactly_one=True, language='en')
                address = location.raw['address']

                # save ime results
                ime_results = {
                    'plume_id': f"{instrument}-{t_overpass.strftime('%Y%m%dt%H%M%S')}-{pick_plume_name}",
                                'plume_latitude': plume_dict[pick_plume_name][0],
                                'plume_longitude': plume_dict[pick_plume_name][1],
                                'datetime': t_overpass.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                'country': address.get('country', ''),
                                'state': address.get('state', ''),
                                'city': address.get('city', ''),
                                'name': sitename,
                                'gas': 'CH4',
                                'cmf_type': 'mf',
                                'plume_bounds': [bounds],
                                'instrument': instrument,
                                'platform': platform,
                                # 'provider': provider,
                                'emission_auto': Q,
                                'emission_uncertainty_auto': Q_err,
                                'emission_uncertainty_random_auto': err_random,
                                'emission_uncertainty_wind_auto': err_wind,
                                'emission_uncertainty_shape_auto': err_shape,
                                'wind_speed_avg_auto': wspd,  # u10
                                'wind_direction_avg_auto': wdir,
                                'wind_source_auto': 'era5',
                                'ueff_ime': u_eff,
                                'alpha1': alpha1,
                                'alpha2': alpha2,
                                'alpha3': alpha3,
                                'niter': niter,
                                'quantile': quantile_value,
                                'size_median': size_median,
                                'sigma_guass': sigma_guass,
                                'wind_weights': wind_weights,
                                'source_tropomi': source_tropomi,
                                'source_trace': source_trace,
                }

                # convert to DataFrame and export data as csv file
                df = pd.DataFrame(data=ime_results, index=[0])
                savename = plume_nc_filename.replace('nc', 'csv')
                savepath = os.path.join(os.path.dirname(filename), savename)
                df.to_csv(savepath, index=False)
                st.success(f'Results are exported to \n \n {savepath}')
