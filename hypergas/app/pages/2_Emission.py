#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Streamlit app for calculating trace gas emission rates"""

import itertools
import os
import sys
from glob import glob

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr

import hypergas
from hypergas.plume_utils import a_priori_mask_data, cm_mask_data
from hypergas.emiss import Emiss

sys.path.append('..')

st.set_page_config(
    page_title="Emission",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

col1, col2 = st.columns([6, 3])


with col2:
    # --- Load data and plot it over background map --- #
    st.info('Load data and check the quickview of map and trace gases', icon="1️⃣")

    # set the folder path
    folderPath = st.text_input('**Enter L2 folder path:**')

    if folderPath:
        # get all geojson files recursively
        gjs_filepath_list = glob(folderPath + '/**/*L2*.geojson', recursive=True)
        gjs_filepath_list = sorted(gjs_filepath_list, key=lambda x: os.path.basename(x))

        # load html files
        html_filepath_list = [gjs_str.replace('.geojson', '.html') for gjs_str in gjs_filepath_list]

        # whether only load plume html files
        only_plume_html = st.toggle('I only want to check plume html files.')

        # the filename could be *L(2/3)*(plume_<num>).html
        #   L2 is the original file, L3*plume is the masked plume file
        if only_plume_html:
            html_filepath_list = [glob(filepath.replace('L2', '*').replace('.html', '*plume*html'))
                                  for filepath in html_filepath_list]
        else:
            html_filepath_list = [glob(filepath.replace('L2', '*').replace('.html', '*html'))
                                  for filepath in html_filepath_list]

        # join sublists into one list
        html_filepath_list = sorted(list(itertools.chain(*html_filepath_list)))

        # show basename in the selectbox
        filelist = [os.path.basename(file) for file in html_filepath_list]
        filename = st.selectbox(f"**Pick HTML file here: (totally {len(filelist)})**",
                                filelist,
                                index=0,
                                )

        # get the full path
        st.info(filename)
        index = filelist.index(filename)
        filename = html_filepath_list[index]

        # read the plume source info from geojson file
        prefix = os.path.join(os.path.dirname(filename), os.path.basename(
            filename.replace('L3', 'L2')).split('_plume')[0]).replace('.html', '')
        gjs_filename = list(filter(lambda x: prefix in x, gjs_filepath_list))[0]
        geo_df = gpd.read_file(gjs_filename)
        geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

        # save into dict for streamlit print
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
            source_code, width=None, height=800, scrolling=False
        )

col3, col4 = st.columns([6, 3])

# set default params which can be modified from the form
params = {'gas': 'CH4',
          'wind_source': None, 'land_mask_source': 'OSM', 'land_only': True,
          'wind_speed': None, 'surface_pressure': None,
          'azimuth_diff_max': 30., 'dist_max': 180.,
          'name': '', 'ipcc_sector': 'Solid Waste (6A)',
          'platform': None, 'source_tropomi': True, 'source_trace': False
          }

# copy the params, this should not be modified
defaults = params.copy()

with col3:
    if filename is not None:
        # --- print existed mask info --- #
        if 'plume' in os.path.basename(filename):
            # show plume scientific image if it exists
            filename_l3_png = filename.replace('.html', '.png')
            if os.path.exists(filename_l3_png):
                st.image(filename_l3_png, caption='IME and CSF results', width=800)
                basename_l3_png = os.path.basename(filename_l3_png)
                st.download_button(
                    label=f'Download {basename_l3_png}',
                    data=open(filename_l3_png, 'rb').read(),
                    file_name=basename_l3_png,
                    mime='image/png',
                )

            # show DataFrame
            file_mask_exist = glob(filename.replace('.html', '.csv'))
            csv_file = filename.replace('.html', '.csv')
            df = pd.read_csv(csv_file)
            st.dataframe(df.T, use_container_width=True)

            # read settings to use directly
            for key in params.keys():
                if key in df.keys():
                    params.update({key: df[key].item()})

            # set the toggle in case users want to create a new mask
            plume_toggle = st.toggle('I still want to create new plume mask.')
        else:
            plume_toggle = True
    else:
        plume_toggle = True
        # copy the default one
        params = defaults.copy()

    if plume_toggle:
        # --- Create plume mask --- #
        with st.form("mask_form"):
            # --- Generate plume mask by submitting the center location --- #
            st.info("Create gas plume mask from selected plume marker.", icon="2️⃣")
            st.warning(
                'Don\'t need to run this again, if you already have the plume HTML file.', icon="☕️")

            # form of submitting the center of plume mask
            st.markdown('**Form for creating plume mask**')

            if plume_dict is not None:
                # input of several params
                # trace gas name
                gases = ('CH4', 'CO2')
                gas = st.selectbox('Trace Gas', gases, index=gases.index(params['gas'])).lower()

                plume_names = list(plume_dict.keys())
                if 'plume' in os.path.basename(filename):
                    file_mask_exist = glob(filename.replace('.html', '.csv'))[0]
                    pick_plume_name_default = file_mask_exist.split('_')[-1][:-4]
                else:
                    pick_plume_name_default = plume_names[0]
                pick_plume_name = st.selectbox("Pick plume here:",
                                               plume_names,
                                               index=plume_names.index(pick_plume_name_default),
                                               )

                wind_source_names = ['ERA5', 'GEOS-FP']
                if params['wind_source'] is None:
                    wind_source = st.selectbox("Pick wind source:",
                                               wind_source_names,
                                               index=0,
                                               )
                else:
                    wind_source = st.selectbox("Pick wind source:",
                                               wind_source_names,
                                               index=wind_source_names.index(params['wind_source']),
                                               )

                azimuth_diff_max = st.number_input('Maximum value of azimuth difference \
                        \n (Please keep the default value, unless there are obvious false plume masks around)',
                                                   value=params['azimuth_diff_max'], format='%f')

                dist_max = st.number_input('Maximum value of dilation distance (meter) \
                        \n (Please keep the default value, unless there are obvious false plume masks around)',
                                           value=params['dist_max'], format='%f')

                only_plume = st.checkbox('Whether only plot plume',
                                         value=True,
                                         )

                land_only = st.checkbox('Whether only considering land pixels',
                                        value=params['land_only'],
                                        )

                land_mask_source_list = ['GSHHS', 'OSM', 'Natural Earth']
                land_mask_source = st.selectbox("Pick data source for creating land mask:",
                                                land_mask_source_list,
                                                index=land_mask_source_list.index(params['land_mask_source']),
                                                )

            submitted = st.form_submit_button("Submit")

            # button for removing plume files
            clean_button = st.form_submit_button("Clean all mask files (nc, html, and png)")
            if clean_button:
                if 'plume' in os.path.basename(filename):
                    mask_files = glob(filename.replace('.html', '.nc'))
                    mask_files.extend([filename])
                    mask_files.extend([filename.replace('.html', '.png')])
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
                    if 'plume' in os.path.basename(filename):
                        ds_name = '_'.join(filename.replace('L3', 'L2').split('_')[:-1]) + '.nc'
                    else:
                        ds_name = filename.replace('.html', '.nc')

                    with xr.open_dataset(ds_name, decode_coords='all') as ds:
                        # create mask and plume html file
                        mask, lon_mask, lat_mask, longitude, latitude, plume_html_filename = a_priori_mask_data(ds, gas, longitude, latitude,
                                                                                                                pick_plume_name, wind_source,
                                                                                                                only_plume, azimuth_diff_max,
                                                                                                                dist_max, filename
                                                                                                                )
                        cm_mask, cm_threshold = cm_mask_data(ds, gas, longitude, latitude)

                        # mask data
                        gas_mask = ds[gas].where(mask)
                        with xr.set_options(keep_attrs=True):
                            gas_cm_mask = ds[gas].where(cm_mask).rename(f'{gas}_cm')  # - cm_threshold
                        gas_cm_mask.attrs['description'] = gas_cm_mask.attrs['description'] + \
                            ' masked by the Carbon Mapper v2 method'

                        # calculate mean wind and surface pressure in the plume if they are existed
                        if all(key in ds.keys() for key in ['u10', 'v10', 'sp']):
                            u10 = ds['u10'].where(mask).mean(dim=['y', 'x'])
                            v10 = ds['v10'].where(mask).mean(dim=['y', 'x'])
                            sp = ds['sp'].where(mask).mean(dim=['y', 'x'])

                            # keep attrs
                            u10.attrs = ds['u10'].attrs
                            v10.attrs = ds['v10'].attrs
                            sp.attrs = ds['sp'].attrs
                            array_list = [gas_mask, gas_cm_mask, u10, v10, sp]
                        else:
                            array_list = [gas_mask, gas_cm_mask]

                        # save useful number for attrs
                        sza = ds[gas].attrs['sza']
                        vza = ds[gas].attrs['vza']
                        start_time = ds[gas].attrs['start_time']

                    # export masked data (plume)
                    if 'plume' in os.path.basename(filename):
                        if pick_plume_name == 'plume0':
                            plume_nc_filename = filename.replace('.html', '.nc')
                        else:
                            # rename the filenames if there are more than one plume in the file
                            plume_nc_filename = filename.replace('plume0', pick_plume_name).replace('.html', '.nc')
                    else:
                        plume_nc_filename = filename.replace('.html', f'_{pick_plume_name}.nc').replace('L2', 'L3')

                    # merge data
                    ds_merge = xr.merge(array_list)

                    # add crs info
                    if ds.rio.crs:
                        ds_merge.rio.write_crs(ds.rio.crs, inplace=True)

                    # clear attrs
                    ds_merge.attrs = ''

                    # set global attributes
                    header_attrs = {'version': hypergas.__name__+'_'+hypergas.__version__,
                                    'filename': ds.attrs['filename'],
                                    'start_time': start_time,
                                    'sza': sza,
                                    'vza': vza,
                                    'plume_longitude': longitude,
                                    'plume_latitude': latitude,
                                    }
                    ds_merge.attrs = header_attrs

                    ds_merge.to_netcdf(plume_nc_filename)

                # save mask setting
                mask_setting = {'gas': gas.upper(),
                                'wind_source': wind_source,
                                'land_only': land_only,
                                }

                # read settings to use directly
                for key in params.keys():
                    if key in mask_setting.keys():
                        params.update({key: mask_setting[key]})

                # convert to DataFrame
                df = pd.DataFrame(data=params, index=[0])

                # add source loc
                df['plume_latitude'] = latitude
                df['plume_longitude'] = longitude

                # export data as csv file
                mask_filename = plume_nc_filename.replace('.nc', '.csv')
                df.to_csv(mask_filename, index=False)
                st.success(f'HTML file is exported to: \n \n {plume_html_filename} \
                            \n \n Mask setting is exported to: \n \n {mask_filename} \
                            \n \n You can type "R" to refresh this page for checking/modifying the plume mask, if you are loading a plume html. \
                            \n \n Otherwise, please select the L3 HTML file manually from the right side, and then go to the next step.', icon="✅")
    else:
        # update variables by passing existed csv file content
        for name, value in params.items():
            globals()[name] = value


with col3:
    with st.form("emiss_form"):
        # --- Create emission rate --- #
        st.info('Estimating the gas emission rate using IME method', icon="3️⃣")

        # sitename for csv export
        name = st.text_input('Sitename (any name you like)', value=params['name'])

        # remove space at the end
        if len(name) > 0:
            name = name.strip()

        # ipcc sector name
        sectors = ('Electricity Generation (1A1)', 'Coal Mining (1B1a)',
                   'Oil & Gas (1B2)', 'Livestock (4B)', 'Solid Waste (6A)', 'Other')
        ipcc_sector = st.selectbox('IPCC sector', sectors, index=sectors.index(params['ipcc_sector']))

        # whether only move mask around land pixels
        land_only = st.checkbox('Whether only considering land pixels',
                                value=params['land_only'],
                                )

        land_mask_source_list = ['GSHHS', 'OSM', 'Natural Earth']
        land_mask_source = st.selectbox("Pick data source for creating land mask:",
                                        land_mask_source_list,
                                        index=land_mask_source_list.index(params['land_mask_source']),
                                        )

        # manual wind speed
        wind_speed = st.number_input(
            'Manual wspd [m/s] (please leave this as "None", if you use the reanalysis wind data)', value=None, format='%f')

        # manual surface pressure
        surface_pressure = st.number_input(
            'Manual surface pressure [Pa] (please leave this as "None", if you use the reanalysis wind data)', value=None, format='%f')

        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.spinner('Calculating emission rate ...'):
                # set output name
                plume_nc_filename = filename.replace('.html', '.nc')
                pick_plume_name = filename.split('_')[-1][:-5]
                filename_l3 = plume_nc_filename.replace('.nc', '.csv')
                df = pd.read_csv(filename_l3)
                longitude_source = df['plume_longitude']
                latitude_source = df['plume_latitude']

                # calculate emissions using the IME method with Ueff
                gas = params['gas'].lower()
                filename_l2b = ('_'.join(filename.split('_')[:-1])+'.nc').replace('L3', 'L2')
                ds_l2b = xr.open_dataset(filename_l2b, decode_coords='all')

                # although it would be faster to run only estimation, using Emiss class is safe to ensure steps are as same as l3_process.py
                # create Emiss class
                emiss = Emiss(ds=ds_l2b, gas=gas, plume_name=pick_plume_name)

                # select connected mask data
                emiss.mask_data(longitude_source, latitude_source,
                                wind_source=wind_source,
                                land_only=land_only,
                                land_mask_source=land_mask_source,
                                only_plume=True,
                                azimuth_diff_max=azimuth_diff_max,
                                dist_max=dist_max
                                )

                # calculate emission rate and export csv file
                emiss.estimate(ipcc_sector, wspd_manual=wind_speed,
                               sp_manual=surface_pressure, land_only=land_only, name=name)
                ds_l2b.close()

                # read the new csv file and print key results
                st.success(f'Results are exported to \n \n {filename_l3}')
                df = pd.read_csv(filename_l3)

                Q = df['emission'].item()
                Q_err = df['emission_uncertainty'].item()
                u_eff = df['ueff_ime'].item()
                l_eff = df['leff_ime'].item()
                l_ime = df['l_ime'].item()
                IME = df['ime'].item()
                err_random = df['emission_uncertainty_random'].item()
                err_wind = df['emission_uncertainty_wind'].item()
                err_calib = df['emission_uncertainty_calibration'].item()

                Q_cm = df['emission_cm'].item()
                IME_cm = df['ime_cm'].item()
                l_cm = df['l_cm'].item()

                Q_fetch = df['emission_fetch'].item()
                Q_fetch_err = df['emission_fetch_uncertainty'].item()
                err_wind_fetch = df['emission_fetch_uncertainty_wind'].item()
                err_ime_fetch = df['emission_fetch_uncertainty_ime'].item()

                Q_csf = df['emission_csf'].item()
                Q_csf_err = df['emission_csf_uncertainty'].item()
                u_eff_csf = df['ueff_csf'].item()
                l_csf = df['l_csf'].item()
                err_random_csf = df['emission_csf_uncertainty_random'].item()
                err_wind_csf = df['emission_csf_uncertainty_wind'].item()
                err_calib_csf = df['emission_csf_uncertainty_calibration'].item()

                # print the emission data
                st.warning(f'''**IME (Ueff):**
                               The {gas.upper()} emission rate is {Q:.2f} kg/h $\pm$ {Q_err/Q*100:.2f}% ({Q_err:.2f} kg/h).
                               [
                               U$_{{eff}}$: {u_eff:.2f} m/s,
                               L$_{{eff}}$: {l_eff:.2f} m,
                               L: {l_ime:.2f} m,
                               IME: {IME:.2f} kg,
                               err_random: {err_random:.2f} kg/h,
                               err_wind: {err_wind:.2f} kg/h,
                               err_calibration: {err_calib:.2f} kg/h,
                               ]
                           ''', icon="🔥")
                st.warning(f'''**IME (Carbon Mapper v2):**
                               The {gas.upper()} emission rate is {Q_cm:.2f} kg/h.
                               [
                               IME: {IME_cm:.2f} kg,
                               L: {l_cm:.2f} m,
                               ]
                           ''', icon="🔥")
                st.warning(f'''**IME-fetch (U10):**
                               The {gas.upper()} emission rate is {Q_fetch:.2f} kg/h $\pm$ {Q_fetch_err/Q_fetch*100:.2f}% ({Q_fetch_err:.2f} kg/h).
                               [
                               err_wind: {err_wind_fetch:.2f} kg/h,
                               err_ime: {err_ime_fetch:.2f} kg/h,
                               ]
                           ''', icon="🔥")
                st.warning(f'''**CSF (Ueff):**
                               The {gas.upper()} emission rate is {Q_csf:.2f} kg/h $\pm$ {Q_csf_err/Q_csf*100:.2f}% ({Q_csf_err:.2f} kg/h).
                               [
                               U$_{{eff}}$: {u_eff_csf:.2f} m/s,
                               L: {l_csf:.2f} m,
                               err_random: {err_random_csf:.2f} kg/h,
                               err_wind: {err_wind_csf:.2f} kg/h,
                               err_calibration: {err_calib_csf:.2f} kg/h,
                               ]
                           ''', icon="🔥")
