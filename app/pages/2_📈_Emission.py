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
import random
import sys
from glob import glob

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
from geopy.geocoders import Nominatim
from hypergas.plume_utils import (a_priori_mask_data, calc_emiss,
                                  calc_emiss_fetch)

sys.path.append('..')

st.set_page_config(
    page_title="Emission",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

col1, col2 = st.columns([6, 3])

# set global attrs for exported NetCDF file
AUTHOR = 'Xin Zhang'
EMAIL = 'xin.zhang@sron.nl; xinzhang1215@gmail.com'
INSTITUTION = 'SRON Netherlands Institute for Space Research'

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
        html_filepath_list = list(itertools.chain(*html_filepath_list))

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
          'wind_source': None, 'land_only': True, 'wind_speed': None,
          'azimuth_diff_max': 30., 'alpha1': 0.0, 'alpha2': 0.80, 'alpha3': 0.40,
          'name': '', 'ipcc_sector': 'Solid Waste (6A)',
          'platform': None, 'source_tropomi': True, 'source_trace': False
          }

# copy the params, this should not be modified
defaults = params.copy()

with col3:
    if filename is not None:
        # --- print existed mask info --- #
        if 'plume' in os.path.basename(filename):
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
                        (Please keep the default value, unless there are obvious false plume masks around)',
                                                   value=params['azimuth_diff_max'], format='%f')

                only_plume = st.checkbox('Whether only plot plume',
                                         value=True,
                                         )

                land_only = st.checkbox('Whether only considering land pixels',
                                        value=params['land_only'],
                                        )

                land_mask_source = st.selectbox("Pick data source for creating land mask:",
                                                ('GSHHS', 'Natural Earth'),
                                                index=0,
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
                        mask, lon_mask, lat_mask, plume_html_filename = a_priori_mask_data(filename, ds, gas, longitude, latitude,
                                                                                           pick_plume_name, wind_source,
                                                                                           land_only, land_mask_source, only_plume,
                                                                                           azimuth_diff_max)

                        # mask data
                        gas_mask = ds[gas].where(mask)

                        # calculate mean wind and surface pressure in the plume if they are existed
                        if all(key in ds.keys() for key in ['u10', 'v10', 'sp']):
                            u10 = ds['u10'].where(mask).mean(dim=['y', 'x'])
                            v10 = ds['v10'].where(mask).mean(dim=['y', 'x'])
                            sp = ds['sp'].where(mask).mean(dim=['y', 'x'])

                            # keep attrs
                            u10.attrs = ds['u10'].attrs
                            v10.attrs = ds['v10'].attrs
                            sp.attrs = ds['sp'].attrs
                            array_list = [gas_mask, u10, v10, sp]
                        else:
                            array_list = [gas_mask]

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
                    header_attrs = {'author': AUTHOR,
                                    'email': EMAIL,
                                    'institution': INSTITUTION,
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

        # ipcc sector name
        sectors = ('Electricity Generation (1A1)', 'Coal Mining (1B1a)',
                   'Oil & Gas (1B2)', 'Livestock (4B)', 'Solid Waste (6A)', 'Other')
        ipcc_sector = st.selectbox('IPCC sector', sectors, index=sectors.index(params['ipcc_sector']))

        # platform for csv output
        platform_names = ('EnMAP', 'EMIT', 'PRISMA')

        # check the platform by filename automatically
        platform_default = 'EnMAP'
        if filename is not None:
            if 'ENMAP' in filename:
                platform_default = 'EnMAP'
            elif 'EMIT' in filename:
                platform_default = 'EMIT'
            elif 'PRS' in filename:
                platform_default = 'PRISMA'

        platform = st.selectbox('Platform', platform_names, index=platform_names.index(platform_default))

        # set pixel resolution
        if platform == 'EMIT':
            instrument = 'emi'
            provider = 'NASA-JPL'
            pixel_res = 60  # meter
            alpha_area = {'alpha1': 0., 'alpha2': 0.68, 'alpha3': 0.48}
            alpha_point = {'alpha1': 0., 'alpha2': 0.30, 'alpha3': 0.51}
            if ipcc_sector == 'Solid Waste (6A)':
                params.update(alpha_area)
                alpha_replace = alpha_point
            else:
                params.update(alpha_point)
                alpha_replace = alpha_area

        elif platform in ['EnMAP', 'PRISMA']:
            instrument = 'hsi'
            provider = 'DLR'
            pixel_res = 30  # meter
            alpha_area = {'alpha1': 0., 'alpha2': 0.80, 'alpha3': 0.40}
            alpha_point = {'alpha1': 0., 'alpha2': 0.46, 'alpha3': 0.41}
            if ipcc_sector == 'Solid Waste (6A)':
                params.update(alpha_area)
                alpha_replace = alpha_point
            else:
                params.update(alpha_point)
                alpha_replace = alpha_area

        # input alphas for calculating U_eff
        st.success('U_eff = alpha1 * np.log(wind_speed_average) + alpha2 + alpha3 * wind_speed_average', icon='🧮')
        alpha1 = st.number_input('alpha1 for Ueff', value=params['alpha1'], format='%f')
        alpha2 = st.number_input('alpha2 for Ueff', value=params['alpha2'], format='%f')
        alpha3 = st.number_input('alpha3 for Ueff', value=params['alpha3'], format='%f')

        # source type
        source_tropomi = st.checkbox('Whether TROPOMI captures the source',
                                     value=params['source_tropomi'],
                                     )

        source_trace = st.checkbox('Whether ClimateTrace dataset contains the source',
                                   value=params['source_trace'],
                                   )

        # whether only move mask around land pixels
        land_only = st.checkbox('Whether only considering land pixels',
                                value=params['land_only'],
                                )

        # manual wind speed
        wind_speed = st.number_input(
            'Manual wspd [m/s] (please leave this as "None", if you use the reanalysis wind data)', value=None, format='%f')

        submitted = st.form_submit_button("Submit")

        if submitted:
            with st.spinner('Calculating emission rate ...'):
                # set output name
                plume_nc_filename = filename.replace('.html', '.nc')
                pick_plume_name = filename.split('_')[-1][:-5]

                # calculate emissions using the IME method with Ueff
                gas = params['gas'].lower()
                wind_speed, wdir, wind_speed_all, wdir_all, wind_source_all, l_eff, u_eff, IME, Q, Q_err, \
                    err_random, err_wind, err_calib = calc_emiss(gas, plume_nc_filename, pick_plume_name,
                                                                 alpha_replace,
                                                                 pixel_res=pixel_res,
                                                                 alpha1=alpha1,
                                                                 alpha2=alpha2,
                                                                 alpha3=alpha3,
                                                                 wind_source=wind_source,
                                                                 wspd=wind_speed,
                                                                 land_only=land_only
                                                                 )

                # calculate emissions using the IME-fetch method with U10
                Q_fetch, Q_fetch_err, err_ime_fetch, err_wind_fetch \
                    = calc_emiss_fetch(gas, plume_nc_filename,
                                       pixel_res=pixel_res,
                                       wind_source=wind_source,
                                       wspd=wind_speed
                                       )

                # print the emission data
                st.warning(f'''**IME (Ueff):**
                               The {gas.upper()} emission rate is {Q:.2f} kg/h $\pm$ {Q_err/Q*100:.2f}% ({Q_err:.2f} kg/h).
                               [
                               U$_{{eff}}$: {u_eff:.2f} m/s,
                               L$_{{eff}}$: {l_eff:.2f} m,
                               IME: {IME:.2f} kg,
                               err_random: {err_random:.2f} kg/h,
                               err_wind: {err_wind:.2f} kg/h,
                               err_calibration: {err_calib:.2f} kg/h,
                               ]
                           ''', icon="🔥")
                st.warning(f'''**IME-fetch (U10):**
                               The {gas.upper()} emission rate is {Q_fetch:.2f} kg/h $\pm$ {Q_fetch_err/Q_fetch*100:.2f}% ({Q_fetch_err:.2f} kg/h).
                               [
                               err_wind: {err_wind_fetch:.2f} kg/h,
                               err_ime: {err_ime_fetch:.2f} kg/h,
                               ]
                           ''', icon="🔥")

                # calculate plume bounds
                with xr.open_dataset(plume_nc_filename) as ds:
                    plume_mask = ~ds[gas].isnull()
                    lon_mask = ds['longitude'].where(plume_mask, drop=True)
                    lat_mask = ds['latitude'].where(plume_mask, drop=True)
                    t_overpass = pd.to_datetime(ds[gas].attrs['start_time'])

                bounds = [lon_mask.min().item(), lat_mask.min().item(),
                          lon_mask.max().item(), lat_mask.max().item()]

                # get the location attrs
                try:
                    geolocator = Nominatim(user_agent='hyper'+str(random.randint(1, 100)))
                    location = geolocator.reverse(
                        f'{plume_dict[pick_plume_name][0]}, {plume_dict[pick_plume_name][1]}', exactly_one=True, language='en')
                    address = location.raw['address']
                except Exception as e:
                    st.warning('Can not access openstreetmap. Leave location info to empty.')
                    address = {}

                # save ime results
                ime_results = {'plume_id': f"{instrument}-{t_overpass.strftime('%Y%m%dt%H%M%S')}-{pick_plume_name}",
                               'plume_latitude': plume_dict[pick_plume_name][0],
                               'plume_longitude': plume_dict[pick_plume_name][1],
                               'datetime': t_overpass.strftime('%Y-%m-%dT%H:%M:%S%z'),
                               'country': address.get('country', ''),
                               'state': address.get('state', ''),
                               'city': address.get('city', ''),
                               'name': name,
                               'ipcc_sector': ipcc_sector,
                               'gas': gas.upper(),
                               'cmf_type': 'mf',
                               'plume_bounds': [bounds],
                               'instrument': instrument,
                               'platform': platform,
                               # 'provider': provider,
                               'emission': Q,
                               'emission_uncertainty': Q_err,
                               'emission_uncertainty_random': err_random,
                               'emission_uncertainty_wind': err_wind,
                               'emission_uncertainty_calibration': err_calib,
                               'emission_fetch': Q_fetch,
                               'emission_fetch_uncertainty': Q_fetch_err,
                               'emission_fetch_uncertainty_ime': err_ime_fetch,
                               'emission_fetch_uncertainty_wind': err_wind_fetch,
                               'wind_speed': wind_speed,
                               'wind_direction': wdir,
                               'wind_source': wind_source,
                               'ime': IME,
                               'ueff_ime': u_eff,
                               'leff_ime': l_eff,
                               'azimuth_diff_max': azimuth_diff_max,
                               'alpha1': alpha1,
                               'alpha2': alpha2,
                               'alpha3': alpha3,
                               'source_tropomi': source_tropomi,
                               'source_trace': source_trace,
                               'wind_speed_all': [wind_speed_all],
                               'wind_direction_all': [wdir_all],
                               'wind_source_all': [wind_source_all],
                               }

                # convert to DataFrame and export data as csv file
                df = pd.DataFrame(data=ime_results, index=[0])
                savename = plume_nc_filename.replace('.nc', '.csv')
                df.to_csv(savename, index=False)
                st.success(f'Results are exported to \n \n {savename}')
