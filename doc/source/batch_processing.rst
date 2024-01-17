================
Batch Processing
================

HyperGas provides multiple Python scripts to process L1 data into L2 and L3 products.
You can find them in the ``<HyperGas_dir>/scripts/`` directory.

.. list-table::
   :header-rows: 1

   * - Product Level
     - Description
     - Variables
     - Format
   * - L1
     - Radiance data
     - longitude, latitude, radiance
     - --
   * - L2
     - Orthorectified retrieval data
     - | longitude, latitude,
       | trace gas enhancement,
       | denoised trace gas enhancement,
       | RGB,
       | surface pressure, 10-m U, 10-m V
     - | NetCDF,
       | PNG,
       | HTML
   * - L3
     - Masked plume data
     - | longitude, latitude,
       | trace gas enhancement,
       | surface pressure, 10-m U, 10-m V,
       | emission rates info
     - | NetCDF,
       | CSV

Users usually run them in this order:
l2b_process.py --> l2b_plot.py --> streamlit app --> l3_reprocess.py (optional) --> l3_plot.py

Here are the key parameters in each script:

L1 --> L2
=========

l2b_process.py
--------------

This script processes L1 data and export retrieval products (NetCDF) to the same directory.

- skip_exist (bool): whether skip already processed data
- data_dir (str): the root dir of L1 data. The script will process all data recursively and save NetCDF files in each directory.

l2b_plot.py
-----------

This script reads L2 data and plots images as PNG files, which are combined into chunked HTML files by folium.
Users can check multiple variables on different days by clicking layers in the HTML panel.

- skip_exist (bool): whether skip already plotted data
- root_dir (str): the root_dir of L2 data
- plot_markers (bool): whether plot pre-saved markers on map
- chunk (int): the chunk of files for each html file. Please set it lower if you meet RAM error

L2 --> L3
=========