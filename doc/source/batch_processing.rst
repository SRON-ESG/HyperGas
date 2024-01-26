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

.. note::

  The complete flow of generating plume data is:

    l2_process.py --> l2b_plot.py --> plume app --> l2_reprocess.py (optional) --> l3_reprocess.py (optional) --> l3_plot.py

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

The streamlit app has been designed to interactively generate L3 products: plume masks and emission rates.
For details, please check :doc:`plume_app`.

Reprocessing
============

HyperGas supports reprocessing L2 and L3 data.

Reprocessing L2 data
--------------------

Users can run ``l2_process.py`` to rerun the retrieval and choose whether to overwrite the NetCDF files.
Note that including plumes in the matched filter statistics breaks the sparsity assumption
and leads to lower enhancements, especially for large plumes.
So, it is necessary to run ``l2_reprocess.py`` to reprocess L2 data with L3 plume masks.

Reprocessing L3 data
--------------------

The ``l3_reprocess.py`` script reads L2 data and L3 nc/csv files together to generate the new L3 product.
