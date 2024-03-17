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

Here are the key points of each script:

L1 --> L2
=========

Step 1: l2b_process.py
----------------------

This script processes L1 data and exports retrieval products (NetCDF) to the same directory.
Users can use :class:`~xarray` to read and plot NetCDF files easily.

The script supports modifying following parameters:

- ``skip_exist (bool)``: Whether skipping already processed data.

- ``species (str, list)``: The gas to be retrieved.

    - 'all': retrieve all supported gases
    - single gas name str, e.g. 'ch4'
    - list of gas names, e.g. ['ch4', 'co2']

- ``data_dir (str)``: The root dir of L1 data.

    The script will process all data recursively and save NetCDF files in each directory.
    For example, users save L1 data like this: ``<data>/<facility_1>/`` and ``<data>/<facility_2>/``.
    If we set ``data_dir = <data>``, the script will process all L1 data under ``facility_1`` and ``<facility_2>``.
    The L2 data will be exported to each folder seperately.

Let's assume users set ``species = 'ch4'``, the output L2 file should have following variables:

.. list-table::
    :header-rows: 1

    * - Variable name
      - Description
    * - y
      - y coordinate of projection
    * - x
      - x coordinate of projection
    * - latitude
      - 2D latitude
    * - longitude
      - 2D longitude
    * - ch4
      - | 2D ch4 enhancement field
        | (retrieved by the strong absorption window)
    * - ch4_comb
      - | 2D ch4 enhancement field
        | (retrieved by a wider absorption window to decrease background noise)
    * - ch4_denoise
      - denoised 2D ch4 field
    * - ch4_comb_denoise
      - | denoised 2D ch4_comb field
        | (this is the default variable to create plume masks later)
    * - radiance_2100
      - | 2D radiance data at 2100 nm 
        | (this is useful to check albedo effects)
    * - rgb
      - 2D RGB
    * - segmentation
      - 2D pixel classification (land or ocean)
    * - sp
      - 2D surface pressure
    * - u10
      - 2D 10 m U wind speed
    * - v10
      - 2D 10 m V wind speed

Step 2: l2b_plot.py
-------------------

This script reads L2 data and plots images as PNG files, which are combined into chunked HTML files by folium.
Users can check multiple variables on different days by clicking layers in the HTML panel.

The script supports modifying following parameters:

- ``skip_exist (bool)``: Whether skip already plotted data.

- ``root_dir (str)``: The root_dir of L2 data.

- ``plot_markers (bool)``: Whether plot pre-saved markers on map.

- ``chunk (int)``: The chunk of files for each html file. Please set it lower if you meet RAM error.

Here is an example of HTML file:

.. image:: ../fig/l2_html.jpg

The layers on the left have two main components: *group* and *variables*.
The *group* level controls all sub-variables: rgb, radiance, different gas products.
The position of wind arrows is located at the scene center and can be moved by clicking and dragging:

.. image:: ../fig/l2_html_wind.jpg

.. note::

  The colorbar limits are hard-coded in ``<HyperGas_dir>/hypergas/folium_map.py``.

L2 --> L3
=========

The streamlit app has been designed to interactively generate L3 products: plume masks and emission rates.
For details, please check :doc:`plume_app`.

Reprocessing (optional)
=======================

HyperGas supports reprocessing L2 and L3 data.

Reprocessing L2 data
--------------------

To rerun the retrieval, users can execute ``l2_process.py`` and decide whether to overwrite the NetCDF files.
However, incorporating plumes into the matched filter statistics might break the sparsity assumption, potentially resulting in lower enhancements, particularly for highly-emitting (> 10 t/h) plumes.
For example, the Norte III landfill emission rate was estimated as 13 t/h, it is updated to 20.7 t/h.
Therefore, it is essential to run ``l2_reprocess.py`` to reprocess L2 data with L3 plume masks.

The ``l2_reprocess.py`` script enables adjustment of the following parameters:

- ``root_dir (str)``: the root directory of L2 data.

- ``species (str)``: the gas intended for reprocessing.

- ``rad_dist (str)``: choose between "normal" or "lognormal" distribution. While the "normal" option may underestimate high emission rates, we have set "lognormal" as the default due to increased noise associated with "lognormal".

- ``land_mask_source (str)``: the origin ("GSHHS" or "Natural Earth") for land classification.

Reprocessing L3 data
--------------------

To calculate the emission rate using the updated L2 data, users have the option to rerun the APP or use the ``l3_reprocess.py`` script.
This script reads both L2 data and L3 nc files concurrently, generating the new L3 products based on the settings specified in the L3 csv file.
It offers the following parameters:

- ``root_dir (str)``: the root directory of L3 data.

- ``reprocess_nc (boolean)``: indicates whether to reprocess the L3 NetCDF data. If disabled, the script will compute the emission rate using the existing L3 NetCDF file and the settings outlined in the csv file.

- ``land_mask_source (str)``: the origin ("GSHHS" or "Natural Earth") for land classification.

.. note::

    Please verify the plume mask carefully. Due to variations in the new retrieval outcomes, the plume mask may differ even with identical csv settings.
