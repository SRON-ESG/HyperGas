Configuration
=============

HyperGas automatically reads settings from a YAML file called ``config.yaml``.
This file is located at ``<HyperGas_dir>/hypergas/config.yaml``.
All path names in the config file are relative paths within ``<HyperGas_dir>/hypergas/``.
For instance, ``resources/absorption`` corresponds to ``<HyperGas_dir>/hypergas/resources/absorption/``.

SRON users
----------

If you are a user at SRON, all input data is already shared within the L4 group.
You can link the data to the ``hypergas`` directory using the following command:

.. code-block:: bash

   cd <HyperGas_dir>/hypergas
   ln -s /deos/xinz/hypergas_data resources

Non-SRON users
--------------

If you are not a user at SRON, please download the input data manually from the `shared repository <https://doi.org/10.5281/zenodo.17369768>`_.
After downloading the data, unzip it into the ``<HyperGas_dir>/hypergas`` directory.
All extracted files will be stored in ``<HyperGas_dir>/hypergas/resources`` directory.
 
absorption_dir
^^^^^^^^^^^^^^

The directory where absorption line data is stored.
Default path: ``resources/absorption``.
The directory structure should be as follows:

.. code-block::

    ├── absorption
    │   ├── absorption_cs_ALL_midlatitudesummer.nc
    │   ├── absorption_cs_ALL_midlatitudewinter.nc
    │   ├── absorption_cs_ALL_standard.nc
    │   ├── absorption_cs_ALL_subarcticsummer.nc
    │   ├── absorption_cs_ALL_subarcticwinter.nc
    │   ├── absorption_cs_ALL_tropical.nc
    │   ├── atmosphere_midlatitudesummer.dat
    │   ├── atmosphere_midlatitudewinter.dat
    │   ├── atmosphere_standard.dat
    │   ├── atmosphere_subarcticsummer.dat
    │   ├── atmosphere_subarcticwinter.dat
    │   ├── atmosphere_tropical.dat


irradiance_dir
^^^^^^^^^^^^^^

The directory where solar irradiance data is stored.
Default path: ``resources/solar_irradiance``.

.. code-block::

    └── solar_irradiance
        └── solar_irradiance_0400-2600nm_highres_sparse.nc

modtran_dir
^^^^^^^^^^^

The directory where MODTRAN LUT data is stored.
Default path: ``resources/modtran_full``.

.. code-block::

    ├── modtran_full
    │   ├── dataset_ch4_full.hdf5
    │   └── dataset_co2_full.hdf5

rgb_dir
^^^^^^^

The directory where illuminants data is stored.
Default path: ``resources/rgb``.

.. code-block::

    ├── rgb
    │   └── D_illuminants.mat

osm_dir
^^^^^^^

The directory where OSM+WorldCover water mask rasters are stored.
Default path: ``resources/OSM_WorldCover``.

era5_dir
^^^^^^^^

The directory where ERA5 surface GRIB data is stored.
Default path: ``resources/ERA5``.
The directory structure should be ``<yyyy>/sl***.grib``:

.. code-block::

    ├── 2022
    │   ├── sl_20220101.grib
    │   ├── sl_20220102.grib
    │   ├── ...........
    ├── 2023
    │   ├── sl_20230101.grib
    │   ├── sl_20230102.grib
    │   └── ...........
    └── 2024
        ├── sl_20240101.grib
        ├── sl_20240102.grib
    │   └── ...........

geosfp_dir
^^^^^^^^^^

The directory where GEOS-FP surface GRIB data is stored.
Default path: ``resources/GEOS-FP``.
The directory structure should be ``<yyyy>/<mm>/<dd>/GEOS.fp.asm.tavg1_2d_slv_Nx.*.V01.nc4``:

.. code-block::

    ├── 2023
    │   ├── 01
    │   │   ├── 01
    │   │   │   ├── GEOS.fp.asm.tavg1_2d_slv_Nx.20230101_0030.V01.nc4
    │   │   │   ├── GEOS.fp.asm.tavg1_2d_slv_Nx.20230101_0130.V01.nc4
    │   │   │   ├── GEOS.fp.asm.tavg1_2d_slv_Nx.20230101_0230.V01.nc4
    │   │   │   ├── GEOS.fp.asm.tavg1_2d_slv_Nx.20230101_0330.V01.nc4
    │   │   │   ├── ..........
    │   │   ├── 02
    │   │   ├── ..
    │   ├── 02
    │   ├── 03
    │   ├── ..
    ├── 2024
    │   └── 01
    │   └── ..

markers_filename
^^^^^^^^^^^^^^^^

The optional csv file saves pre-defined markers.
It should contain at least two columns: *latitude* and *longitude*.
The batch processing script ``l2b_plot.py`` will place CircleMarkers on the map.
Clicking on a marker will display the correcponding DataFrame information.
Default: ``resources/markers/markers.csv``

spacetrack_usename and spacetrack_password
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The username and password of `spacetrack <https://www.space-track.org/auth/login>`_.
If the HSI data don't have SZA/VZA info, HyperGas will automatically calculate them through the spacetrack api.
