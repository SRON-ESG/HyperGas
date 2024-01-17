Configuration
=============

HyperGas automatically reads settings from a YAML file called ``config.yaml``.
This file is located at ``<HyperGas_dir>/hypergas/config.yaml``.
All path names in the config file are relative paths under ``<HyperGas_dir>/hypergas/``.
For instance, ``resources/absorption`` is equivalent to ``<HyperGas_dir>/hypergas/resources/absorption/``.

Parameters
----------


absorption_dir
^^^^^^^^^^^^^^


The directory where absorption line data is stored.
Default path: ``resources/absorption``.
The directory structure should be as follows:

.. code-block::

    ├── absorption
    │   ├── absorption_cs_CH4_ALL_midlatitudesummer.csv
    │   ├── absorption_cs_CH4_ALL_midlatitudewinter.csv
    │   ├── absorption_cs_CH4_ALL_standard.csv
    │   ├── absorption_cs_CH4_ALL_subarcticsummer.csv
    │   ├── absorption_cs_CH4_ALL_subarcticwinter.csv
    │   ├── absorption_cs_CH4_ALL_tropical.csv
    │   ├── absorption_cs_CH4_SWIR_midlatitudesummer.csv
    │   ├── absorption_cs_CO2_ALL_midlatitudesummer.csv
    │   ├── absorption_cs_CO2_ALL_midlatitudewinter.csv
    │   ├── absorption_cs_CO2_ALL_standard.csv
    │   ├── absorption_cs_CO2_ALL_subarcticsummer.csv
    │   ├── absorption_cs_CO2_ALL_subarcticwinter.csv
    │   ├── absorption_cs_CO2_ALL_tropical.csv
    │   ├── absorption_cs_CO2_SWIR_midlatitudesummer.csv
    │   ├── ............


irradiance_dir
^^^^^^^^^^^^^^

The directory where solar irradiance data is stored.
Default path: ``resources/irradiance_dir``.

.. code-block::

    └── solar_irradiance
        └── solar_irradiance_0400-2600nm_highres_sparse.dat

rgb_dir
^^^^^^^

The directory where illuminants data is stored.
Default path: ``resources/rgb``.

.. code-block::

    ├── rgb
    │   └── D_illuminants.mat

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

The csv file which saves pre-defined markers.
It should contains at least two columns: *latitude* and *longitude*.
The batch processing script ``l2b_plot.py`` will add CircleMarkers on Map and click it to see correcponding DataFrame info.
Default: ``resources/markers/markers.csv``

spacetrack_usename and spacetrack_password
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The username and password of `spacetrack <https://www.space-track.org/auth/login>`_.
If the HSI data don't have SZA/VZA info, HyperGas will automatically calculate them through the spacetrack api.