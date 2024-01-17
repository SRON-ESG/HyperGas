==========
Quickstart
==========

Loading and accessing data
==========================

To work with HSI data you must create a :class:`~hypergas.hyper.Hyper` object and
it uses Satpy to get access to the data.
You need to input what files to read and what Satpy Reader should read them:

    >>> from hypergas import Hyper
    >>> from glob import glob
    >>> filenames = glob('EMIT_L1B_*_001_20230814T100819*nc')
    >>> hyp = Hyper(filenames, reader='emit_l1b') # EnMAP: "hsi_l1b"; EMIT: "emit_l1b"; PRISMA: "hyc_l1"

To load data from the files use the :meth:`Hyper.load <hypergas.hyper.Hyper.load>`
method. Printing the Scene object will list each of the
:class:`xarray.DataArray` objects currently loaded:

    >>> hyp.load()
    >>> print(hyp.scene)
    <xarray.DataArray 'radiance' (bands: 253, y: 1280, x: 1242)>
    dask.array<where, shape=(253, 1280, 1242), dtype=float32, chunksize=(253, 1280, 1242), chunktype=numpy.ndarray>
    Coordinates:
        fwhm     (bands) float32 dask.array<chunksize=(253,), meta=np.ndarray>
    * bands    (bands) float32 381.0 388.4 395.8 ... 2.478e+03 2.486e+03 2.493e+03
    Dimensions without coordinates: y, x
    Attributes: (12/20)
        start_time:           2023-08-14 10:08:19
        end_time:             2023-08-14 10:08:31
        long_name:            Radiance Data
        units:                uW cm-2 sr-1 nm-1
        name:                 radiance
        file_type:            ['emit_l1b_rad']
        ...                   ...
        area:                 Shape: (1280, 1242)\nLons: [[52.19963936194076 52.1...
        reader:               emit_l1b
        _satpy_id:            DataID(name='radiance', modifiers=())
        ancillary_variables:  []
        sza:                  36.839778900146484
        vza:                  8.523066520690918
    <xarray.DataArray 'rgb' (bands: 3, y: 1280, x: 1242)>
    Coordinates:
    * bands    (bands) int64 650 560 470
    Dimensions without coordinates: y, x
    Attributes: (12/20)
        start_time:           2023-08-14 10:08:19
        end_time:             2023-08-14 10:08:31
        long_name:            true color (RGB)
        units:                1
        name:                 rgb
        file_type:            ['emit_l1b_rad']
        ...                   ...
        area:                 Shape: (1280, 1242)\nLons: [[52.19963936194076 52.1...
        reader:               emit_l1b
        _satpy_id:            DataID(name='rgb', modifiers=())
        ancillary_variables:  []
        sza:                  36.839778900146484
        vza:                  8.523066520690918

By default, HyperGas loads data while disregarding water vapor bands.
If you want keep them, you can set ``drop_waterbands`` to ``False``:

    >>> hyp.load(drop_waterbands=False)

Calculating measurement values and navigation coordinates
=========================================================

Like Satpy, measurement values can be calculated from a DataArray within a scene, using .values to get a fully calculated numpy array:

    >>> radiance = hyp.scene['radiance']
    >>> radiance_meas = radiance.values

The ``area`` attribute of the DataArray, if present, can be converted to latitude and longitude arrays.

    >>> radiance_lon, radiance_lat = radiance.attrs['area'].get_lonlats()

Visualizing data
================

To visualize loaded data in jupyter notebook:

.. code-block:: python

    >>> import os
    >>> import folium
    >>> from hypergas.folium_map import Map

    >>> # convert to Dataset
    >>> ds = hyp.scene.to_xarray(datasets='rgb')
    >>> ds.attrs['filename'] = os.path.abspath(filename[0])

    >>> # initialize folium map
    >>> m = Map(ds, ['rgb'])
    >>> m.initialize()
    >>> m.plot(show_layers=[True], opacities=[0.7])

    >>> # show the map in notebook
    >>> layer_control = folium.LayerControl(collapsed=False, position='topleft', draggable=True)
    >>> m.map.add_child(layer_control)

    >>> # export to html file
    >>> m.export()


Retrieval
=========

The :meth:`Hyper.retrieve <hypergas.hyper.Hyper.retrieve>` function calculates the trace gas enhancement using matched filter.
You can input species like "ch4" or "co2" to run the retrieval directly.

    >>> hyp.retrieve(species='ch4')
    >>> ch4 = hyp.scene['ch4']
    >>> print(ch4)
    <xarray.DataArray 'ch4' (bands: 1, y: 1280, x: 1242)>
    Dimensions without coordinates: bands, y, x
    Attributes: (12/21)
        start_time:           2023-08-14 10:08:19
        end_time:             2023-08-14 10:08:31
        long_name:            methane_enhancement
        units:                ppb
        name:                 ch4
        file_type:            ['emit_l1b_rad']
        ...                   ...
        reader:               emit_l1b
        _satpy_id:            DataID(name='ch4', modifiers=())
        ancillary_variables:  []
        sza:                  36.839778900146484
        vza:                  8.523066520690918
        description:          methane enhancement derived by the 2100~2450 nm window

If you prefer your own wavelength range, you can modify the ``wvl_intervals`` parameter,
which can be a list or multi-list.

    >>> hyp.retrieve(wvl_intervals=[1300, 2500], species='ch4')
    >>> hyp.retrieve(wvl_intervals=[[1600, 1750], [2110, 2450]], species='ch4')

Saving to disk
==============

To save loaded datasets to disk as NetCDF file:

.. code-block:: python

    >>> vnames = ['u10', 'v10', 'sp', 'rgb', 'radiance_2100', 'ch4']
    >>> loaded_names = [x['name'] for x in hyp.scene.keys()]

    >>> # drop not loaded vnames
    >>> vnames = [vname for vname in vnames if vname in loaded_names]

    >>> # set global attrs
    >>> header_attrs = {'author': 'AUTHOR'}
    >>> hyp.scene.save_datasets(datasets=vnames, filename='test.nc',
                                header_attrs=header_attrs, writer='cf'
                                )