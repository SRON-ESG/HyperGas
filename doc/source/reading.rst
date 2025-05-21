=======
Reading
=======

To read the hyperspectral L1 radiance data, users can choose between two options:

- Use `Satpy <https://satpy.readthedocs.io/>`_ to read the raw data in :class:`~xarray.DataArray` and :class:`~xarray.Dataset`.
- Use the :class:`~hypergas.hyper.Hyper` class from HyperGas, which is built on top of `Satpy <https://satpy.readthedocs.io/>`_.
  This approach reads the required data for trace gas retrieval and combines it into a single :class:`~xarray.Dataset` with user-friendly coordinates.

Using Satpy
===========

To read data automatically, Satpy requires a list of ``filenames`` along with the corresponding ``reader`` name.
The available hyperspectral readers are: ``'hsi_l1b'`` (EnMAP), ``'emit_l1b'`` (EMIT), ``'hyc_l1'`` (PRISMA).
Below is an example of how to read EMIT L1 RAD and OBS data:

.. code-block:: python

    >>> from satpy import Scene
    >>> from glob import glob
    >>> filenames = glob(data_dir+'EMIT_L1B_*')
    >>> scn = Scene(filenames=filenames, reader='emit_l1b')

Users can explore the available datasets and select the variables they want to load:

    >>> print(scn.available_dataset_names())
    ['cosine_i',
    'earth_sun_distance',
    'elev',
    'glt_x',
    'glt_y',
    'path_length',
    'radiance',
    'saa',
    'solar_phase',
    'surface_aspect',
    'surface_slope',
    'sza',
    'time',
    'vaa',
    'vza']

    >>> scn.load(['glt_x', 'glt_y', 'radiance', 'sza', 'vza'])
    >>> print(scn)
    <xarray.DataArray 'vza' (y: 1280, x: 1242)> Size: 6MB
    dask.array<getitem, shape=(1280, 1242), dtype=float32, chunksize=(1280, 1242), chunktype=numpy.ndarray>
    Coordinates:
        bands    <U94 376B 'To-sensor zenith (0 to 90 degrees from zenith)'
    Dimensions without coordinates: y, x
    Attributes: (12/18)
        long_name:            To-sensor zenith (0 to 90 degrees from zenith)
        name:                 vza
        file_type:            ['emit_l1b_obs']
        standard_name:        sensor_zenith_angle
        units:                degree
        nc_group:             None
        ...                   ...
        geotransform:         [ 4.91330392e+01  5.42232520e-04 -0.00000000e+00  4...
        spatial_ref:          GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",...
        area:                 Shape: (1280, 1242)\nLons: [[49.133416730544425 49....
        reader:               emit_l1b
        _satpy_id:            DataID(name='vza', modifiers=())
        ancillary_variables:  []
    <xarray.DataArray 'sza' (y: 1280, x: 1242)> Size: 6MB
    dask.array<getitem, shape=(1280, 1242), dtype=float32, chunksize=(1280, 1242), chunktype=numpy.ndarray>
    Coordinates:
        bands    <U94 376B 'To-sun zenith (0 to 90 degrees from zenith)'
    Dimensions without coordinates: y, x
    Attributes: (12/18)
        long_name:            To-sun zenith (0 to 90 degrees from zenith)
        name:                 sza
        file_type:            ['emit_l1b_obs']
        standard_name:        solar_zenith_angle
        units:                degree
        nc_group:             None
        ...                   ...
        geotransform:         [ 4.91330392e+01  5.42232520e-04 -0.00000000e+00  4...
        spatial_ref:          GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",...
        area:                 Shape: (1280, 1242)\nLons: [[49.133416730544425 49....
        reader:               emit_l1b
        _satpy_id:            DataID(name='sza', modifiers=())
        ancillary_variables:  []
    <xarray.DataArray 'glt_y' (ortho_y: 1976, ortho_x: 2336)> Size: 18MB
    dask.array<where, shape=(1976, 2336), dtype=int32, chunksize=(1976, 2336), chunktype=numpy.ndarray>
    Dimensions without coordinates: ortho_y, ortho_x
    Attributes: (12/19)
        _FillValue:           -9999
        long_name:            GLT sample Lookup
        units:                pixel location
        name:                 glt_y
        file_type:            ['emit_l1b_rad', 'emit_l1b_obs']
        standard_name:        glt_y
        ...                   ...
        geotransform:         [ 4.91330392e+01  5.42232520e-04 -0.00000000e+00  4...
        spatial_ref:          GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",...
        area:                 Shape: (1280, 1242)\nLons: <xarray.DataArray 'lon' ...
        reader:               emit_l1b
        _satpy_id:            DataID(name='glt_y', modifiers=())
        ancillary_variables:  []
    <xarray.DataArray 'glt_x' (ortho_y: 1976, ortho_x: 2336)> Size: 18MB
    dask.array<where, shape=(1976, 2336), dtype=int32, chunksize=(1976, 2336), chunktype=numpy.ndarray>
    Dimensions without coordinates: ortho_y, ortho_x
    Attributes: (12/19)
        _FillValue:           -9999
        long_name:            GLT sample Lookup
        units:                pixel location
        name:                 glt_x
        file_type:            ['emit_l1b_rad', 'emit_l1b_obs']
        standard_name:        glt_x
        ...                   ...
        geotransform:         [ 4.91330392e+01  5.42232520e-04 -0.00000000e+00  4...
        spatial_ref:          GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",...
        area:                 Shape: (1280, 1242)\nLons: <xarray.DataArray 'lon' ...
        reader:               emit_l1b
        _satpy_id:            DataID(name='glt_x', modifiers=())
        ancillary_variables:  []
    <xarray.DataArray 'radiance' (bands: 285, y: 1280, x: 1242)> Size: 2GB
    dask.array<transpose, shape=(285, 1280, 1242), dtype=float32, chunksize=(285, 1280, 1242), chunktype=numpy.ndarray>
    Coordinates:
        fwhm     (bands) float32 1kB dask.array<chunksize=(285,), meta=np.ndarray>
    * bands    (bands) float32 1kB 381.0 388.4 395.8 ... 2.486e+03 2.493e+03
    Dimensions without coordinates: y, x
    Attributes: (12/18)
        long_name:            Radiance Data
        units:                uW cm-2 sr-1 nm-1
        name:                 radiance
        file_type:            ['emit_l1b_rad']
        standard_name:        radiance
        nc_group:             None
        ...                   ...
        geotransform:         [ 4.91330392e+01  5.42232520e-04 -0.00000000e+00  4...
        spatial_ref:          GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",...
        area:                 Shape: (1280, 1242)\nLons: [[49.133416730544425 49....
        reader:               emit_l1b
        _satpy_id:            DataID(name='radiance', modifiers=())
        ancillary_variables:  []


To quickly view the loaded variables, use Satpy's ``show`` function or the ``plot`` function of :class:`~xarray.DataArray`:

    >>> scn.show('sza')
    >>> scn['sza'].plot()

For more details about Satpy's Reading function, please refer to their `website <https://satpy.readthedocs.io/en/stable/reading.html>`_.

Using HyperGas
==============

The :class:`~hypergas.hyper.Hyper` class includes Satpy reading functions and provides useful data pre-processing.

    >>> from hypergas import Hyper
    >>> hyp = Hyper(filenames, reader='emit_l1b')
    >>> hyp.load(drop_waterbands=True)

The Satpy :class:`~satpy.scene.Scene` class can be accessed easily:

    >>> print(hyp.scene.keys())
    [DataID(name='glt_x', modifiers=()),
    DataID(name='glt_y', modifiers=()),
    DataID(name='radiance', modifiers=()),
    DataID(name='radiance_2100', modifiers=()),
    DataID(name='rgb', modifiers=()),
    DataID(name='sp'),
    DataID(name='sza', modifiers=()),
    DataID(name='u10'),
    DataID(name='v10'),
    DataID(name='vza', modifiers=())]

    >>> print(hyp.scene['sp'])
    <xarray.DataArray 'sp' (y: 1280, x: 1242)> Size: 13MB
    array([[100246.05889694, 100244.88959609, 100243.73284768, ...,
            100027.19022245, 100032.67183027, 100036.84124356],
        [100248.05846289, 100246.90273037, 100245.74194296, ...,
            100031.91140584, 100037.37840683, 100041.53576746],
        [100250.06514698, 100248.91619575, 100247.75542524, ...,
            100036.55923353, 100042.06735457, 100046.23049991],
        ...,
        [101221.46603699, 101221.47502242, 101221.48402823, ...,
            101200.94422086, 101200.92076079, 101200.90309345],
        [101221.50476201, 101221.5137563 , 101221.52277096, ...,
            101200.98610785, 101200.96267115, 101200.94502143],
        [101221.54342366, 101221.5524268 , 101221.56145031, ...,
            101201.02787439, 101201.00446104, 101200.98682891]])
    Dimensions without coordinates: y, x
    Attributes:
        area:          Shape: (1280, 1242)\nLons: [[49.133416730544425 49.1337882...
        sensor:        EMIT
        geotransform:  [ 4.91330392e+01  5.42232520e-04 -0.00000000e+00  4.060421...
        spatial_ref:   GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137...
        filename:      /home/xinz/Documents/githab/HyperGas-GMD/data/ch4_cases/se...
        long_name:     surface pressure
        units:         Pa
        name:          sp
        _satpy_id:     DataID(name='sp')

Several new variables are available compared to the ones returned by Satpy:
``radiance_2100``, ``rgb``, ``sp``, ``u10``, and ``v10``.

    - ``radiance_2100``: Radiance closest to the 2100 nm wavelength, useful for analyzing albedo effects.
    - ``rgb``:  An RGB image generated by HyperGas using the `HSI2RGB <https://github.com/JakobSig/HSI2RGB>`_ method applied to HSI L1 data.
      If this method fails, bands near 650, 560, and 470 nm are combined with Gamma normalization.
    - ``sp``: 2D surface pressure
    - ``u10``: 2D 10 m U wind speed
    - ``v10``: 2D 10 m V wind speed


The attributes of ``radiance`` have also been updated:

    - The number of bands decreases from 285 to 253 due to the ``drop_waterbands=True`` setting.
    - The mean values of Solar Zenith Angle (SZA) and View Zenith Angle (VZA) are now saved as attributes.

    >>> print(hyp.scene['radiance'])
    <xarray.DataArray 'radiance' (bands: 253, y: 1280, x: 1242)> Size: 2GB
    dask.array<where, shape=(253, 1280, 1242), dtype=float32, chunksize=(253, 1280, 1242), chunktype=numpy.ndarray>
    Coordinates:
        fwhm     (bands) float32 1kB dask.array<chunksize=(253,), meta=np.ndarray>
    * bands    (bands) float32 1kB 381.0 388.4 395.8 ... 2.486e+03 2.493e+03
    Dimensions without coordinates: y, x
    Attributes: (12/20)
        long_name:            Radiance Data
        units:                uW cm-2 sr-1 nm-1
        name:                 radiance
        file_type:            ['emit_l1b_rad']
        standard_name:        radiance
        nc_group:             None
        ...                   ...
        area:                 Shape: (1280, 1242)\nLons: [[49.133416730544425 49....
        reader:               emit_l1b
        _satpy_id:            DataID(name='radiance', modifiers=())
        ancillary_variables:  []
        sza:                  30.461334228515625
        vza:                  9.43835163116455
