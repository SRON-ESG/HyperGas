.. HyperGas documentation master file, created by
   sphinx-quickstart on Wed Jan 10 09:43:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HyperGas's Documentation!
=======================================

.. image:: ../fig/logo.png

HyperGas is a python library for reading, processing, and writing data from Hyperspectral Imagers (HSI).
The reading function is built upon HSI readers
(`EMIT <https://github.com/pytroll/satpy/pull/2592>`_,
`EnMAP <https://github.com/pytroll/satpy/pull/2590>`_,
and PRISMA)
from the `Satpy <https://satpy.readthedocs.io/>`_ package.
Satpy converts HSI L1 data to the common Xarray :class:`~xarray.DataArray` and :class:`~xarray.Dataset` classes,
facilitating interoperability with HyperGas.

Key features of HyperGas include:

- Creating RGB (Red/Green/Blue) images by combining multiple spectral bands
- Retrieving trace gas enhancements (e.g., methane, carbon dioxide)
- Denoising retrieval outputs
- Exporting results in various formats, including PNG, HTML, and CF-compliant NetCDF files
- Estimating gas emission rates and saving them in CSV format

Go to the HyperGas project page (coming soon) for source code and downloads.

HyperGas is designed to easily support the retrieval of trace gases for any HSI instruments.
The following table displays the HSI data that HyperGas supports.

.. _project: https://github.com/zxdawn/HyperGas


.. list-table::
   :header-rows: 1

   * - Name
     - Link
     - Satpy reader name
   * - EMIT
     - https://earth.jpl.nasa.gov/emit/
     - emit_l1b
   * - EnMAP
     - https://www.enmap.org/
     - hsi_l1b
   * - PRISMA
     - https://prisma.asi.it/
     - hyc_l1

Documentation
=============

.. toctree::
   :caption: Basic Information
   :maxdepth: 2

   overview
   install
   config
   data_download
   quickstart
 
.. toctree::
   :caption: Level 1 Product
   :maxdepth: 2

   reading

.. toctree::
   :caption: Level 2 Product
   :maxdepth: 2

   retrieval
   orthorectification
   denoising

.. toctree::
   :caption: Level 3 Product
   :maxdepth: 2

   plume_mask

.. toctree::
   :caption: Level 4 Product
   :maxdepth: 2

   emission

.. toctree::
   :caption: Workflow
   :maxdepth: 2

   batch_processing
   plume_app

.. toctree::
   :caption: Developer Uuide

   dev_guide

.. toctree::
    :maxdepth: 1

    HyperGas API <api/modules>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
