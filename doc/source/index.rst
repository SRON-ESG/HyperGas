.. HyperGas documentation master file, created by
   sphinx-quickstart on Wed Jan 10 09:43:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HyperGas's Documentation!
=======================================

HyperGas is a python library for reading, processing, and writing data from Hyperspectral Imagers (HSI).
The reading function is built upon HSI readers
(`EMIT <https://github.com/pytroll/satpy/pull/2592>`_,
`EnMAP <https://github.com/pytroll/satpy/pull/2590>`_,
and PRISMA)
from the `Satpy <https://satpy.readthedocs.io/>`_ package.
Satpy converts HSI L1 data to the common Xarray :class:`~xarray.DataArray` and :class:`~xarray.Dataset` classes,
facilitating interoperability with HyperGas.

Key features of HyperGas include:

- Creating RGB (Red/Green/Blue) images by combining multiple bands
- Retrieving trace gases enhancements
- Denoising retrieval results
- Writing output data in various formats such as PNG, HTML, and CF standard NetCDF files
- Calculation of gas emission rates and saving them in CSV files

Go to the HyperGas project_ page for source code and downloads.

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
   :maxdepth: 2

   overview
   install
   config
   data_download
   quickstart
   batch_processing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
