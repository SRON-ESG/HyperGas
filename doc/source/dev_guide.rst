=================
Developer's Guide
=================

Adding a Custom Gas Retrieval
=============================

In order to add a customed gas retrieval to HyperGas,
developers must ensure that the absorption cross section data (``absorption_cs_ALL_*.nc``)
and atmosphere profile (``atmosphere_*.dat``) contain the relevant species.

Once the data is prepared, developers can proceed to adjust the ``<HyperGas_dir>/hypergas/config.yaml`` file.
Below is an illustration of how to include carbon monoxide (CO):


.. code-block:: yaml

    co2:
        name: carbon dioxide
        wavelength: [1930, 2200]
        full_wavelength: [1300, 2500]
        rad_source: model
        concentrations: [0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0]  # for rad_source=="model", units: ppm
        # concentrations: [0.0e+4, 2.0e+4, 4.0e+4, 8.0e+4, 1.6e+5, 3.2e+5, 6.4e+5, 1.28e+6] # for rad_source=="lut", units: ppm m
        units: ppm

.. note::

    - The ``wavelength`` refers to a narrow strong absorption window, while ``full_wavelength`` encompasses a broader range.
    - Set ``rad_source`` as ``model`` since the ``lut`` option exclusively supports ch4 and co2.
    - Make sure that the ``concentrations (ppm)`` cover a typical emission range.
    - The term ``units`` specifies the units used for the L2 product outputs.
