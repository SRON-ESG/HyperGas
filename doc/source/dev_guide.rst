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

    co:
        name: carbon monoxide
        wavelength: [2305, 2385]
        full_wavelength: [1300,2500]
        rad_source: model
        concentrations: [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
        units: ppb

.. note::

    - The ``wavelength`` refers to a narrow strong absorption window, while ``full_wavelength`` encompasses a broader range.
    - Set ``rad_source`` as ``model`` since the ``lut`` option exclusively supports ch4 and co2.
    - Make sure that the ``concentrations (ppm)`` cover a typical emission range.
    - The term ``units`` specifies the units used for the L2 product outputs.
