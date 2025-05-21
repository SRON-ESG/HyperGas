=================
Developer's Guide
=================

Adding a Custom Gas Retrieval
=============================

To add a custom gas retrieval to HyperGas, developers must ensure that both the absorption cross-section data (``absorption_cs_ALL_*.nc``) and the atmospheric profile (``atmosphere_*.dat``) include the relevant species.

Once the data is prepared, developers can proceed to modify the ``<HyperGas_dir>/hypergas/config.yaml`` file.
Below is an example of how to include carbon monoxide (CO):


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

Modifying wind calibrations
===========================

ALl wind calibration factors are stored in the ``<HyperGas_dir>/hypergas/config.yaml`` file.

Below is an example of IME calibration for a point source detected by EMIT:

.. code-block:: yaml

    ime_calibration: # ueff = alpha1*log(wspd) + alpha2 + alpha3*wspd
      ch4:
        point-source:
          EMIT:
            alpha1: 0.
            alpha2: 0.43
            alpha3: 0.35
            resid: 0.05

Build documentation
===================

HyperGas’s documentation is built using Sphinx.
All documentation is in the ``doc/`` directory of the project repository.
For building the documentation, additional packages are needed. These can be installed with

.. code-block:: bash

    pip install -e ".[all]"

After editing the source files there, the documentation can be generated locally:

.. code-block:: bash

    cd doc
    make html

Your ``build`` directory should contain an ``index.html`` that you can open in your browser.
