=================
Developer's Guide
=================

.. _devinstall:

Development installation
========================

For general installation guidance, please check the :doc:`install` page.
When setting up HyperGas for development, work from a cloned Git repository and install it in editable mode to ensure your local modifications are picked up automatically.
It’s best practice to create an isolated environment, such as a conda environment for this purpose.
For example, with conda you might run:

.. code-block:: bash

    $ conda create -n hypergas-dev python=3.12
    $ conda activate hypergas-dev

This will set up a fresh conda environment named "hypergas-dev" that uses Python 3.12.
After that, the second command activates the environment so any subsequent conda, python, or pip operations will run inside it.

If you intend to contribute to the project, start by forking the repository and cloning your own fork.
Once you have your copy of the code, you can install HyperGas in development mode by running:

.. code-block:: bash

    $ conda install --only-deps hypergas
    $ pip install -e .

The first command pulls in all of the dependencies required by the HyperGas package on conda-forge,
but does not install HyperGas itself.
The second command must be executed from the top-level directory of your cloned HyperGas project
(the one containing the ``pyproject.toml`` file), and this is what installs the package in editable mode.

At this point, any changes you make to the Python files in your clone will be picked up immediately by your active conda environment.

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
