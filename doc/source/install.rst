=========================
Installation Instructions
=========================

HyperGas can be installed through conda-forge (using conda), PyPI (using pip), or directly from source (using pip with Git).
The instructions below cover installing stable releases of HyperGas.
For guidance on installing the development or unstable version, see :doc:`dev_guide`.

Conda-based Installation
========================

You can set up HyperGas in a conda environment by pulling the package from the conda-forge channel.
If you don’t already have conda installed, the simplest option is to install `Miniforge <https://conda-forge.org/download/>`_, which provides a lightweight base system.

The examples below use ``-c conda-forge`` to ensure that all packages come specifically from the conda-forge repository.
If you prefer to make conda-forge your default source, you can run:

.. code-block:: bash

   $ conda config --add channels conda-forge

In a new conda environment
--------------------------

We suggest setting up a dedicated environment for working with HyperGas.
Creating an environment with HyperGas (and any other packages you want) right away is
typically quicker than creating the environment first and installing packages afterward.

Using Miniforge (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Miniforge is the quickest way to install HyperGas dependencies:

1. Create the environment using ``mamba`` (the fast package manager for ``conda``): 
   You can create a fresh environment and install HyperGas in a single step by running:

    .. code-block:: bash

        $ mamba create -c conda-forge -n hypergas_env python hypergas

2. After the environment is created, make sure to activate it so that any future Python or conda commands use the correct setup:

    .. code-block:: bash
    
        $ conda activate hypergas_env

.. hint::

   If you want to activate the installed environment by default,
   you can add the activating line to your ``~/.bashrc``.

Using Anaconda or Miniconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default method (slower)

.. code-block:: bash

    $ conda create -c conda-forge -n hypergas_env python hypergas
    $ conda activate hypergas_env

Using mamba (faster)

.. code-block:: bash

    $ conda install -n base mamba
    $ (optional: initialize shell) mamba shell hook --shell bash
    $ mamba create -c conda-forge -n hypergas_env python hypergas
    $ conda activate hypergas_env

.. note::

    If your ``conda`` version is older than 23.10, we recommend updating it
    to take advantage of the faster `mamba <https://conda.github.io/conda-libmamba-solver/user-guide/>`_ feature:

    .. code-block:: bash

        $ conda update -n base conda

In an existing environment
--------------------------

If you already have a conda environment active and want to add HyperGas to it, you can install it with:

.. code-block:: bash

    $ conda install -c conda-forge hypergas

Pip-based Installation
======================

HyperGas can be installed directly from the Python Package Index (PyPI).
If you’d like to set up an isolated environment for working with HyperGas, you can use virtualenv.

To install the core HyperGas package along with its essential Python dependencies, run:

.. code-block:: bash

    $ pip install hypergas


Note 1: Dependencies Requiring Manual Installation
==================================================

Users must install two dependencies manually due to installation constraints.
The `tobac <https://github.com/tobac-project/tobac>`_ package is not avaliable on PyPI.
Please check the `tobac documentation <https://tobac.readthedocs.io/en/latest/>`_
for installation instructions.
The `EnPT <https://git.gfz-potsdam.de/EnMAP/GFZ_Tools_EnMAP_BOX/EnPT>`_ package is not installed by default via PyPI,
because GDAL can cause installation errors.
Please check the `EnPT documentation <https://enmap.git-pages.gfz-potsdam.de/GFZ_Tools_EnMAP_BOX/EnPT/doc/installation.html>`_
for installation instructions.

Note 2: Update Satpy
====================

Because the hyperspectral readers are not yet merged into the official Satpy package,
you will need to install the development version of Satpy:

.. code-block:: bash

    $ pip install git+https://github.com/zxdawn/satpy.git@hyper

Note 3: Fix Spectral Python (SPy)
=================================

To prevent the ``np.linalg.inv`` singular matrix error,
you'll need to make a small modification in the Spectral Python package.

1. Locate your Spectral Python installation. You can find its path by running the following in Python:

.. code-block:: python

    import spectral
    print(spectral.__file__)

2. Open the file ``spectral/algorithms/algorithms.py`` and replace the line around 750:

.. code-block:: python

    np.linalg.inv(self._cov)

with:

.. code-block:: python

    np.linalg.pinv(self._cov)

For more details on this issue, refer to the `issue on GitHub <https://github.com/spectralpython/spectral/issues/159>`_.
