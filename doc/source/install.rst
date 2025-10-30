=========================
Installation Instructions
=========================

Conda-based Installation
========================

TBD

Pip-based Installation
======================

TBD

Installation from source
========================

Follow the steps below to clone the repository and set up the environment for HyperGas.

Step 1: Clone the repository
----------------------------

.. code-block:: bash

    $ git clone git@github.com/SRON-ESG/HyperGas.git
    $ cd HyperGas

Step 2: Create an environment
-----------------------------

We recommend creating a separate environment for your work with HyperGas.
You can choose one of the following methods to create and set up your environment.

Using Miniforge (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Miniforge is the quickest way to install HyperGas dependencies:

1. Install Miniforge from `here <https://conda-forge.org/download/>`_.

2. Create the environment using ``mamba`` (the fast package manager for ``conda``): 

    .. code-block:: bash
    
        $ mamba env create -f environment.yml

3. Activate the environment to ensure all future Python or conda commands use this environment:

    .. code-block:: bash
    
        $ mamba activate hypergas

.. hint::

   If you want to activate the installed environment by default,
   you can add the activating line to your ``~/.bashrc``.

Using Anaconda or Miniconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Default method (slower)

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda env create -f environment.yml
    $ conda activate hypergas

Using mamba (faster)

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -n base mamba
    $ (optional: initialize shell) mamba shell hook --shell bash
    $ mamba env create -f environment.yml
    $ conda activate hypergas

.. note::

    If your ``conda`` version is older than 23.10, we recommend updating it
    to take advantage of the faster `mamba <https://conda.github.io/conda-libmamba-solver/user-guide/>`_ feature:

    .. code-block:: bash

        $ conda update -n base conda

Step 3: Install HyperGas
------------------------

Once the environment is set up, run ``pip install -e .`` inside the ``HyperGas`` folder.

Step 4: Update Satpy
--------------------

Because the hyperspectral readers are not yet merged into the official Satpy package,
you will need to install the development version of Satpy:

.. code-block:: bash

    $ pip install git+https://github.com/zxdawn/satpy.git@hyper

Step 5: Fix Spectral Python (SPy)
---------------------------------

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
