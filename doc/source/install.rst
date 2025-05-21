=========================
Installation Instructions
=========================

Clone the repository first and choose perferred installation method below.

.. code-block:: bash

    $ git clone git@github.com:zxdawn/HyperGas.git
    $ cd HyperGas

Step 1: Create Env
==================

Using Miniforge (recommended)
-----------------------------

Using `Miniforge <https://conda-forge.org/download/>`_ is the fastest way to install HyperGas:

.. code-block:: bash

    $ mamba env create -f environment.yml
    $ mamba activate hypergas

.. hint::

   If you want to activate the installed environment by default,
   you can add the activating line to your ``~/.bashrc``.

Using Anaconda or Miniconda
---------------------------

Default way (slow)
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda env create -f environment.yml
    $ conda activate hypergas


Using mamba (faster)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda install -n base mamba
    $ mamba env create -f environment.yml
    $ conda activate hypergas

.. note::

    If your ``conda`` version is older than 23.10, we recommend updating it
    to take advantage of the faster `mamba <https://conda.github.io/conda-libmamba-solver/user-guide/>`_ feature:

    .. code-block:: bash

        $ conda update -n base conda

Step 2: Install HyperGas
========================

Run ``pip install -e .`` inside the ``HyperGas`` folder.

Step 3: Update Satpy
====================

Because the Hyperspectral readers have not been merged, you need to update the satpy package after the basic installation.

.. code-block:: bash

    $ pip install git+https://github.com/zxdawn/satpy.git@hyper

Step 4: Fix Spectral Python (SPy)
=================================

Edit ``spectral/algorithms/algorithms.py`` to prevent the ``np.linalg.inv`` singular matrix error
(See this `issue <https://github.com/spectralpython/spectral/issues/159>`_).

.. code-block:: python

    # You can find the <spectral_path> like this:
    import spectral
    print(spectral.__file__)
