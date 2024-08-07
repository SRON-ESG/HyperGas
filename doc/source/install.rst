=========================
Installation Instructions
=========================

Clone the repository first and choose perferred installation method below.

.. code-block:: bash

    git clone git@gitlab.sron.nl:esg/tropomi-l4/hsi/HyperGas.git
    cd HyperGas

Step 1: Create Env
==================

Install all packages from the ``environment.yml`` file:

.. code-block:: bash

    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda env create -f environment.yml

If you use ``mamba``, you can run:

.. code-block:: bash

    mamba env create -f environment.yml

You need to activate the new environment before importing hypergas.

.. code-block:: bash

    conda activate hypergas

If you want to activate it by default, you can add the above line to your ``~/.bashrc``.

Finally, run ``pip install -e .`` inside the ``HyperGas`` folder.

Step 2: Update Satpy
====================

Because the Hyperspectral readers have not been merged, you need to update the satpy package after the basic installation.

.. code-block:: bash

    pip install git+https://github.com/zxdawn/satpy.git@hyper


Step 3: Fix spectral
====================

Edit ``spectral/algorithms/algorithms.py`` to prevent the ``np.linalg.inv`` singular matrix error
(See this `issue <https://github.com/spectralpython/spectral/issues/159>`_).

.. code-block:: python

    # You can find the <spectral_path> like this:
    import spectral
    print(spectral.__file__)
