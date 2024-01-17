=========================
Installation Instructions
=========================

Clone the repository first and choose perferred installation method below.

.. code-block:: bash

    git clone git@gitlab.sron.nl:esg/tropomi-l4/hsi/HyperGas.git


Conda (recommended)
===================

Install all packages from the ``environment.yml`` file:

.. code-block:: bash

    conda env create -f environment.yml

If you use ``mamba``, you can run:

.. code-block:: bash

    mamba env create -f environment.yml

You need to activate the new environment before importing hypergas.

.. code-block:: bash

    conda activate hypergas

If you want to activate it by default, you can add the above line to your ``~/.bashrc``.

Pip (not recommended)
=====================

Run ``pip install -e .`` inside the ``HyperGas`` folder.
Please note that HyperGas depends on multiple packages that may cause problems when installed with pip.

Update Satpy (necessary)
========================

Because the Hyperspectral readers have not been merged, you need to update the satpy package after the basic installation.

.. code-block:: bash

    pip install git+https://github.com/zxdawn/satpy.git@hyper
