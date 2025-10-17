========
Overview
========

HyperGas is designed to facilitate the retrieval of trace gases for HSI instruments with ease.
All necessary details for performing these operations are internally configured in HyperGas.
This means that users do not need to concern themselves with the specific implementation,
but rather focus on their desired outcome.
Most of the features offered by HyperGas can be customized using keyword arguments.

The following sections describe the various components and concepts of HyperGas.
Additionally, the :doc:`quickstart` guide presents straightforward example of HyperGas.
After understanding each step, users can proceed to the :doc:`batch_processing` page for batch processing.

.. image:: ../fig/workflow.jpg
   :alt: Mind map link: https://gitmind.com/app/docs/mwe3luac

Reading
=======

HyperGas uses `Satpy <https://satpy.readthedocs.io/>`_ to directly read HSI L1 data,
which offers support for a wide range of satellite datasets.
For detailed information, please refer to Satpy's documentation.
Since HSI file formats vary across different instruments,
we have integrated multiple HSI readers into Satpy, ensuring a standardized data loading interface.
This makes it easy to add new HSI data for HyperGas.
See :doc:`reading` for more information.

Retrieval
=========

HyperGas emploies a linearized matched filter to retrieve the trace gas enhancements.
This technique has been successfully applied to both satellite and aircraft observations.
HyperGas applies the matched filter to each cluster separately to account for the different background signals in land and water pixels.
See :ref:`databases` for more information about watermask.

Besides the linear matched filter, HyperGas also supports lognormal matched filter and Cluster-tuned matched filter.
See :ref:`algorithms` for more information.

Orthorectification
==================

Hyperspectral Level 1 data is provided in sensor geometry,
which means the image data has only image coordinates rather than map coordinates.
HyperGas addresses this limitation by supporting both automatic and manual orthorectification methods.
See :doc:`orthorectification` for more information.

Denoising
=========

To mitigate the noisy background, we perform the matched filter over a wider spectral range (e.g., 1300 :math:`\sim` 2500 nm for methane and carbon dioxide).
Then, we apply a Chambolle total variance denoising
`(TV) filter <https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle>`_
with the `J-Invariance calibration <https://scikit-image.org/docs/stable/auto_examples/filters/plot_j_invariant_tutorial.html>`_
to obtain a smoothed gas enhancement field, which is used for generating plume masks.
The TV filter aims to minimize a cost function between the original and smoothed images.
See :doc:`denoising` for more information.

Plume masking
=============

HyperGas uses a two-step process for plume masking.
The first step is using `tobac <https://github.com/tobac-project/tobac>`_
to automatically generate masks.
The second step is selecting plumes by assigning a plume marker.
See :doc:`plume_mask` for more information.

Emission estimation
===================

HyperGas supports two widely used methods for emission estimation:
Integrated Mass Enhancement (IME) and Cross-Sectional Flux (CSF).
See :doc:`emission` for more information.
