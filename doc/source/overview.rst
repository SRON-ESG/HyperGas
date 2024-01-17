========
Overview
========

HyperGas is designed to facilitate the retrieval of trace gases for HSI instruments with ease.
All necessary details for performing these operations are internally configured in HyperGas.
This means that users do not need to concern themselves with the specific implementation,
but rather focus on their desired outcome.
Most of the features offered by HyperGas can be customized using keyword arguments.
The following sections describe the various components and concepts of HyperGas.
Additionally, the :doc:quickstart guide presents straightforward example of HyperGas.

Reading
=======

HyperGas uses `Satpy <https://satpy.readthedocs.io/>`_ to directly read HSI L1 data,
which offers support for a wide range of satellite datasets.
For detailed information, please refer to Satpy's documentation.
Since HSI file formats vary across different instruments,
we have integrated multiple HSI readers into Satpy, ensuring a standardized data loading interface.
This makes it easy to add new HSI data for HyperGas.

RGB Composite
=============

HyperGas applied `HSI2RGB <https://github.com/JakobSig/HSI2RGB>`_ to generate the RGB image from HSI L1 data.
If this method failed, bands data close to 650, 560, and 470 nm are combined with a Gamma norm.

Retrieval
=========

HyperGas emploies a linearized matched filter to retrieve the trace gas enhancements.
This technique has been successfully applied to both satellite and aircraft observations.
The matched filter assumes a spectrally flat background and models the background radiance spectrum as a Gaussian distribution
with a mean vector :math:`\mu` and a covariance matrix :math:`\Sigma`.
The radiance spectrum (:math:`L`) can be represented by two hypotheses: H0, which assumes the absence of plume, and H1, where the plume is present.

.. math::
    H_0: L \sim \mathcal{N}(\mu,\Sigma);
    H_1: L \sim \mathcal{N}(\mu+\alpha t, \Sigma)

Here, :math:`t` represents the target signature, which is the product of two components:
the background radiance (:math:`\mu`) and the negative gas absorption coefficient (:math:`k`).
To calculate :math:`k`, we employ a forward model and convolve it with the imager's central wavelength and full width at half maxima (FWHM).
The scale factor :math:`\alpha` is derived from the first-order Taylor expansion of Beer-Lambert's law.
The maximum likelihood estimate of :math:`\alpha` is:

.. math::
    \alpha = \frac{(t-\mu)^T\Sigma^{-1}(L-\mu)}{(t-\mu)^T\Sigma^{-1}(t-\mu)} 

Denoising
=========

To mitigate the noisy background, we initially perform the same retrieval over the 1300 :math:`\sim` 2500 nm window.
Then, we apply a Chambolle total variance denoising
`(TV) filter <https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle>`_
to obtain a smoothed enhancement field.
The TV filter aims to minimize the cost function between the original and smoothed images.
Considering the lower SNR value of PRISMA, we select a denoising weight of 90, which is higher than the weight of 50 used for EMIT and EnMAP.

Writing
=======

HyperGas enables users to save data in various formats, including PNG, HTML, and data file formats such as NetCDF.
Please refer to the documentation on writing (see :doc:writing) for detailed information.

Emission Rates
==============

