===================
Emission Estimation
===================

Two widely used methods for estimating source emission rates from plume observations are the integrated mass enhancement (IME) method
(`Varon et al. 2018 <https://doi.org/10.5194/amt-11-5673-2018>`_),
which relates the total plume mass enhancement to the emission rate through a wind-speed-dependent parameterization,
and the cross-sectional flux (CSF) method
(`Varon et al. 2018 <https://doi.org/10.5194/amt-11-5673-2018>`_,
`Kuhlmann et al. 2024 <https://doi.org/10.5194/gmd-17-4773-2024>`_),
which estimates the source rate as the product of methane enhancement and wind speed integrated across the plume width.


IME
===


We apply the IME method to estimate gas emission rates (:math:`Q` in kg h-1):

.. math::
   Q = \frac{IME \cdot U_{eff,IME} }{L}

where :math:`IME` is the total gas mass (kg) within the plume mask,
:math:`L` (m) is the square root of the plume area,
and :math:`U_{eff,IME}` is the effective wind speed (m/s).

The coefficiencies for wind calibration are saved in ``<HyperGas_dir>/hypergas/config.yaml``.
See :doc:`dev_guide` for more information.

CSF
===

Another method to calculate the emission rate is the CSF method.
This method is especially useful if gaps in the detected plume,
for example, caused by low albedo, make the estimate based on the total IME less reliable.

The source rate is estimated as the product of the cross-plume gas enhancement integral
and a different effective wind speed :math:`U_{eff,CSF}`):

.. math::
   Q = U_{eff,CSF} \int_{a}^{b} {\Delta}X(x,y) \,dy

Here, the x-axis aligns with the wind direction, while the y-axis is oriented perpendicular to it.
The integral is evaluated between the plume boundaries [a, b], as defined by the cross section in the plume mask.
The spacing between CSF sections is set to 2.5 $\times$ pixel resolution, rather than 1 $\times$ pixel resolution,
to reduce overlap between adjacent sections and ensure sufficient independence of sampled data.

A centerline curve is fitted to the plume mask using the `ddeq <https://ddeq.readthedocs.io/>`_ library
(`Kuhlmann et al. 2024 <https://doi.org/10.5194/gmd-17-4773-2024>`_),
which constructs a two-dimensional curve based on the pixels within the plume area
(`Kuhlmann et al. 2020 <https://doi.org/10.5194/amt-13-6733-2020>`_).
The plume length is defined as the distance along this curve from the source to the farthest detectable pixel.

Below is the plume case detected by EMIT:

.. image:: ../fig/estimation_case.jpg

To run the emission estimates, please refer to :doc:`batch_processing` for the workflow.
