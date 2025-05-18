=============
Plume Masking
=============

Plume masking is essential for estimating trace gas emissions.
However, using a constant value or a fixed standard deviation may result in inconsistent masks across different plumes.
To address this issue, HyperGas provides automatic plume detection and masking using
`watershed segmentation <https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_watershed.html>`_.

.. _a_priori_mask:

A priori mask
=============

Since the background field can be noisy, we recommend denoising the data first, as described in :doc:`denoising`.
After generating a denoised gas enhancement field, HyperGas uses the `tobac <https://github.com/tobac-project/tobac>`_
package to detect high-value features and create a mask via watershed segmentation.
For further details, refer to the `tobac's documentation <https://tobac.readthedocs.io/en/latest/>`_.

Briefly, the watershed method treats pixel values as a topographic surface and separates them into catchment basins.
Threshold values of 2 and 3 standard deviations are used to identify multiple localized high-enhancement features
and nearby areas with high enhancement values.

.. code-block:: python

   >>> from hypergas.a_priori_mask import Mask
   >>> import matplotlib.patches as patches

   >>> # assign the DataArray to Scene
   >>> hyp.scene['ch4_denoise'] = ch4_denoise

   >>> # run detection and masking
   >>> thresholds, features, da_plume_mask = Mask(hyp.scene, varname='ch4_denoise').get_feature_mask()

   >>> # plot the denoised data and mask
   >>> fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

   >>> ch4_denoise.plot(ax=axs[0, 0], vmin=ch4_denoise.mean())
   >>> da_plume_mask.plot(ax=axs[0, 1], cmap='tab20')

   >>> ch4_denoise.plot(ax=axs[1, 0], vmin=ch4_denoise.mean())
   >>> da_plume_mask.plot(ax=axs[1, 1], cmap='tab20')

   >>> # add zoom-in rectangle
   >>> for ax in axs[0, :]:
   >>>     rect = patches.Rectangle((700, 250), 100, 100, linewidth=1, edgecolor='r', facecolor='none')
   >>>     ax.add_patch(rect)
   >>>     ax.set_title('Full scene')

   >>> # plot zoom-in image
   >>> for ax in axs[1, :]:
   >>>     ax.set_xlim(700, 800)
   >>>     ax.set_ylim(250, 350)
   >>>     ax.set_title('Zoom in')

.. image:: ../fig/plume_mask.jpg

.. _pick_plume:

Pick plume
==========

As shown above, two plumes are visible within the red boundary box.
HyperGas can select a plume based on the source location provided by the user.
To ensure that the selected plumes originate from the same emission source,
HyperGas constrains the azimuth difference of the oriented envelope (minimum rotated rectangle) to less than 30°,
assuming minimal variation in wind direction near the source.
HyperGas then dilates the overlapping masks (e.g., by 180 m) and merges them,
using the mask that contains the emission source to identify those from the same origin.

The following steps were taken to create the plume mask for the northern plume,
which is truncated and exhibits a curved shape.
This required manual adjustments to the plume detection settings:
increasing the azimuth difference threshold from 30° to 180°, and
raising the plume distance limit from 180 m to 360 m.

.. code-block:: python

   >>> from hypergas.plume_utils import a_priori_mask_data

   >>> # set the plume source
   >>> lat_target = 40.2508
   >>> lon_target = 49.6322

   >>> # assign plume mask to the dataset
   >>> ds_emit['ch4_mask'] = da_plume_mask

   >>> # it will try to find the plume closest to the input location
   >>> mask, lon_mask, lat_mask, lon_target, lat_target = a_priori_mask_data(ds_emit, gas='ch4',
   ...                                                                       lon_target=lon_target, lat_target=lat_target,
   ...                                                                       pick_plume_name='plume0', wind_source='ERA5',
   ...                                                                       az_max=180, dist_max=360,
   ...                                                                      )

   >>> # plot the selected plume mask
   >>> fig, ax = plt.subplots()
   >>> mask.plot(ax=ax, x='longitude', y='latitude', cmap='Greys_r')
   >>> ax.scatter(lon_target, lat_target, marker='*', s=100)
   >>> ax.set_xlim(49.6, 49.7)
   >>> ax.set_ylim(40.2, 40.3)

.. image:: ../fig/plume_mask_sel.jpg
