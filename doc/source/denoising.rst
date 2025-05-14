=========
Denoising
=========

HyperGas applies a Chambolle total variance denoising
`(TV) filter <https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_tv_chambolle>`_
with the `J-Invariance calibration <https://scikit-image.org/docs/stable/auto_examples/filters/plot_j_invariant_tutorial.html>`_
to generate a smoothed gas enhancement field.
The denoising example on the scikit-image website explains it clearly.
Below is an image showing the differences in denoised results using various weights:

.. image:: ../fig/denoise_cat.png
   :alt: Denoising example from scikit-image
   :target: https://scikit-image.org/docs/stable/auto_examples/filters/plot_j_invariant_tutorial.html

Here is a real-world example of a denoised methane field using three different weights:

.. code-block:: python

   >>> from hypergas.denoise import Denoise

   >>> # denoise data and get calibrated weight
   >>> ch4_denoise = Denoise(hyp.scene, varname='ch4', method='calibrated_tv_filter').smooth()
   >>> denoise_weight = float(ch4_denoise.attrs['description'].split('weight=')[1][:-1])

   >>> # test different weights
   >>> ch4_denoise_low_weight = Denoise(hyp.scene, varname='ch4', method='tv_filter', weight=denoise_weight/2).smooth()
   >>> ch4_denoise_high_weight = Denoise(hyp.scene, varname='ch4', method='tv_filter', weight=denoise_weight*2).smooth()

   >>> # plot results
   >>> fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
   >>> axs = axs.flatten()

   >>> hyp_emit.scene['ch4'].where(hyp_emit.scene['segmentation']>0).plot(ax=axs[0], vmin=0, vmax=150)
   >>> ch4_emit_denoise_low_weight.where(hyp_emit.scene['segmentation']>0).plot(ax=axs[1], vmin=0, vmax=150)
   >>> ch4_emit_denoise.where(hyp_emit.scene['segmentation']>0).plot(ax=axs[2], vmin=0, vmax=150)
   >>> ch4_emit_denoise_high_weight.where(hyp_emit.scene['segmentation']>0).plot(ax=axs[3], vmin=0, vmax=150)

   >>> axs[0].set_title(f'Original data')
   >>> axs[1].set_title(f'Denoising weight = {int(emit_denoise_weight/2)}')
   >>> axs[2].set_title(f'Denoising weight = {int(emit_denoise_weight)}')
   >>> axs[3].set_title(f'Denoising weight = {int(emit_denoise_weight*2)}')

   >>> for ax in axs:
   >>>     ax.set_xlim(700, 800)
   >>>     ax.set_ylim(250, 350)

.. image:: ../fig/denoise_ch4.jpg

The TV filter with calibrated weight effectively removes background noise while preserving the structure of the gas plume.
In contrast, using the doubled weight over-smooths the field and obscures important features.
