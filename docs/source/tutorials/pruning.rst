Prune Branches
==============

Learn how to prune branches from gridded skeleton results to create a branchless structure (spine).

Pruning a Skeleton
------------------

To prune branches from a skeleton file and save the resulting spine:

.. code-block:: python

    from crispy.pruning import Skeleton

    # Define input and output file paths
    skel_file = "ridge_skel.fits"   # Gridded skeleton file
    spine_file = "ridge_spine.fits" # Output file for pruned skeleton (spine)

    # Load the skeleton and prune branches
    skel_obj = Skeleton.Skeleton(skel_file)
    skel_obj.prune_from_scratch()

    # Save the pruned skeleton to a file
    skel_obj.save_pruned_skel(spine_file, overwrite=True)


.. note::

   The 2D and 3D pruning algorithm in this module is based on the 2D algorithm from the `FilFinder` software.
   If you use this feature, please cite:

   Koch & Rosolowsky. "FilFinder: Filamentary structure in molecular clouds."
   (`2016 <https://ui.adsabs.harvard.edu/abs/2016ascl.soft08009K>`_).

Output
------

The pruned file, saved as `ridge_spine.fits`, contains the branchless skeleton (spine). It is compatible with
common FITS viewers like SAOImage DS9 or analysis tools such as `astropy`. The 3D images can also be quickly
visualized using CRISPy's tools (see :doc:`Visualize <visualize>`).