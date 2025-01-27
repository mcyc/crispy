Tutorials
=========

    Welcome to tutorials! We will go through the basics on how to detect density ridges from a `.fits`
    image and post-process the results.


Basic Usage
~~~~~~~~~~~

Here, we will focus on how to identify density ridges in a 2D or 3D `.fits`
image and post-process the results. Specifically, we will go over how to grid the results
back into the image space and prune branches from the gridded results.

Find Ridges
^^^^^^^^^^^^^^^^^^^^^^^^

To detect density ridges in a `.fits` image :

.. code-block:: python

    from crispy import image_ridge_find as irf

    # Set up the data and input parameters
    image_file = "input_image.fits" # Input .fits image file
    thres = 0.5                     # Minimum density threshold for valid data points
    h = 2                           # Smoothing bandwidth (recommended: Nyquist-sampled resolution)

    # Find ridges (e.g., filaments)
    ridge_data = irf.run(image_file, h=h, thres=thres)

To save the results to a `.txt` file:

.. code-block:: python

    # Save the results to a file
    ridge_file = "ridge_coords.txt"
    irf.write_output(coords=ridge_data, fname=ridge_file)

**Tips on Parameter Choices:**

    - **Smoothing Bandwidth** (``h``): Controls the resolution of ridge detection. For Nyquist-sampled images,
      start with ``h=2``. Larger values suppress noise at the expense of resolution.
    - **Density Threshold** (``thres``): The minimum intensity for a pixel to be included in the run.
    - **Convergence Criterion** (``eps``): Defines the precision for ridge convergence. Lower precision are
      faster to converge.
    - **Walker Parameters**: Include options for their initial placements. Reducing the number of walkers
      needed can greatly reduce the run time needed.


Grid Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To grid the ridge detection results back onto the native grid of the input image:

.. code-block:: python

    from crispy import grid_ridge as gr

    # Input and output file paths
    ridge_file = "ridge_coords.txt"     # File containing the ridge coordinates
    image_file = "input_image.fits"     # Input/reference image file
    skel_file = "gridded_skel.fits"     # File to save gridded results (skeletons)

    # Grid ridge results
    gr.grid_skel(readFile=ridge_file, imgFile=image_file, writeFile=skel_file)

Prune Branches
^^^^^^^^^^^^^^^^^^^^^^^
To prune branches from the gridded results (i.e., skeletons) into a branchless structure (i.e., spine):

.. code-block:: python

    from crispy.pruning import Skeleton

    # Input and output file paths
    skel_file = "ridge_skel.fits"   # Gridded skeleton file
    spine_file = "ridge_spine.fits" # Output file for pruned skeletons (spines)

    # Load skeleton and apply pruning
    skel_obj = Skeleton.Skeleton(skel_file)
    skel_obj.prune_from_scratch()

    # Save the pruned skeleton
    skel_obj.save_pruned_skel(spine_file, overwrite=True)


.. note ::

    If you use the branch-pruning function, please also cite the `FilFinder` software for the
    2D pruning algorithm from which `MUFASA`'s 3D version is based on:

    Koch & Rosolowsky. "FilFinder: Filamentary structure in molecular clouds."
    (`2016 <https://ui.adsabs.harvard.edu/abs/2016ascl.soft08009K>`_).
