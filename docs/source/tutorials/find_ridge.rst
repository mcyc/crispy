Find Ridges
===========

In this tutorial, learn how to detect density ridges in `.fits` images and regrid teh results back into image space.

Setup & Detect Ridges
-------------------------

To detect ridges in an image:

.. code-block:: python

    from crispy import image_ridge_find as irf

    # Set up the input image and parameters
    image_file = "input_image.fits"  # Path to the input .fits image
    thres = 0.5                      # Minimum density threshold
    h = 2                            # Smoothing bandwidth

    # Detect ridges
    ridge_data = irf.run(image_file, h=h, thres=thres)


**Parameter Tips**

- ``h`` (smoothing bandwidth): controls ridge detection resolution. Use `h=2` for Nyquist-sampled images. Higher values reduce noise but lower detail.
- ``thres`` (density threshold): minimum intensity for valid pixels. Increase this value for noisier data and faster computing time.
- ``eps`` (convergence precision): lower values improve convergence accuracy but may slow down processing.
- **Walkers**: reduce the number of walkers for faster processing.

.. important::

    Computing time scales nonlinear with the number of walker and image points. Use the ``thres``
    parameter strategically to minimize the number of image points (i.e., voxels) needed for ridge detection.

Save Results
----------------

To save the ridge coordinates to a file:

.. code-block:: python

    # Save results to a text file
    ridge_file = "ridge_coords.txt"
    irf.write_output(coords=ridge_data, fname=ridge_file)


Grid Results
-----------------------

To map the ridge coordinates back onto the native image grid:

.. code-block:: python

    from crispy import grid_ridge as gr

    # Define file paths
    ridge_file = "ridge_coords.txt"     # Ridge coordinates
    image_file = "input_image.fits"     # Input reference image
    skel_file = "gridded_skel.fits"     # Output skeleton file

    # Grid ridge results
    gr.grid_skel(readFile=ridge_file, imgFile=image_file, writeFile=skel_file)

The output, `gridded_skel.fits`, contains the skeletonized ridge map aligned with the input image.
