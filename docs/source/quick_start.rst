Quick Start
===========
    Let's get started with **CRISPy**. Use the navigation bar on the right to skip ahead
    as needed.

Ridge Finding
~~~~~~~~~~~~~

.. note::

   **Install:** If you haven't installed CRISPy yet, see :doc:`Install <installation>`.

To detect density ridges in a `.fits` image:

.. code-block:: python

    from crispy import image_ridge_find as irf

    # Input parameters
    filename = "input_image.fits"   # Input image file
    thres = 0.5                     # Minimum density threshold for valid data points
    h = 2                           # Smoothing bandwidth (recommended: Nyquist-sampled resolution)

    # Finding ridges
    ridge_data = irf.run(filename, h=h, thres=thres)

To save the results to a `.txt` file

.. code-block:: python

    # Save the results to a file
    irf.write_output(ridge_data, "output_ridges.txt")

For more details, please see :doc:`Tutorials <tutorials/index>`