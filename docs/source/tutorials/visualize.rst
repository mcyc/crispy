Visualization
======================

Learn how to visualize CRISPy's results in interactive 3D plots, along with the data cube from which
the ridges were identified. Use the navigation bar to skip ahead to a specific topic.

Volume Rendering in 3D
---------------------------

Ridge visualization are more useful in the context of the original images. CRISPy provides two options
for 3D volume rendering: point cloud and isosurface volume.


Point Cloud Rendering
^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    Point cloud rendering is computationally light and recommended for typical usage.

To volume render the image cube efficiently using :func:`render_point_cloud() <crispy.visualize.render_point_cloud>`:

.. code-block:: python

    from crispy.visualize import render_point_cloud
    fig = render_point_cloud(cube, showfig=True, bins=15, vmin=None)

The result: an interactive 3D scatter plot with semi-transparent, color-coded points representing the volume
structures in the cube image.

Isosurface Rendering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To volume render with :func:`render_volume() <crispy.visualize.render_volume>` using layers of semi-transparent
isosurfaces:

.. code-block:: python

    from crispy.visualize import render_volume
    fig = render_volume(cube, showfig=True, surface_count=10, vmin=None)

Result: a smooth 3D rendering of the data cube visualized with isosurfaces.

.. tip::

    **Minimum value**: to conserve computation strategically, use a ``vmin`` value similar to the ``thres`` value
    used for ridge detection when visualising with
    :func:`render_point_cloud() <crispy.visualize.render_point_cloud>` and
    :func:`render_volume() <crispy.visualize.render_volume>` to render only the structures
    that host ridges.

    **Marker size**: depending on the size of image, adjust the ``size`` parameter in
    :func:`render_point_cloud() <crispy.visualize.render_point_cloud>` as needed to improve the
    appearance of the render.

Visualizing Ridges
--------------------------

Pre-gridded Ridges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To plot the pre-gridded ridges with :func:`ridge_trace_3D() <crispy.visualize.ridge_trace_3D>` from saved results,
first create a trace object:

.. code-block:: python

    from crispy.visualize import ridge_trace_3D
    from crispy.grid_ridge import read_table

    # read the saved results
    rname = 'ridges.txt'
    ridge = read_table(rname, useDict=False)
    x,y,z = ridge[0], ridge[1], ridge[2]

    # create a trace object for plotting
    trace = ridge_trace_3D(x, y, z, size=2, color='darkred')

then plot it either over the previously rendered volume:

.. code-block:: python

    fig.add_trace(trace)
    fig.show()

or as a standalone plot:

.. code-block:: python

   import plotly.graph_objects as go
   go.Figure(data=[trace]).show()

Gridded Skeletons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To plot ridges that has already been gridded back to the image space (i.e., skeletons) or their pruned
counterparts (i.e., spines), use either :func:`mask_trace_3D() <crispy.visualize.mask_trace_3D>` to create a
trace and plot their coordinates with a 3D scatter plot over the previously rendered structures as:

.. code-block:: python

    from crispy.visualize import mask_trace_3D
    trace mask_trace_3D(skeleton, showfig=True, opacity=0.5)
    fig.add_trace(trace)
    fig.show()

or :func:`skel_volume() <crispy.visualize.skel_volume>` to plot them as isosurface volumes using the
following:

.. code-block:: python

    from crispy.visualize import skel_volume
    fig = skel_volume(skeleton, showfig=True, opacity=0.5, fig=fig)

Saving Plots
--------------------------

To save the results as an interactive HTML file:

.. code-block:: python

    fig.write_html("ridges_3D.html")

For more, see the :doc:`API Reference <../api/index>`.
