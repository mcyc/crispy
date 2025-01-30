import numpy as np
from astropy.io import fits
from astropy.stats import median_absolute_deviation as mads
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def render_point_cloud(cube, savename=None, showfig=False, bins=15, vmin=None, vmax=None,
                       cmap="magma_r", z_stretch=1, size=2, fig=None, cbar_label="",
                       min_opacity=0.01, max_opacity=0.3):
    """
    Render a 3D scatter plot from a 3D data cube with efficient visualization.

    The function generates a 3D point cloud visualization from a 3D numpy array,
    coloring the points based on their values and assigning opacity based on percentile bins.
    A single colorscale spans all bins.

    Parameters
    ----------
    cube : numpy.ndarray
        A 3D numpy array of float values to visualize. NaN values are ignored.
    savename : str, optional
        Path to save the interactive HTML file. If None, the figure is not saved. Default is None.
    showfig : bool, optional
        Whether to display the figure interactively. Default is False.
    bins : int, optional
        Number of percentile bins for dividing the data. Default is 15.
    vmin : float, optional
        Minimum value for normalization. If None, the 10th percentile of the data is used. Default is None.
    vmax : float, optional
        Maximum value for normalization. If None, the 99th percentile of the data is used. Default is None.
    cmap : str, optional
        Colormap for the data points. Uses Plotly-compatible colormap names. Default is "magma_r".
    z_stretch : float, optional
        Scaling factor for the Z-axis to modify aspect ratio. Default is 1.
    size : int, optional
        The marker (i.e., point) size used to render the point cloud. Default is 2
    fig : plotly.graph_objects.Figure, optional
        Existing Plotly figure to add the scatter plot to. If None, a new figure is created. Default is None.
    cbar_label : str, optional
        Label for the colorbar. Default is an empty string.
    min_opacity : float, optional
        Minimum value of the marker opacity. Default is 0.01
    max_opacity : float, optional
        Maximum value of the marker opacity. Default is 0.3
    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object containing the 3D scatter plot.

    Notes
    -----
    - Points outside the range [vmin, vmax] are excluded.
    - Percentile bins determine point opacity for better depth visualization.
    - A shared color axis is used for consistency across bins.

    Examples
    --------
    >>> import numpy as np
    >>> X, Y, Z = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    >>> cube = np.sin(np.pi * X) * np.cos(np.pi * Z) * np.sin(np.pi * Y)
    >>> render_point_cloud(cube, showfig=True, bins=4)
    """
    # ensures plotly-compatible endianness
    cube = _ensure_endianness(cube)

    # Flatten the array and remove NaN values
    valid_values = cube[np.isfinite(cube)].flatten()

    # Determine vmin and vmax if not provided
    if vmin is None:
        vmin = np.percentile(valid_values, 10)
    if vmax is None:
        # filter outliers and add a small padding
        vpad = 0.5 / bins if bins < 5 else 0.1
        vmax = np.percentile(valid_values, 99.5) * (1 + vpad)

    # Mask values outside the range [vmin, vmax]
    mask = np.isfinite(cube) & (cube >= vmin) & (cube <= vmax)
    masked_values = cube[mask]
    z, y, x = np.where(mask)

    # ensures plotly-compatible endianness
    x = _ensure_endianness(x)
    y = _ensure_endianness(y)
    z = _ensure_endianness(z)

    # Compute percentiles and opacity levels
    percentiles = np.linspace(0, 100, bins + 1)
    cutoffs = np.percentile(masked_values, percentiles)
    opacity_levels = np.linspace(min_opacity, max_opacity, bins)

    # Use provided figure or create a new one
    if fig is None:
        fig = go.Figure()

    # Add a scatter trace for each bin
    for i in range(bins):
        bin_mask = (masked_values >= cutoffs[i]) & (masked_values < cutoffs[i + 1])
        sub_x = x[bin_mask]
        sub_y = y[bin_mask]
        sub_z = z[bin_mask]
        sub_values = masked_values[bin_mask]

        fig.add_trace(go.Scatter3d(
            x=sub_x,
            y=sub_y,
            z=sub_z,
            mode='markers',
            marker=dict(
                size=size,
                color=sub_values,
                colorscale=cmap,
                cmin=vmin,
                cmax=vmax,
                opacity=opacity_levels[i],
                coloraxis="coloraxis",  # Link to the shared color axis
                showscale = False # don't show colorbar for individual bins
            ),
            name=f'Bin {i + 1} ({cutoffs[i]:.2f}-{cutoffs[i + 1]:.2f})',
            showlegend=False
        ))

    # Adjust aspect ratio based on array dimensions
    shape = cube.shape
    max_dim = max(shape)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, shape[2]], title='X'),
            yaxis=dict(range=[0, shape[1]], title='Y'),
            zaxis=dict(range=[0, shape[0]], title='Z'),
            aspectratio=dict(
                x=shape[2] / max_dim,
                y=shape[1] / max_dim,
                z=shape[0] / max_dim * z_stretch
            )
        ),
        coloraxis=dict(
            colorscale=cmap,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title=cbar_label)
        ),  # Define a shared color axis
        title="",
        showlegend=True
    )

    # Save the figure if savename is provided
    if savename is not None:
        fig.write_html(savename)

    # Show the figure if showfig is True
    if showfig:
        fig.show()

    return fig


def ridge_trace_3D(x, y, z, size=2.5, color='black', opacity=0.5, name='ridges'):
    """
    Create a 3D scatter trace for visualizing ridge points.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates of the ridge points in 3D space.
    size : int, optional
        Marker size. Default is 2.
    color : str, optional
        Marker color. Default is "black".
    opacity : float, optional
        Opacity of the markers. Default is 0.5.
    name : str, optional
        Name for the trace. Default is "ridge".

    Returns
    -------
    plotly.graph_objects.Scatter3d
        A Plotly 3D scatter trace.
    """
    # create trace, with plotly-compatible endianness ensured
    trace = go.Scatter3d(
        x=_ensure_endianness(x),
        y=_ensure_endianness(y),
        z=_ensure_endianness(z),
        mode='markers',
        marker=dict(
            size=size,  # Marker size
            color=color,  # Static color
            opacity=opacity
        ),
        name=name,
        showlegend = False
    )
    return trace


def mask_trace_3D(mask3D, size=2.5, color='black', opacity=0.9, name='spines'):
    """
    Generate a 3D scatter plot of True-valued positions in a 3D boolean array.

    Parameters
    ----------
    mask3D : numpy.ndarray
        A 3D boolean numpy array where True values indicate points to be plotted.
    size : int, optional
        Size of the scatter plot markers. Default is 2.
    color : str, optional
        Color of the markers. Default is "black".
    opacity : float, optional
        Opacity of the markers, ranging from 0 (transparent) to 1 (opaque). Default is 0.9.
    name : str, optional
        Label for the trace in the Plotly legend. Default is "spines".

    Returns
    -------
    plotly.graph_objects.Scatter3d
        A Plotly 3D scatter trace representing the masked points.
    """
    # Find the indices of True values in the 3D array
    z, y, x = np.where(mask3D)

    # Create a scatter plot, ensures plotly-compatible endianness
    trace = go.Scatter3d(
        x=_ensure_endianness(x),  # x-coordinates
        y=_ensure_endianness(y),  # y-coordinates
        z=_ensure_endianness(z),  # z-coordinates
        mode='markers',
        marker=dict(
            size=size,  # Size of the markers
            color=color,  # Color of the markers
            symbol='circle',
            opacity=opacity
        ),
        name=name,
        showlegend = False
    )

    return trace


def _get_xyz(cube):
    """
    Generate 3D coordinate grids for a data cube.
    """
    im = cube
    nx, ny, nz = im.shape[2], im.shape[1], im.shape[0]
    z, y, x = np.meshgrid(np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1), indexing='ij')

    # ensures plotly-compatible endianness
    x = _ensure_endianness(x)
    y = _ensure_endianness(y)
    z = _ensure_endianness(z)
    return x, y, z


def skel_volume(image, savename=None, showfig=True, opacity=0.75, colorscale='inferno', fig=None, z_stretch=1,
                cbar_label=''):
    """
    Render a 3D skeleton volume using isosurface visualization.

    This function provides a specialized version of `render_volume()` for visualizing skeleton-like structures
    or spines in 3D datasets. It simplifies the parameterization for this specific use case.

    Parameters
    ----------
    image : numpy.ndarray
        A 3D binary or float array representing the skeleton structure. If the data type is boolean, it
        will be converted to an integer array for visualization.
    savename : str, optional
        Path to save the interactive HTML file. If None, the figure is not saved. Default is None.
    showfig : bool, optional
        Whether to display the figure interactively. Default is True.
    opacity : float, optional
        Opacity of the skeleton isosurfaces. Default is 0.75.
    colorscale : str, optional
        Colormap for the isosurfaces. Uses Plotly-compatible colormap names. Default is "inferno".
    fig : plotly.graph_objects.Figure, optional
        Existing Plotly figure to add the volume rendering to. If None, a new figure is created. Default is None.
    z_stretch : float, optional
        Scaling factor for the Z-axis to modify aspect ratio. Default is 1.
    cbar_label : str, optional
        Label for the colorbar. Default is an empty string.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object containing the 3D skeleton volume rendering.

    Notes
    -----
    - This function is a wrapper around `render_volume()` with preset parameters tailored for skeleton visualization.
    - If `image` contains NaN values, they will be handled by `render_volume()`.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.random((50, 50, 50)) > 0.95  # Generate a random binary skeleton
    >>> skel_volume(image, showfig=True, opacity=0.5)
    """
    if isinstance(image, bool):
        image = image.astype(np.uint8)

    # ensures plotly-compatible endianness
    image = _ensure_endianness(image)

    return render_volume(image, savename=savename, showfig=showfig, vmin=1e-3, vmax=1e-2, surface_count=1,
                         opacity=opacity, colorscale=colorscale, showscale=False, fig=fig, z_stretch=z_stretch,
                         cbar_label=cbar_label)


def render_volume(cube, savename=None, showfig=False, vmin=None, vmax=None, surface_count=12,
                  opacity=None, colorscale='YlGnBu', z_stretch=1, showscale=True, fig=None,
                  val_fill=0.0, cbar_label=''):
    """
    Render a 3D volume using layers of isosurfaces.

    This function creates a 3D interactive volume rendering visualization of a 3D data cube
    by plotting multiple isosurfaces at specified value intervals.

    Parameters
    ----------
    cube : numpy.ndarray or str
        A 3D numpy array of float values or a file path to a FITS file. If the input is a file path,
        the data will be loaded and used. NaN values are replaced with `val_fill` before rendering.
    savename : str, optional
        Path to save the interactive HTML file. If None, the figure is not saved. Default is None.
    showfig : bool, optional
        Whether to display the figure interactively. Default is False.
    vmin : float, optional
        Minimum isosurface value. If None, the 10-sigma level above the estimated RMS is used. Default is None.
    vmax : float, optional
        Maximum isosurface value. If None, the 99.99th percentile of the data is used. Default is None.
    surface_count : int, optional
        Number of isosurfaces to plot. Higher values create finer visualization but increase rendering cost. Default is 12.
    opacity : float, optional
        Opacity of the isosurfaces. If None, it is set to `2 / surface_count` for semi-transparency. Default is None.
    colorscale : str, optional
        Colormap for the isosurfaces. Uses Plotly-compatible colormap names. Default is "YlGnBu".
    z_stretch : float, optional
        Scaling factor for the Z-axis to modify aspect ratio. Default is 1.
    showscale : bool, optional
        Whether to display the color scale bar. Default is True.
    fig : plotly.graph_objects.Figure, optional
        Existing Plotly figure to add the volume rendering to. If None, a new figure is created. Default is None.
    val_fill : float, optional
        Value to replace NaN voxels in the cube. Default is 0.0.
    cbar_label : str, optional
        Label for the colorbar. Default is an empty string.

    Returns
    -------
    plotly.graph_objects.Figure
        The Plotly figure object containing the 3D volume rendering.

    Notes
    -----
    - NaN values in the cube are replaced with `val_fill` before visualization.
    - Isosurface values range between `vmin` and `vmax`, divided into `surface_count` levels.
    - The visualization requires non-NaN input data for accurate results.

    Examples
    --------
    >>> import numpy as np
    >>> X, Y, Z = np.mgrid[-1:1:30j, -1:1:30j, -1:1:30j]
    >>> cube = np.sin(np.pi * X) * np.cos(np.pi * Z) * np.sin(np.pi * Y)
    >>> render_volume(cube, showfig=True, surface_count=10)
    """
    if isinstance(cube, str):
        cube, hdr = fits.getdata(cube, header=True)
        cube = cube.copy()

    # note: the visualization may not work if NaN values are present. Try replacing NaN with values like zeros.
    cube[np.isnan(cube)] = val_fill

    X, Y, Z = _get_xyz(cube) # x,y,z endianess already checked

    # ensures plotly-compatible endianness
    cube = _ensure_endianness(cube)

    if fig is None:
        fig = make_subplots(rows=1, cols=1)

    if vmax is None:
        # use 99.99 percentile
        vmax = np.percentile(cube, 99.99)
        print("no vamx provided, using the 99.99 percentile value: {}".format(np.round(vmax,2)))

    if vmin is None:
        # use the estimated 10-sigma
        vmin = mads(cube, ignore_nan=True)*10.0

        if vmin > vmax:
            vmin = vmax/2.0
        print("no vmin provided, using the 10 sigma above the rms value: {}".format(np.round(vmin,2)))

    if opacity is None:
        if surface_count > 2:
            opacity = 2.0/surface_count
        else:
            opacity = 1.0

    fig.add_volume(x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=cube.flatten(),
        colorscale=colorscale,
        isomin=vmin,
        isomax=vmax,
        opacity=opacity,
        surface_count=surface_count,
        showscale = showscale
    )

    # Adjust aspect ratio based on cube dimensions
    shape = cube.shape
    max_dim = max(shape)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, shape[2]], title='X'),
            yaxis=dict(range=[0, shape[1]], title='Y'),
            zaxis=dict(range=[0, shape[0]], title='Z'),
            aspectratio=dict(
                x=shape[2] / max_dim,
                y=shape[1] / max_dim,
                z=shape[0] / max_dim * z_stretch
            )
        ),
        coloraxis=dict(
            colorscale=colorscale,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title=cbar_label)
        ),  # Define a shared color axis
        title="",
        showlegend=True
    )

    fig.update_scenes(zaxis_autorange="reversed")

    if savename is not None:
        fig.write_html(savename)

    if showfig:
        fig.show()

    return fig


def _ensure_endianness(data):
    return np.ascontiguousarray(data, dtype=data.dtype.newbyteorder('='))