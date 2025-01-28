import numpy as np
from astropy.io import fits
from astropy.stats import median_absolute_deviation as mads
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def render_point_cloud(cube, savename=None, showfig=False, bins=5, vmin=None, vmax=None,
                       cmap="magma_r", z_stretch=1, fig=None, cbar_label=""):
    """
    Creates a 3D scatter plot of values in a 3D ndarray with coloring based on values and
    opacity determined by percentile bins. A single colorscale spans all bins.

    Parameters
    ----------
    cube : ndarray
        A 3D numpy array of float values. NaN values will be ignored.
    savename : str, optional
        The saving path for the interactive HTML file.
    showfig : bool, optional
        Specify whether to show the figure.
    bins : int, optional
        The number of percentile bins to divide the data into. Default is 5.
    vmin : float, optional
        Minimum value for normalization. If None, the 10th percentile is used.
    vmax : float, optional
        Maximum value for normalization. If None, the 99th percentile is used.
    cmap : str, optional
        Colormap for data points. Default is "magma_r".
    z_stretch : float, optional
        Scaling factor for the Z-axis. Default is 1.
    fig : plotly.graph_objects.Figure, optional
        The Plotly figure object to add the scatter plot to. If None, a new figure will be created.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object representing the 3D scatter plot.
    """
    # Flatten the array and remove NaN values
    valid_values = cube[np.isfinite(cube)].flatten()

    # Determine vmin and vmax if not provided
    if vmin is None:
        vmin = np.percentile(valid_values, 10)
    if vmax is None:
        vmax = np.percentile(valid_values, 99)

    # Mask values outside the range [vmin, vmax]
    mask = np.isfinite(cube) & (cube >= vmin) & (cube <= vmax)
    masked_values = cube[mask]
    z, y, x = np.where(mask)

    # Compute percentiles and opacity levels
    percentiles = np.linspace(0, 100, bins + 1)
    cutoffs = np.percentile(masked_values, percentiles)
    opacity_levels = np.linspace(0.01, 0.3, bins)

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
                size=3,
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


def ridge_trace(x, y, z, size=2, color='darkred', opacity=0.5, name='ridge'):
    # a wrapper to provide a trace to plot ridge in 3D based on its coordinates
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
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


def get_xyz(cube):
    im = cube
    nx, ny, nz = im.shape[2], im.shape[1], im.shape[0]
    Z, Y, X = np.meshgrid(np.arange(0, nz, 1), np.arange(0, ny, 1), np.arange(0, nx, 1), indexing='ij')
    return X, Y, Z


def skel_volume(image, savename=None, showfig=True, opacity=0.7, colorscale='inferno', fig=None):
    # specialized version that only volume renders skeletons and spines

    if isinstance(image, bool):
        image = image.astype(np.uint8)

    return render_volume(image, savename=savename, showfig=showfig, isomin=1e-3, isomax=1e-2, surface_count=1,
                         opacity=opacity, colorscale=colorscale, showscale=False, fig=fig)


def render_volume(cube, savename=None, showfig=False, isomin=None, isomax=None, surface_count=21, opacity=None,
                  colorscale='YlGnBu', showscale=True, fig=None, val_fill=0.0):
    '''
    3D volume rendering using layers of isosurfaces.

    :param cube:
        <ndarray or str> the cube to be volume rendered
    :param savename:
        <str> the saving path for the interactive HTML file
    :param showfig:
        <boolean> specify whether or not to show the figure
    :param isomin:
        <float> the maximum value of the isosurface. Default is the 99.99 percentile of the cube.
    :param isomax:
        <float> the minimum value of the isosurface. Default is the 10-sigma value of the estimated rms level.
    :param surface_count:
        <int> number of isosurfaces to plot. 20 is usually a good number for higher quality visulation. Default is 3
        to make the rendering more effecient.
    :param opacity:
        <float> the opacity of the isosurfaces. This value needs to be small enough to see through all the surfaces.
         The default value is 2 divided by then umber of isosurfaces.
    :param colorscale:
        <str> color map of the isosurfaces, using matplotlib's colormap
    :param fig:
        <plotly's Figure object> the figure for which the 3d scatter will be ploted on. If None, a new figure will be generated.
    :param val_fill:
        <float> The value to replace nan-voxels with. Default is 0. Note that the volume rendering may not work if the cube
         contains nan vlaues

    :return fig:
        a plotly figure object
    '''

    if isinstance(cube, str):
        cube, hdr = fits.getdata(cube, header=True)
        cube = cube.copy()

    # note: the visualization may not work if NaN values are present. Try replacing NaN with values like zeros.
    cube[np.isnan(cube)] = val_fill

    X, Y, Z = get_xyz(cube)

    if fig is None:
        fig = make_subplots(rows=1, cols=1)

    if isomax is None:
        # use 99.99 percentile
        isomax = np.percentile(cube, 99.99)
        print("no isomax provided, using the 99.99 percentile value: {}".format(np.round(isomax,2)))

    if isomin is None:
        # use the estimated 10-sigma
        isomin = mads(cube, ignore_nan=True)*10.0

        if isomin > isomax:
            isomin = isomax/2.0
        print("no isomin provided, using the 10 sigma above the rms value: {}".format(np.round(isomin,2)))

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
        isomin=isomin,
        isomax=isomax,
        opacity=opacity,
        surface_count=surface_count,
        showscale = showscale
    )

    fig.update_scenes(zaxis_autorange="reversed")

    if savename is not None:
        fig.write_html(savename)

    if showfig:
        fig.show()

    return fig