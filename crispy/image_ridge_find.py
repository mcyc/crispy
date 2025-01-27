"""
Functions for density ridge identification in gridded images.

This module provides tools to process images, apply the SCMS algorithm, and identify density ridges
from FITS or NumPy array-based data. It includes utilities for walker initialization, ridge finding,
and output handling, enabling seamless integration into image analysis workflows.
"""

import numpy as np
import astropy.io.fits as fits
from skimage import morphology
from os.path import splitext

from . import scms as scms_mul

########################################################################################################################

def run(image, h=1, eps=1e-2, maxT=1000, thres=0.135, ordXYZ=True, crdScaling=None, converge_frac=99, ncpu=None,
        walkerThres=None, overmask=None, walkers=None, min_size=9, return_unconverged=True, f_h=5):
    """
    Identify density ridges in a gridded image using the SCMS algorithm.

    This function serves as a wrapper for the SCMS algorithm to identify density ridges
    in FITS images or NumPy arrays. It supports configurable smoothing, convergence criteria,
    and walker initialization for ridge identification.

    Parameters
    ----------
    image : str or ndarray
        Input image as a FITS file path or a NumPy array.

    h : float, optional, default=1
        Smoothing bandwidth of the Gaussian kernel.

    eps : float, optional, default=1e-2
        Convergence criteria for individual walkers. A smaller value increases precision.

    maxT : int, optional, default=1000
        Maximum number of iterations allowed for the walkers to converge.

    thres : float, optional, default=0.135
        Lower intensity threshold. Pixels below this threshold are ignored.

    ordXYZ : bool, optional, default=True
        If True, reorders coordinates to XYZ format (native Python uses ZYX).

    crdScaling : list of float, optional
        Scaling factors for each axis. Coordinates are scaled during processing and
        rescaled in the output.

    converge_frac : float, optional, default=99
        Fraction of walkers that need to converge for the process to terminate, expressed as a percentage.

    ncpu : int, optional
        Number of CPUs to use. If None, defaults to one less than the total number of CPUs.

    walkerThres : float, optional
        Lower intensity threshold for placing walkers. Overrides the automated walker placement
        if provided.

    overmask : ndarray of bool, optional
        Boolean mask specifying which pixels to include in ridge finding, in addition to the `thres` criteria.

    walkers : ndarray, optional
        Custom initial positions for walkers. Overrides automated walker placement if provided.

    min_size : int, optional, default=9
        Minimum size (in pixels) for structures to be considered. Smaller structures are removed.

    return_unconverged : bool, optional, default=True
        If True, includes both converged and unconverged walkers in the output. Otherwise, only converged walkers are returned.

    f_h : float, optional, default=5
        Filtering factor used to exclude data points based on their distance to all other points.

    Returns
    -------
    ndarray or tuple of ndarray
        Coordinates of the density ridges. If `return_unconverged` is True, returns a tuple:
        (converged walkers, unconverged walkers). Otherwise, only the converged walkers are returned.

    Notes
    -----
    - The image can be a 2D or 3D array. For 3D images, smoothing and walker placement apply
      to all axes equally unless `crdScaling` is provided.
    - If `crdScaling` is used, the coordinates are scaled back to their original values in the output.
    - This function is designed to work efficiently with gridded images and leverages multiprocessing.

    Examples
    --------
    Process a FITS image and retrieve density ridges:

    >>> from crispy import image_ridge_find as irf
    >>> ridges = irf.run("input_image.fits", h=2, thres=0.2, maxT=500)

    Use custom walkers for initialization:

    >>> walkers = np.array([[10, 20], [15, 25], [20, 30]])
    >>> ridges = irf.run("input_image.fits", walkers=walkers, maxT=200)

    """
    if isinstance(image, str):
        image = fits.getdata(image)

    X, G, weights, D = image2data(image, thres=thres, ordXYZ=ordXYZ, walkerThres=walkerThres, overmask=overmask,
                                  min_size=min_size)

    if walkers is not None:
        print("Using user provided walkers")
        G = walkers

    if crdScaling is not None:
        crdScaling = np.array(crdScaling)
        X = X[:]/crdScaling[:, None]
        G = G[:]/crdScaling[:, None]

    kwargs = dict(eps=eps, maxT=maxT, weights=weights, converge_frac=converge_frac, ncpu=ncpu,
                  return_unconverged=return_unconverged, f_h=f_h)
    G = scms_mul.find_ridge(X, G, D, h, 1, **kwargs)

    def scale_back(G):
        return G[:] * crdScaling[:, None]

    if crdScaling is not None:
        if isinstance(G, tuple):
            # if unconverged walkers are returned
            print("return all")
            return scale_back(G[0]), scale_back(G[1])
        else:
            # return G[:]*crdScaling[:, None]
            return scale_back(G)
    else:
        return G


def _append_walkers(coords_1, coords_2):
    # append walkers from coords_2 to coords_1
    return np.append(coords_1, coords_2, axis=0)


def _append_suffix(fname, suffix='unconverged'):
    # appended suffix to a given path or filename
    name_root, extension = splitext(fname)
    return "{}_{}{}".format(name_root, suffix, extension)


def write_output(coords, fname, **kwargs):
    """
    Write SCMS output coordinates to a file.

    This function saves the SCMS ridge-finding results as a text file.
    If unconverged walkers are included, they are saved to a separate file
    with an appended suffix.

    Parameters
    ----------
    coords : ndarray or tuple of ndarray
        Coordinates of the walkers to save. If a tuple is provided, it should
        contain:
        - `coords[0]`: Converged walker coordinates.
        - `coords[1]`: Unconverged walker coordinates.

    fname : str
        File name or path to save the output. The file extension determines
        the format (e.g., `.txt`).

    **kwargs : dict, optional
        Additional arguments passed to `numpy.savetxt`, such as formatting
        options (`fmt`) or delimiter (`delimiter`).

    Notes
    -----
    - If `coords` contains both converged and unconverged walkers (as a tuple),
      the unconverged coordinates are saved in a separate file with a suffix
      (e.g., `_unconverged`) appended to `fname`.
    - The default behavior uses `numpy.savetxt` for saving coordinates, allowing
      for flexible formatting via `**kwargs`.

    Examples
    --------
    Save only converged walkers:

    >>> import numpy as np
    >>> from crispy import image_ridge_find as irf
    >>> coords = np.array([[1, 2, 3], [4, 5, 6]])
    >>> irf.write_output(coords, "output.txt")

    Save both converged and unconverged walkers:

    >>> coords_converged = np.array([[1, 2, 3], [4, 5, 6]])
    >>> coords_unconverged = np.array([[7, 8, 9]])
    >>> irf.write_output((coords_converged, coords_unconverged), "output.txt", fmt="%.3f")
    """
    def write(coords, savename):
        if coords.ndim !=2:
            coords = coords.reshape(coords.shape[0:2])
        np.savetxt(savename, coords, **kwargs)

    if isinstance(coords, tuple):
        # save unconverged results too if present
        write(coords[0], fname) # converged walkers

        write(coords[1], _append_suffix(fname)) # unconverged walkers

    else:
        write(coords, fname)


def read_output(fname, get_unconverged=True):
    """
    Read SCMS output files and retrieve walker coordinates.

    This function reads the results of the SCMS ridge-finding process from a file.
    It can optionally include unconverged walker coordinates if available.

    Parameters
    ----------
    fname : str
        Path to the main output file containing converged walker coordinates.

    get_unconverged : bool, optional, default=True
        If True, attempts to read the file containing unconverged walker coordinates
        alongside the converged data.

    Returns
    -------
    coords : ndarray
        Coordinates of converged walkers from the main output file.

    coords_unconverged : ndarray, optional
        Coordinates of unconverged walkers if `get_unconverged` is True and the
        corresponding file exists.

    Notes
    -----
    - The file name for unconverged walkers is derived by appending a suffix
      (e.g., `_unconverged`) to the main file name.
    - If the unconverged file does not exist and `get_unconverged` is True, an
      exception may be raised depending on the file handling logic.

    Examples
    --------
    Read only the converged walkers:

    >>> from crispy import image_ridge_find as irf
    >>> coords = irf.read_output("output.txt", get_unconverged=False)

    Read both converged and unconverged walkers:

    >>> coords, coords_unconverged = irf.read_output("output.txt", get_unconverged=True)
    """
    def read(fname):
        coords = np.loadtxt(fname, unpack=True)
        return np.expand_dims(coords.T, axis=-1)

    # get name of the unconverged file
    fname_uc = _append_suffix(fname)

    if get_unconverged:
        return read(fname), read(fname_uc)

    else:
        return read(fname)


def image2data(image, thres=0.5, ordXYZ=True, walkerThres=None, overmask=None, min_size=9):
    """
    Convert an image into a format compatible with the SCMS algorithm.

    This function processes the input image to generate data structures required for
    ridge finding using the SCMS algorithm. It filters the image based on intensity
    thresholds and optionally removes small structures.

    Parameters
    ----------
    image : ndarray
        Input image as a NumPy array.

    thres : float, optional, default=0.5
        Minimum intensity value for a pixel to be included in the SCMS run.
        Pixels below this threshold are ignored.

    ordXYZ : bool, optional, default=True
        If True, reorders coordinates to XYZ format (native Python uses ZYX).
        Supports n-dimensional equivalents for higher-dimensional data.

    walkerThres : float, optional
        Minimum intensity value for placing walkers. Defaults to 1.1 * `thres`
        if not specified.

    overmask : ndarray of bool, optional
        Boolean mask specifying additional voxels to include in the SCMS run,
        supplementing the threshold-based mask.

    min_size : int, optional, default=9
        Minimum size (in pixels) for connected structures to be retained.
        Smaller structures are removed for noise suppression.

    Returns
    -------
    X : ndarray
        Pixel coordinates of the input image that meet the intensity threshold,
        formatted for kernel density estimation.

    G : ndarray
        Initial walker positions for the SCMS algorithm, derived from the
        threshold mask or `walkerThres`.

    weights : ndarray
        Intensity values of the pixels in `X`, representing their weights.

    D : int
        Dimensionality of the input image (e.g., 2D or 3D).

    Notes
    -----
    - The function automatically handles small structure removal when `min_size` is set.
    - If no `walkerThres` is provided, it defaults to slightly above `thres` (1.1 * `thres`).
    - The `ordXYZ` parameter ensures compatibility with SCMS expectations, which use XYZ ordering.

    Examples
    --------
    Prepare image data for ridge finding:

    >>> import numpy as np
    >>> from crispy import image_ridge_find as irf
    >>> image = np.random.random((100, 100))
    >>> X, G, weights, D = irf.image2data(image, thres=0.2, min_size=5)

    Use a custom mask to include specific voxels:

    >>> mask = np.zeros_like(image, dtype=bool)
    >>> mask[10:20, 10:20] = True
    >>> X, G, weights, D = irf.image2data(image, thres=0.2, overmask=mask)
    """
    im_shape = image.shape
    D = len(im_shape)
    indices = np.indices(im_shape)


    if walkerThres is None:
        walkerThres = thres * 1.1

    if overmask is None:
        overmask = np.isfinite(image)

    # mask the density field
    mask = image > thres
    Gmask = image > walkerThres

    mask = np.logical_and(mask, overmask)
    Gmask = np.logical_and(Gmask, overmask)

    if min_size is not None:
        # remove structures with sizes less than min_size number of pixels
        mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=1)
        # ensure the walker is placed only over the masked image
        Gmask = np.logical_and(Gmask, mask)


    if ordXYZ:
        # order it in X, Y, Z instead of Z, Y, X
        indices = np.flip(indices, 0)

    # get indices of the grid used for KDE
    X = np.array([i[mask] for i in indices])
    X = X[np.newaxis, :].swapaxes(0, -1)

    # get indices of the test points
    G = np.array([i[Gmask] for i in indices])
    G = G[np.newaxis, :].swapaxes(0, -1)

    # get masked image
    weights = image[mask]
    return X, G, weights, D


def threshold_local(image, *args, **kwargs):
    """
    Apply a local thresholding method to an image for binarization.

    Parameters
    ----------
    image : ndarray
        Input image to be thresholded.
    *args : tuple
        Positional arguments passed to the thresholding function.
    **kwargs : dict
        Keyword arguments passed to the thresholding function.

    Returns
    -------
    mask : ndarray
        Binary mask of the same shape as `image`, where True values represent
        pixels above the local threshold.
    """
    try:
        from skimage.filters import threshold_local
        mask = image > threshold_local(image, *args, **kwargs)
    except ImportError:
        # for older versions of
        from skimage.filters import threshold_adaptive
        mask = threshold_adaptive(image, *args, **kwargs)

    return mask