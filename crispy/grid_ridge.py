"""
Functions for gridding and processing CRISPy results on reference images.

This module provides tools to map CRISPy ridge detection results onto a reference image grid, clean and refine
the skeleton structures, and handle input and output operations. Key functionalities include skeleton gridding,
labeling, pruning, and advanced 2D/3D structure manipulation for astrophysical data.
"""

__author__ = 'mcychen'

import numpy as np
from astropy.io import fits
from skimage import morphology
from astropy.utils.console import ProgressBar

from .pruning.pruning import endPoints
from .pruning.structures import get_footprints

# ======================================================================================================================#
# higher-level wrapper
def grid_skel(readFile, imgFile, writeFile, **kwargs):
    """
    Map raw CRISPy results onto a reference image grid and save the gridded results.

    Takes ungridded CRISPy results and aligns them with the grid of a
    reference image (typically the image from which CRISPy was run). The gridded results
    are saved to a specified output file in FITS format.

    Parameters
    ----------
    readFile : str
        Path to the .txt file containing the ungridded (raw) CRISPy results.

    imgFile : str
        Path to the .fits file of the reference grid image.

    writeFile : str
        Path to the .fits file where the gridded CRISPy results will be saved.

    **kwargs : dict, optional
        Additional keyword arguments to customize the behavior of the `clean_grid` function.
        Defaults are:

        coord_in_xfirst : bool, default=True
            Whether the input coordinates have the x-dimension first.
        start_index : int, default=0
            Starting index for gridding the skeleton.
        min_length : int, default=6
            Minimum length of structures to retain.
        method : str, default="robust"
            Gridding method, either "robust" or "fast".

    Returns
    -------
    None
        The gridded skeleton is saved directly to the specified `writeFile`.

    Notes
    -----
    - The `clean_grid` function is responsible for handling the gridding and filtering of skeleton structures.
    - The output file is saved in FITS format with the reference image's header metadata.

    Examples
    --------
    Grid CRISPy results and save them to a FITS file:

    >>> from crispy import grid_ridge
    >>> grid_ridge.grid_skel("results.txt", "reference_image.fits", "gridded_results.fits", min_length=10,
    method="fast")

    """
    kwargs_default = dict(coord_in_xfirst=True, start_index=0, min_length=6, method="robust")
    kwargs = {**kwargs_default, **kwargs}

    crds = read_table(readFile)
    img, hdr = fits.getdata(imgFile, header=True)
    skel_cube = clean_grid(crds, img, **kwargs)
    write_skel(writeFile, skel_cube, header=hdr)


# ======================================================================================================================#
# input and output

def read_table(fname, useDict=False):
    """
    Read filament skeleton data from a file.

    Reads the skeleton coordinates identified by the SCMS algorithm
    from a text file. It can return the data as either a NumPy array or a dictionary,
    depending on the user's preference.

    Parameters
    ----------
    fname : str
        Path to the text file containing the skeleton data.

    useDict : bool, optional, default=False
        If True, returns the skeleton data as a dictionary with coordinate labels
        (`'xind, yind, zind'` for 3D or `'xind, yind'` for 2D). Otherwise, returns a NumPy array.

    Returns
    -------
    ndarray or dict
        - If `useDict` is False: A NumPy array containing the skeleton coordinates.
        - If `useDict` is True: A dictionary with labeled coordinates.

    Notes
    -----
    - The skeleton file is expected to have columns representing coordinate indices.
    - If the data is 3D, the dictionary keys will be `'xind, yind, zind'`.
    - For 2D data, the dictionary keys will be `'xind, yind'`.

    Examples
    --------
    Read skeleton data as a NumPy array:

    >>> from crispy import grid_ridge
    >>> data = grid_ridge.read_table("skeleton_data.txt")
    >>> print(data.shape)

    Read skeleton data as a dictionary:

    >>> data_dict = grid_ridge.read_table("skeleton_data.txt", useDict=True)
    >>> print(data_dict.keys())
    """
    values = np.loadtxt(fname, unpack=True)

    if useDict:
        if np.shape(values)[0] == 3:
            keys = ('xind, yind', 'zind')
        else:
            keys = ('xind, yind')
        keys = keys.split(', ')
        return dict(zip(keys, values))

    else:
        return values


def write_skel(filename, data, header):
    """
    Write a gridded image to a FITS file.

    Parameters
    ----------
    filename : str
        The name of the FITS file to write.
    data : ndarray
        The image data to be written to the FITS file.
    header : fits.Header
        The FITS header to include in the file.

    """
    # write gridded skeleton
    data = data.astype('uint8')
    fits.writeto(filename=filename, data=data, header=header, overwrite=True)


# ======================================================================================================================#
# label structures (this is the only place where sklearn is needed in the package


def label_ridge(coord, eps=1.0, min_samples=5):
    """
    Label unconnected ridges using DBSCAN clustering.

    Applies the DBSCAN algorithm to identify and label distinct,
    unconnected ridge structures in the input coordinates.

    Parameters
    ----------
    coord : ndarray
        Coordinates of the ridge points, shape (n, D), where `n` is the number
        of points and `D` is the dimensionality.

    eps : float, optional, default=1.0
        Maximum distance between two points to be considered part of the same ridge.

    min_samples : int, optional, default=5
        Minimum number of points required to form a cluster.

    Returns
    -------
    labels : ndarray
        Cluster labels for each point, shape (n,). Points labeled `-1` are considered noise.

    Notes
    -----
    - DBSCAN is a density-based clustering algorithm that groups points based on spatial
      proximity. Points that do not belong to any cluster are assigned the label `-1`.
    - This function requires `scikit-learn` for the DBSCAN implementation.

    Examples
    --------
    Label ridge points in a 2D space:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> coords = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11]])
    >>> labels = grid_ridge.label_ridge(coords, eps=2.0, min_samples=2)
    >>> print(labels)
    [ 0  0  0  1  1 ]
    """
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coord)
    labels = db.labels_
    return labels


def clean_grid(coord, refdata, coord_in_xfirst=False, start_index=1, min_length=6, method="robust"):
    """
    Process and grid CRISPy coordinates onto a reference image, labeling and cleaning skeleton structures.

    Takes CRISPy coordinates, labels distinct ridge structures using DBSCAN,
    grids them onto a reference image, and removes endpoints that might connect separate structures.
    Structures shorter than a specified length are filtered out.

    Parameters
    ----------
    coord : ndarray
        Coordinates of the ridge points, shape (n, D), where `n` is the number of points
        and `D` is the dimensionality (2D or 3D).

    refdata : ndarray
        Reference image array defining the grid dimensions.

    coord_in_xfirst : bool, optional, default=False
        If True, assumes the input coordinates are ordered with x as the first axis.
        If False, assumes z is the first axis for 3D data or y for 2D data.

    start_index : int, optional, default=1
        The starting index for gridding the skeleton onto the reference image.

    min_length : int, optional, default=6
        Minimum length (in pixels) for structures to be retained.

    method : {"robust", "fast"}, optional, default="robust"
        Method for cleaning skeleton endpoints:
        - "robust": Ensures diagonal connections are handled but is computationally intensive.
        - "fast": Faster but may miss diagonally connected structures.

    Returns
    -------
    skel_cube : ndarray
        Binary array with the same shape as `refdata`, where gridded skeleton structures
        are set to `True`.

    Notes
    -----
    - The DBSCAN algorithm is used to label distinct ridge structures, grouping nearby points
      into clusters and treating outliers as noise.
    - Endpoints are removed to avoid overlap between distinct structures when gridded.
    - The current implementation supports only 2D and 3D data.

    Examples
    --------
    Grid and clean ridge coordinates for a 3D reference image:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> coords = np.array([[0, 0, 0], [1, 1, 1], [10, 10, 10]])
    >>> ref_image = np.zeros((20, 20, 20))
    >>> skel_cube = grid_ridge.clean_grid(coords, ref_image, min_length=5, method="fast")
    >>> print(skel_cube.shape)
    (20, 20, 20)
    """
    # label filaments
    coord = coord.T
    labels = label_ridge(coord, eps=1.0, min_samples=3)

    skel_cube = np.zeros(refdata.shape, dtype=bool)

    if method == "robust":
        # define the space where end points may be considered connected by 8-neighborhood in 2D and 26-neighbourhood in
        footprint = get_footprints(ndim=refdata.ndim, width=5)

    print("---gridding {} distinct skeletons---".format(np.max(labels)))

    for lb in ProgressBar(range(np.max(labels) + 1)):
        # create a full skeleton
        # loop through all the lables (except for -1, which is label for noise)
        skl = grid_skeleton(coord[labels == lb].T, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)
        skl = morphology.skeletonize(skl)
        skl = skl.astype(bool)

        if skl.sum() > min_length:
            # only keep the structure if it has more pixels than the min_length

            omask = np.logical_and(labels != lb, labels >= 0)
            others = grid_skeleton(coord[omask].T, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)
            others = morphology.skeletonize(others)

            # remove the endpoint pixels that may connect one structure from another
            if method == "robust":
                # robust is much less efficient, but ensure the endpoints that diagonally connects distinct structures
                # are removed
                endpts = endPoints(skl)
                try:
                    endpts_lg = morphology.binary_dilation(endpts, footprint=footprint)
                except TypeError:
                    endpts_lg = morphology.binary_dilation(endpts, selem=footprint)

                # find where the structures are connected
                overlap_pt = np.logical_and(endpts_lg, others)
                if np.sum(overlap_pt) > 0:
                    skl[np.logical_and(endpts, endpts_lg)] = False

            elif method == "fast":
                # fast method to remove ends that are "connected to other stuctures when gridded
                # note, may miss diagonally connected structures
                others = morphology.binary_dilation(others)
                skl[np.logical_and(skl, others)] = False

            skel_cube[skl] = True

    return skel_cube


# note: this method designed to work on ppv structures
# more general 3d cleaning has yet to be implemented
def clean_grid_ppv(coord, refdata, coord_in_xfirst=False, start_index=1, min_length=6, method="robust"):
    """
    Process and grid CRISPy coordinates in PPV space onto a reference image, labeling and cleaning skeleton structures.

    Grids CRISPy coordinates onto a position-position-velocity (PPV) reference image, labels distinct
    ridge structures using DBSCAN, and removes endpoints to prevent overlap between structures. Structures shorter
    than a specified projected length are filtered out, and vertical segments are truncated based on a velocity threshold.

    Parameters
    ----------
    coord : ndarray
        Coordinates of the ridge points, shape (n, D), where `n` is the number of points
        and `D` is the dimensionality (typically 3D in PPV space).

    refdata : ndarray
        Reference PPV image array defining the grid dimensions.

    coord_in_xfirst : bool, optional, default=False
        If True, assumes the input coordinates are ordered with x as the first axis.
        If False, assumes z is the first axis.

    start_index : int, optional, default=1
        The starting index for gridding the skeleton onto the reference image.

    min_length : int, optional, default=6
        Minimum projected length (in pixels) for structures to be retained.

    method : {"robust", "fast"}, optional, default="robust"
        Method for cleaning skeleton endpoints:
        - "robust": Handles diagonal connections but is computationally intensive.
        - "fast": Faster but may miss diagonally connected structures.

    Returns
    -------
    skel_cube : ndarray
        Binary array with the same shape as `refdata`, where gridded skeleton structures
        are set to `True`.

    Notes
    -----
    - DBSCAN is used to label ridge structures with higher resultion than the image grid
    - Endpoints are removed to avoid overlap between distinct structures.
    - Vertical segments in velocity are removed based on a threshold (`delVelMax` set to 2 pixels).
    - The current implementation supports only 3D PPV data.

    Examples
    --------
    Grid and clean ridge coordinates in PPV space:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> coords = np.array([[0, 0, 0], [1, 1, 1], [10, 10, 10]])
    >>> ref_image = np.zeros((20, 20, 20))  # Reference PPV image
    >>> skel_cube = grid_ridge.clean_grid_ppv(coords, ref_image, min_length=5, method="robust")
    >>> print(skel_cube.shape)
    (20, 20, 20)
    """
    delVelMax = 2

    # label the filaments
    coord = coord.T
    labels = label_ridge(coord, eps=1.0, min_samples=3)

    skel_cube = np.zeros(refdata.shape, dtype=bool)

    print("---gridding {} distinct skeletons---".format(np.max(labels)))

    for lb in ProgressBar(range(np.max(labels) + 1)):
        # create a full skeleton
        # loop through all the lables (except for -1, which is label for noise)
        skl = grid_skeleton(coord[labels == lb].T, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)
        skl = morphology.skeletonize(skl)
        skl = skl.astype(bool)

        len2d = get_2d_length(skl)
        if len2d > min_length:
            # only keep the structure if it has projected length longer than min_length

            omask = np.logical_and(labels != lb, labels >= 0)
            others = grid_skeleton(coord[omask].T, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)
            others = morphology.skeletonize(others)

            # remove overlaping pixels
            if method == "robust":
                # remove a pixel from the end points
                # robust is much less efficient, but
                endpts = endPoints(skl)
                try:
                    endpts_lg = morphology.binary_dilation(endpts, footprint=morphology.cube(5))
                except TypeError:
                    endpts_lg = morphology.binary_dilation(endpts, selem=morphology.cube(5))

                # find where the structures are connected
                overlap_pt = np.logical_and(endpts_lg, others)
                if np.sum(overlap_pt) > 0:
                    skl[np.logical_and(endpts, endpts_lg)] = False

            elif method == "fast":
                # fast method to remove ends that are too close to other stuctures
                # note, may miss diagonally connected structures
                others = morphology.binary_dilation(others)
                skl[np.logical_and(skl, others)] = False

            # remove vertical segments with delVelMax number of pixels
            # note: if
            skl2d = skl.sum(axis=0)
            skl[:, skl2d > delVelMax] = False

            skel_cube[skl] = True

    # final cleaning to remove small objects
    skel_cube = morphology.remove_small_objects(skel_cube, min_size=min_length, connectivity=2)
    return skel_cube


# ======================================================================================================================#
# grid function

def grid_skeleton(coord, refdata, coord_in_xfirst=False, start_index=1):
    """
    Map CRISPy skeleton coordinates onto a reference image grid.

    Takes CRISPy ridge coordinates and grids them onto a binary mask with the
    same shape as a reference image. The resulting mask highlights the skeletonized structure
    aligned to the grid.

    Parameters
    ----------
    coord : ndarray
        Coordinates of the ridge points, shape (n, D), where `n` is the number of points
        and `D` is the dimensionality (2D or 3D).

    refdata : ndarray
        Reference image array defining the grid dimensions.

    coord_in_xfirst : bool, optional, default=False
        If True, assumes the input coordinates are ordered with x as the first axis.
        If False, assumes z is the first axis for 3D data or y for 2D data.

    start_index : int, optional, default=1
        Starting index for mapping the skeleton coordinates to the reference grid.

    Returns
    -------
    mask : ndarray
        Binary mask with the same shape as `refdata`, where skeletonized points are set to 1
        and all other points are 0.

    Notes
    -----
    - The coordinates are rounded to the nearest integer and adjusted for the starting index
      before mapping onto the reference grid.
    - This function supports both 2D and 3D data.

    Examples
    --------
    Map 3D ridge coordinates onto a reference image grid:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> coords = np.array([[0, 0, 0], [1, 1, 1], [10, 10, 10]])
    >>> ref_image = np.zeros((20, 20, 20))  # Reference image
    >>> mask = grid_ridge.grid_skeleton(coords, ref_image)
    >>> print(mask.shape)
    (20, 20, 20)
    """
    # if the passed in coordinates are in the order of x, y, and z, instead z, y, and x.
    if coord_in_xfirst:
        coord[[0, -1]] = coord[[-1, 0]]

    # round the pixel coordinates into the nearest integer
    if coord.dtype != 'int64':
        coord = np.rint(coord).astype(int)

    coord = coord - start_index
    coord = np.swapaxes(coord, 0, 1)
    coords = tuple(zip(*coord))

    mask = np.zeros(shape=refdata.shape)
    mask[coords] = 1

    return mask


def make_skeleton(coord, refdata, rm_sml_obj=True, coord_in_xfirst=False, start_index=1, min_length=6):
    """
    Map CRISPy skeleton coordinates onto a reference grid and clean the skeleton.

    Grids CRISPy ridge coordinates onto a binary mask based on a reference
    image. It optionally removes small objects and structures shorter than a specified
    length to produce a cleaned skeleton map.

    Parameters
    ----------
    coord : ndarray
        Coordinates of the ridge points, shape (n, D), where `n` is the number of points
        and `D` is the dimensionality (2D or 3D).

    refdata : ndarray
        Reference image array defining the grid dimensions.

    rm_sml_obj : bool, optional, default=True
        If True, removes small objects shorter than `min_length` from the skeletonized map.

    coord_in_xfirst : bool, optional, default=False
        If True, assumes the input coordinates are ordered with x as the first axis.
        If False, assumes z is the first axis for 3D data or y for 2D data.

    start_index : int, optional, default=1
        Starting index for mapping the skeleton coordinates to the reference grid.

    min_length : int, optional, default=6
        Minimum length (in pixels) for structures to be retained.

    Returns
    -------
    mask : ndarray
        Binary array with the same shape as `refdata`, representing the cleaned skeleton.
        Structures shorter than `min_length` are removed if `rm_sml_obj` is True.

    Notes
    -----
    - The skeleton is gridded using the `grid_skeleton` function and further processed
      to remove small objects or short structures.
    - Cleaning operations assume the skeleton is 1-pixel wide and connected by an 8-neighbor
      connectivity in 2D or 26-neighbor connectivity in 3D.

    Examples
    --------
    Create and clean a 3D skeleton:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> coords = np.array([[0, 0, 0], [1, 1, 1], [10, 10, 10]])
    >>> ref_image = np.zeros((20, 20, 20))  # Reference image
    >>> mask = grid_ridge.make_skeleton(coords, ref_image, rm_sml_obj=True, min_length=5)
    >>> print(mask.shape)
    (20, 20, 20)
    """

    mask = grid_skeleton(coord, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)

    # remove small object shorter than a certain length (this assumes the skeleton is truely 1-pixel in width)
    if rm_sml_obj:
        mask = mask.astype('bool')
        # connectivity = 2 to ensure "vortex/diagonal" connection
        mask = morphology.remove_small_objects(mask, min_size=min_length, connectivity=2)

        # label each connected structure
        # Whether to use 4- or 8- "connectivity". In 3D, 4-"connectivity" means connected pixels have to share face,
        # whereas with 8-"connectivity", they have to share only edge or vertex.
        mask, num = morphology.label(mask, connectivity=2, return_num=True)

        # remove filaments that does not meet the aspect ratio criterium in the pp space
        for i in range(1, num + 1):
            mask_i = mask == i
            fil = np.sum(mask_i, axis=0)
            fil = fil.astype('bool')
            fil = morphology.skeletonize(fil)

            if np.sum(fil) < min_length:
                # note: this method may not be able to pick up short spine with lots of branches
                mask[mask_i] = False

        # re-label individual branches
        if False:
            mask, num = morphology.label(mask, connectivity=2, return_num=True)
        else:
            mask = mask / mask

        mask = mask.astype('int')

    return mask


def get_2d_length(skl3d):
    """
    Calculate the sky-projected length of a 3D skeleton.

    Computes the length of a skeleton structure when projected onto a 2D plane.
    The projection is performed by collapsing the third dimension of the input 3D skeleton array.

    Parameters
    ----------
    skl3d : ndarray
        3D binary array representing the skeletonized structure, where `True` or `1`
        indicates skeleton points and `False` or `0` represents the background.

    Returns
    -------
    length : int
        The total number of pixels in the projected 2D skeleton.

    Notes
    -----
    - The function collapses the 3D skeleton along the third axis using a logical OR operation
      and then applies 2D skeletonization to the resulting binary image.
    - This method is useful for evaluating the extent of structures in position-position space
      regardless of the velocity axis.

    Examples
    --------
    Compute the 2D length of a 3D skeleton:

    >>> import numpy as np
    >>> from skimage.morphology import skeletonize
    >>> from crispy import grid_ridge
    >>> skl3d = np.zeros((10, 10, 10), dtype=bool)
    >>> skl3d[0, 0, :] = True  # A straight skeleton in 3D
    >>> length = grid_ridge.get_2d_length(skl3d)
    >>> print(length)
    1
    """
    skl = np.any(skl3d, axis=0)
    skl = skl.astype('bool')
    skl = morphology.skeletonize(skl)
    return np.sum(skl)


def uniq_per_pix(coord, mask, coord_in_xfirst=False, start_index=1):
    """
    Reduce a list of ridge coordinates to one unique point per pixel.

    Processes ridge coordinates to retain a single representative point
    per pixel based on a provided binary mask. The representative point is selected
    as the one with the median value along the last coordinate axis.

    Parameters
    ----------
    coord : ndarray
        Ridge coordinates, shape (D, n), where `D` is the number of dimensions
        (e.g., 2 or 3) and `n` is the number of points.

    mask : ndarray
        Binary mask array, shape matching the reference grid, where `True` indicates
        pixels of interest.

    coord_in_xfirst : bool, optional, default=False
        If True, assumes the input coordinates are ordered with x as the first axis.
        If False, assumes z is the first axis for 3D data or y for 2D data.

    start_index : int, optional, default=1
        Starting index for the coordinate system. Adjusts the input coordinates before
        processing.

    Returns
    -------
    coord_uniq : ndarray
        Reduced set of coordinates, shape (D, m), where `m` is the number of unique
        pixels with a representative coordinate.

    Notes
    -----
    - This function is optimized for cases where the input mask represents gridded,
      one-voxel-wide skeletons or spines.
    - The median value along the last axis (e.g., z in 3D) is used to select the
      representative point for each pixel.

    Examples
    --------
    Reduce ridge coordinates to one per pixel:

    >>> import numpy as np
    >>> from crispy import grid_ridge
    >>> coords = np.array([[0, 1, 1, 2], [0, 0, 0, 0], [0, 0, 1, 1]])  # 3D coordinates
    >>> mask = np.zeros((3, 3, 3), dtype=bool)
    >>> mask[1, 1, 0] = True
    >>> mask[1, 1, 1] = True
    >>> reduced_coords = grid_ridge.uniq_per_pix(coords, mask)
    >>> print(reduced_coords)
    [[1]
     [1]
     [0]]
    """
    # if the passed in coordinates are in the order of x, y, and z, instead z, y, and x.
    if coord_in_xfirst:
        coord[[0, -1]] = coord[[-1, 0]]

    # round the pixel coordinates into the nearest integer
    if coord.dtype != 'int64':
        crds_int = np.rint(coord).astype(int)
    else:
        msg = (f"The provided coord are of type {coord.dtype} instead of the supported int")
        raise ValueError(msg)

    crds_int = crds_int - start_index
    crds_int = np.swapaxes(crds_int, 0, 1)
    crds_int = tuple(zip(*crds_int))

    # get indicies of where the mask is true
    idx_mask = np.argwhere(mask)

    # get the coordinate index in the smae format as the mask indicies
    crds_int = np.array(crds_int).T
    coord = coord.T

    coord_uniq = []

    for i, idx in enumerate(idx_mask):
        mask_same = np.all(crds_int - idx_mask[i] == 0, axis=1)
        crd_at_pix = coord[mask_same]
        if crd_at_pix.size == 0:
            print("[ERROR]: crd at pix size: {}; there may be a mismatch in the start_index".format(crd_at_pix.size))

        # get index of the point with the median last-coordinate value within a pixel
        # (e.g., in 3D, index of the point with the median z value)
        z_vals = crd_at_pix[:, -1]
        med_idx = np.argsort(z_vals)[len(z_vals) // 2]
        coord_uniq.append(crd_at_pix[med_idx])

    return np.array(coord_uniq).T
