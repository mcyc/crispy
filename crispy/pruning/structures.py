"""
Provides specialized binary structures and connectivity footprints for 2D and 3D skeleton processing.
"""

import numpy as np

#=====================================================================
# Supports 2D or 3D

def get_base_block(ndim, return_cent_idx=False):
    """
    Generate a base block array for 2D or 3D skeleton structures.

    Parameters
    ----------
    ndim : int
        The number of dimensions (2 or 3).
    return_cent_idx : bool, optional
        If True, also returns the central index of the block. Defaults to False.

    Returns
    -------
    base_block : ndarray
        A binary array with a single central pixel set to 1.
    cent_idx : tuple, optional
        The central index of the block, returned if `return_cent_idx` is True.
    """
    if ndim == 2:
        base_block = np.zeros((3,3))
        base_block[1,1] = 1
        cent_idx = (1,1)

    elif ndim == 3:
        base_block = np.zeros((3, 3, 3))
        base_block[1, 1, 1] = 1
        cent_idx = (1, 1, 1)

    else:
        msg = f"The provided ndim, {ndim}, is not supported. Only 2 or 3 dimensions are accepted."
        raise ValueError(msg)

    if return_cent_idx:
        return base_block, cent_idx

    else:
        return base_block


def get_footprints(ndim, width=5, dtype=np.uint8):
    """
    Generate a footprint array representing connectivity in 2D or 3D space.

    Parameters
    ----------
    ndim : int
        The number of dimensions (2 or 3).
    width : int, optional
        The size of the footprint array along each dimension. Defaults to 5.
    dtype : data-type, optional
        The desired data type of the footprint array. Defaults to np.uint8.

    Returns
    -------
    footprint : ndarray
        A binary array of ones representing the connectivity footprint.
    """
    if ndim == 3:
        #footprint = morphology.cube(5)
        shape = (width, width, width)

    elif ndim == 2:
        #footprint = morphology.square(5)
        shape = (width, width)
    else:
        msg = f"The provided ndim, {ndim}, is not supported. Only 2 or 3 dimensions are accepted."
        raise ValueError(msg)

    return np.ones(shape, dtype=dtype)


#=====================================================================
# 2D structures

two_con = np.ones((3, 3), dtype=np.uint8)

# Create 1 to 2-connected elements to use with binary hit-or-miss
struct1 = np.array([[1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 0]])

struct2 = np.array([[0, 0, 1],
                    [1, 1, 0],
                    [0, 0, 0]])

# Next check the three elements which will be double counted
check1 = np.array([[1, 1, 0, 0],
                   [0, 0, 1, 1]])

check2 = np.array([[0, 0, 1, 1],
                   [1, 1, 0, 0]])

check3 = np.array([[1, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1]])


#=====================================================================
# 3D structures

two_con_3D = np.ones((3, 3, 3), dtype=np.uint8)