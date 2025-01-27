"""
A place host specialized binary strucutres

"""

import numpy as np

#=====================================================================
# Supports 2D or 3D

def get_base_block(ndim, return_cent_idx=False):
    # ndim is the number of dimensions

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
    # define the space where end points may be considered connected by 8-neighborhood in 2D and 26-neighbourhood in

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

def eight_con():
    return np.ones((3, 3))

# Create 4 to 8-connected elements to use with binary hit-or-miss
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