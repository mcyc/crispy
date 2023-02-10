__author__ = 'mcychen'

import numpy as np
from scipy import ndimage
import copy
from astropy.io import fits
from skimage import morphology
from astropy.utils.console import ProgressBar


#======================================================================================================================#
# input and output

def read_table(fname, useDict = False):
    '''
    read in the filaments skeleton identified by SCMC algorithm
    :param fname:
        (String) File name of the filament skeleton
    :param useDict:
        (Boolean) Whether or not to return the read in values as a dictionary. The keys assigned assumes that the
        coordinates eitehr have 2 or 3 dimensions
    :return:
    '''

    values = np.loadtxt(fname,unpack=True)

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
    # write gridded skeleton
    data = data.astype('uint8')
    fits.writeto(filename=filename, data=data, header=header, overwrite=True)


#======================================================================================================================#
# label structures (this is the only place where sklearn is needed in the package


def label_ridge(coord, eps=1.0, min_samples=5):
    # use DBSCAN to label different, unconnected ridges
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coord)
    labels = db.labels_
    return labels



def clean_grid(coord, refdata, coord_in_xfirst=False, start_index=1, min_length = 6, method="robust"):
    # take CRISPY coordinates, label individual filaments, and grid them with one pixel taken off from their ends
    # DB scan is used to avoid sub-sampling with the cube grid
    # note: the current implementation only works for 2D or 3D

    # label the filaments
    coord = coord.T
    labels = label_ridge(coord, eps=1.0, min_samples=3)

    skel_cube = np.zeros(refdata.shape, dtype=bool)

    if method == "robust":
        # define the space where end points may be considered connected by 8-neighborhood in 2D and 26-neighbourhood in
        # 3D
        if refdata.ndim == 3:
            selem = morphology.cube(5)
        elif refdata.ndim == 2:
            selem = morphology.square(5)
        else:
            print("[ERROR] the dimension of the refdata (i.e., {}) is invalid".format(refdata.ndim))

    print("---gridding {} distinct skeletons---".format(np.max(labels)))

    for lb in ProgressBar(range(np.max(labels) + 1)):
    #for lb in range(np.max(labels) + 1):
        # create a full skeleton
        # loop through all the lables (except for -1, which is label for noise)
        skl = grid_skeleton(coord[labels == lb].T, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)
        skl = morphology.skeletonize_3d(skl)
        skl = skl.astype(bool)

        if skl.sum() > min_length:
            # only keep the structure if it has more pixels than the min_length

            omask = np.logical_and(labels != lb, labels >= 0)
            others = grid_skeleton(coord[omask].T, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)
            others = morphology.skeletonize_3d(others)

            # remove the endpoint pixels that may connect one structure from another
            if method == "robust":
                # robust is much less efficient, but ensure the endpoints that diagonally connects distinct structures
                # are removed
                endpts = endPoints(skl)
                endpts_lg =  morphology.binary_dilation(endpts, selem=selem)

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
def clean_grid_ppv(coord, refdata, coord_in_xfirst=False, start_index=1, min_length = 6, method="robust"):
    # take CRISPY coordinates, label individual filaments, and grid them with one pixel taken off from their ends
    # DB scan is used to avoid sub-sampling with the cube grid
    delVelMax = 2

    # label the filaments
    coord = coord.T
    labels = label_ridge(coord, eps=1.0, min_samples=3)

    skel_cube = np.zeros(refdata.shape, dtype=bool)

    print("---gridding {} distinct skeletons---".format(np.max(labels)))

    for lb in ProgressBar(range(np.max(labels) + 1)):
    #for lb in range(np.max(labels) + 1):
        # create a full skeleton
        # loop through all the lables (except for -1, which is label for noise)
        skl = grid_skeleton(coord[labels == lb].T, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)
        skl = morphology.skeletonize_3d(skl)
        skl = skl.astype(bool)

        len2d = get_2d_length(skl)
        if len2d > min_length:
            # only keep the structure if it has projected length longer than min_length

            omask = np.logical_and(labels != lb, labels >= 0)
            others = grid_skeleton(coord[omask].T, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)
            others = morphology.skeletonize_3d(others)

            # remove overlaping pixels
            if method == "robust":
                # remove a pixel from the end points
                # robust is much less efficient, but
                endpts = endPoints(skl)
                endpts_lg =  morphology.binary_dilation(endpts, selem=morphology.cube(5))

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
    skel_cube = morphology.remove_small_objects(skel_cube, min_size=min_length, connectivity=2, in_place=False)
    return skel_cube


#======================================================================================================================#
# grid function

def grid_skeleton(coord, refdata, coord_in_xfirst=False, start_index=1):
    # take the coordinates of the SCMS skeleton, and map it onto a map or a cube

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



def make_skeleton(coord, refdata, rm_sml_obj = True, coord_in_xfirst=False, start_index=1, min_length = 6):
    # take the coordinates of the SCMS skeleton, and map it onto a map or a cube

    mask = grid_skeleton(coord, refdata, coord_in_xfirst=coord_in_xfirst, start_index=start_index)

    # remove small object shorter than a certain length (this assumes the skeleton is truely 1-pixel in width)
    if rm_sml_obj:

        #min_length = 6 #9 # 3 binwidth

        mask = mask.astype('bool')
        # connectivity = 2 to ensure "vortex/diagonal" connection
        mask = morphology.remove_small_objects(mask, min_size=min_length, connectivity=2, in_place=False)

        # label each connected structure
        # Whether to use 4- or 8- "connectivity". In 3D, 4-"connectivity" means connected pixels have to share face,
        # whereas with 8-"connectivity", they have to share only edge or vertex.
        mask, num = morphology.label(mask, neighbors=8, return_num=True)
        print(num)

        # remove filaments that does not meet the aspect ratio criterium in the pp space
        for i in range(1, num+1):
            mask_i = mask==i
            fil = np.sum(mask_i, axis=0)
            fil = fil.astype('bool')
            fil = morphology.skeletonize(fil)

            if np.sum(fil) < min_length:
                # note: this method may not be able to pick up short spine with lots of branches
                mask[mask_i] = False

        # re-label individual branches
        if False:
            mask, num = morphology.label(mask, neighbors=8, return_num=True)
        else:
            mask = mask/mask

        mask = mask.astype('int')
        print(num)

    return mask


def get_2d_length(skl3d):
    # return length of the sky-projected filament
    skl = np.any(skl3d, axis=0)
    skl = skl.astype('bool')
    skl = morphology.skeletonize(skl)
    return np.sum(skl)


def uniq_per_pix(coord, mask, coord_in_xfirst=False, start_index=1):
    # take a list of ridge coordinates and return
    # the method is most efficient if the mask consists of gridded, one voxel wide spines/skeletons

    # if the passed in coordinates are in the order of x, y, and z, instead z, y, and x.
    if coord_in_xfirst:
        coord[[0, -1]] = coord[[-1, 0]]

    # round the pixel coordinates into the nearest integer
    if coord.dtype != 'int64':
        crds_int = np.rint(coord).astype(int)
    else:
        print("[ERROR]: the provided coord are of the type {} instead of intergers").format(coord.dtype)
        return None

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
        #z_vals = crd_at_pix[:, 2]
        z_vals = crd_at_pix[:, -1]
        med_idx = np.argsort(z_vals)[len(z_vals) // 2]
        coord_uniq.append(crd_at_pix[med_idx])

    return np.array(coord_uniq).T



#======================================================================================================================#
# gridded structures

def branchedPoints(skel, endpt = None):
    # note: it's more efficient to find the body-points than to find branch-points
    pt = bodyPoints(skel)
    pt = np.logical_and(skel, np.logical_not(pt))

    # if no end-points are defined, find the end-points first and remove them
    if endpt is None:
        print("calculating end points...")
        endpt = endPoints(skel)

    pt = np.logical_and(pt, np.logical_not(endpt))

    return pt


# identify body points (points with only two neighbour by 8-connectivity)
def bodyPoints(skel):

    # for 2D skeleton
    if np.size(skel.shape) == 2:
        base_block = np.zeros((3,3))
        base_block[1,1] = 1

    # for 3D skeleton
    elif np.size(skel.shape) == 3:
        base_block = np.zeros((3,3,3))
        base_block[1,1,1] = 1

    else:
        print("[ERROR] the skeleton is neither 2 or 3 dimensions in size!")
        return None

    ptList = []

    # iterate over the "top" layer
    i = 0
    for idx_top, v_top in np.ndenumerate(base_block[0]):
        # for each cell in the "top" layer, iterate over teh "bottom" layer
        for idx_bottom, v_bottom in np.ndenumerate(base_block[2]):
            str_block = base_block.copy()
            # populate the two neighbours
            str_block[(0,) + idx_top] = 1
            str_block[(2,) + idx_bottom] = 1
            ptList.append(str_block)

    # now add the permutations that are rotationally symmetric to the above list
    ptListOri = copy.deepcopy(ptList)
    for i in ptListOri:
        ptList.append(np.swapaxes(i, 0, 1))

    # again, for a 3D skeleton
    if np.size(skel.shape) == 3:
        for i in ptListOri:
            ptList.append(np.swapaxes(i, 0, 2))

    # remove the redundant elements
    ptList = np.unique(np.array(ptList), axis = 0)

    pt = np.full(np.shape(skel), False, dtype=bool)

    for pt_i in ptList:
        pt = pt + ndimage.binary_hit_or_miss(skel, structure1=pt_i)

    return pt


# identify end points (points with only two neighbour by 8-connectivity)
# (only works if the skeleton is on 1-pixel in width by 8-connectivity and not 4-connectivity)
def endPoints(skel):

    # for 2D skeleton
    if np.size(skel.shape) == 2:
        base_block = np.zeros((3,3))
        base_block[1,1] = 1
        cent_idx = (1,1)

    # for 3D skeleton
    elif np.size(skel.shape) == 3:
        base_block = np.zeros((3,3,3))
        base_block[1,1,1] = 1
        cent_idx = (1,1,1)

    else:
        print("[ERROR] the skeleton is neither 2 or 3 dimensions in size!")
        return None

    epList = []

    # iterate over all permutation of endpoints
    # Note: this does not account for "end points" that are only a pixel long
    for index, value in np.ndenumerate(base_block):
        if index != cent_idx:
            str_block = base_block.copy()
            str_block[index] = 1
            epList.append(str_block)

    ep = np.full(np.shape(skel), False, dtype=bool)

    for ep_i in epList:
        ep = ep + ndimage.binary_hit_or_miss(skel, structure1=ep_i)

    return ep