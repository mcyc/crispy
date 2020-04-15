__author__ = 'mcychen'

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from astropy.io import fits
from astropy.wcs import WCS
from skimage import morphology

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


def write_skel(filename, data, header):
    data = data.astype('uint8')
    fits.writeto(filename=filename, data=data, header=header, overwrite=True)


def make_skeleton(coord, refdata, rm_sml_obj = True, coord_in_xfirst=False, start_index=1, min_length = 6):
    # take the coordinates of the SCMS skeleton, and map it onto a map or a cube


    '''
    # if the passed in coordinates are in the order of x, y, and z, instead z, y, and x.
    if coord_in_xfirst:
        #coord[:,0], coord[:,-1] = coord[:,-1], coord[:,0]
        #coord[0,:], coord[-1,:] = coord[-1,:], coord[0,:]
        coord[[0, -1]] = coord[[-1, 0]]

    # round the pixel coordinates into the nearest integer
    if coord.dtype != 'int64':
        coord = np.rint(coord).astype(int)

    coord = coord - start_index
    coord = np.swapaxes(coord, 0, 1)
    coords = tuple(zip(*coord))

    mask = np.zeros(shape=refdata.shape)
    mask[coords] = 1
    '''

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
        print num

        # remove filaments that does not meet the aspect ratio criterium in the pp space
        for i in range(1, num+1):
            mask_i = mask==i
            fil = np.sum(mask_i, axis=0)
            fil = fil.astype('bool')
            fil = morphology.skeletonize(fil)

            if np.sum(fil) < min_length:
                # note: this method may not be able to pick up short spine with lots of branches
                mask[mask_i] = False

        '''
        if False:
            # to ensure the spine is properly connected
            # If I understand correctly, dilation increase the width of the filament by 1 pixel in every direction)
            #mask = morphology.binary_dilation(mask)
            #mask = morphology.skeletonize_3d(mask)
        '''

        # re-label individual branches
        if False:
            mask, num = morphology.label(mask, neighbors=8, return_num=True)
        else:
            mask = mask/mask

        mask = mask.astype('int')

        print num

    return mask


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

        # get index of the point with the median last-coordinate value within a pixel
        # (e.g., in 3D, index of the point with the median z value)
        #z_vals = crd_at_pix[:, 2]
        z_vals = crd_at_pix[:, -1]
        med_idx = np.argsort(z_vals)[len(z_vals) // 2]
        coord_uniq.append(crd_at_pix[med_idx])

    return np.array(coord_uniq).T
