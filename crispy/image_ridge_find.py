import numpy as np
import gc
import astropy.io.fits as fits
import scms as scms_mul
from skimage.morphology import binary_dilation, binary_erosion, remove_small_holes, remove_small_objects
from skimage.morphology import disk, skeletonize_3d, ball

########################################################################################################################

def run(image, h=1, eps=1e-03, maxT=1000, thres=0.135, ordXYZ=True, crdScaling=None, converge_frac=99, ncpu=None,
        walkerThres=None, walker_frac=None, walker_mask=None):
    '''
    The wrapper for scmspy_multiproc to identify density ridges in fits images

    :param image:
        <string or ndarray> The input fits file name of the image or the image itself

    :param h:
        <float> The smoothing bandwidth of the Gaussian kernel.

    :param eps:
        <float> Convergence criteria for individual walkers.

    :param maxT:
        <int> The maximum number of iterations allowed for the run.

    :param thres:
        <float> The lower intensity threshold for which the image will be used to find ridges.

    :param ordXYZ:
        <bool> Whether or not to work with coordinates ordered in XYZ rather than the native python ZYX

    :param crdScaling:
        <list> Factors to scale the coordinates by during the run for each axis. The coordinates will be scaled by 1/factor for the
        run and scaled back for the output.

    :param converge_frac:
        <int or float> The criteria to end the run in terms of the percentage of walkers that have converged.
        (Note: occasionally some walkers have

    :param ncpu:
        <int> The number of CPUs to use. If None, defaults to n-1 where n is the number of CPUs available.

    :param walkerThres:
        <float> The lower intensity threshold for where the walker will be placed on the the image.

    :return:
        Coordinates of the ridge as defined by the walkers.
    '''

    if isinstance(image, basestring):
        image = fits.getdata(image)

    X, G, weights, D = image2data(image, thres=thres, ordXYZ=ordXYZ, walkerThres=walkerThres, walker_frac=walker_frac,
                                  walker_mask=walker_mask)

    if crdScaling is not None:
        crdScaling = np.array(crdScaling)
        X = X[slice(None)]/crdScaling[slice(None), None]
        G = G[slice(None)]/crdScaling[slice(None), None]

    kwargs = {'eps':eps, 'maxT':maxT, 'wweights':weights, 'converge_frac':converge_frac, 'ncpu':ncpu}
    G = scms_mul.find_ridge(X, G, D, h, 1, **kwargs)

    if crdScaling is not None:
        return G[slice(None)]*crdScaling[slice(None), None]
    else:
        return G


def write_output(coords, fname, **kwargs):
    # write the SCMS output as a list of coordinates in text file
    if coords.ndim !=2:
        coords = coords.reshape(coords.shape[0:2])
    np.savetxt(fname, coords, **kwargs)


def image2data(image, thres = 0.5, ordXYZ = True, walkerThres=None, walker_frac=None, clean_mask=True, rmSpikes=True,
                                  walker_mask=None):
    # convert the input image into the native data format of SCMS
    # i.e., pixel coordinates (X), walker coordinates (G), image weights (weights), number of image dimensions (D)

    im_shape = image.shape
    D = len(im_shape)
    indices = np.indices(im_shape)

    if ordXYZ:
        # order it in X, Y, Z instead of Z, Y, X
        indices = np.flip(indices, 0)

    if walkerThres is None:
        walkerThres = thres * 1.1

    # mask the density field
    mask = image > thres
    Gmask = image > walkerThres

    if clean_mask:
        print("Polishing the mask to remove noisy features")
        # created a clean mask on the sky
        # note: this is performed because 1 isolated pixel on the sky can still be 'large' in 3D given its spectral size

        if image.ndim == 2:
            mask = clean_mask_2d(mask, disk_r=3)
        else:
            mask_2d_clean = clean_mask_2d(np.any(mask, axis=0), disk_r=3)
            print("original mask size: {}".format(np.sum(mask)))
            mask = clean_mask_3d(mask)
            print("initial clean mask size: {}".format(np.sum(mask)))
            mask = np.logical_and(mask, mask_2d_clean[np.newaxis, slice(None), slice(None)])
            if rmSpikes:
                print("removing spectral spikes: {}".format(np.sum(mask)))
                mask = remove_spec_spikes_3d(mask)
            if True:
                mask = skeletonize_3d(mask)
                mask = binary_dilation(mask, ball(3))
        print("final mask size: {}".format(np.sum(mask)))


    if not walker_frac is None:
        # randomly sample pixels within Gmask to within a fraction specified by the user
        from numpy import random
        print("Placing the walkers randomly  with a filling fraction of :{}".format(walker_frac))
        Z = random.random(image.shape)
        ZMask = Z < walker_frac
        # free some memory
        del Z
        gc.collect()
        Gmask = np.logical_and(Gmask, ZMask)

    # make sure walkers are indeed placed inside the image mask
    Gmask = np.logical_and(mask, Gmask)

    if not walker_mask is None:
        Gmask = np.logical_and(walker_mask, Gmask)

    # get indices of the grid used for KDE
    X = np.array([i[mask] for i in indices])
    X = X[np.newaxis, slice(None)].swapaxes(0, -1)

    # get indices of the test points
    G = np.array([i[Gmask] for i in indices])
    G = G[np.newaxis, slice(None)].swapaxes(0, -1)

    # get masked image
    weights = image[mask]
    gc.collect()
    return X, G, weights, D


#=======================================================================================================================


def clean_mask_2d(mask, disk_r=3):
    # clean 2d thredhold mask
    mask = binary_erosion(mask, disk(disk_r))
    gc.collect()
    mask = remove_small_objects(mask, min_size=25)
    gc.collect()
    mask = binary_dilation(mask, disk(disk_r))
    gc.collect()
    mask = remove_small_holes(mask, area_threshold=25)
    gc.collect()
    return mask


def clean_mask_3d(mask):
    mask = binary_dilation(mask)
    mask = binary_erosion(mask)
    gc.collect()
    return mask


def remove_spec_spikes_3d(mask):
    # assumes the model naturally have noise spikes in them
    spikes = skeletonize_3d(mask)
    return np.logical_and(mask, ~spikes)