import numpy as np
import astropy.io.fits as fits
from skimage import morphology
from os.path import splitext, isfile

from . import scms as scms_mul
from importlib import reload
reload(scms_mul)


########################################################################################################################

def run(image, h=1, eps=1e-02, maxT=1000, thres=0.135, ordXYZ=True, crdScaling=None, converge_frac=99, ncpu=None,
        walkerThres=None, overmask=None, walkers=None, min_size=9, return_unconverged=True, f_h=5):
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

    :param overmask:
        <boolean ndarray>

    :param walkers:
        <> The coordinates of the walkers to be used. If not None, it superceeds the auotmated placement of walkers

    :param return_unconverged:
        <boolean> Returns both the converged and unconvered walker if True. Else, returns only the converged walkers

    :param: f_h : float, optional
        <float> Factor used to filter out data points based on distance to all other data points (default is 5).

    :return:
        Coordinates of the ridge as defined by the walkers.
    '''

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

    #kwargs = {'eps':eps, 'maxT':maxT, 'weights':weights, 'converge_frac':converge_frac, 'ncpu':ncpu}
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


def append_walkers(coords_1, coords_2):
    # append walkers from coords_2 to coords_1
    return np.append(coords_1, coords_2, axis=0)


def append_suffix(fname, suffix='unconverged'):
    # appended suffix to a given path or filename
    name_root, extension = splitext(fname)
    return "{}_{}{}".format(name_root, suffix, extension)


def write_output(coords, fname, **kwargs):
    # write the SCMS output as a list of coordinates in text file

    def write(coords, savename):
        if coords.ndim !=2:
            coords = coords.reshape(coords.shape[0:2])
        np.savetxt(savename, coords, **kwargs)

    if isinstance(coords, tuple):
        # save unconverged results too if present
        write(coords[0], fname) # converged walkers

        '''
        suffix = "unconverged"
        name_root, extension = splitext(fname)
        write(coords[1], "{}_{}{}".format(name_root, suffix, extension)) # unconverged walkers
        '''
        write(coords[1], append_suffix(fname)) # unconverged walkers

    else:
        write(coords, fname)


def read_output(fname, get_unconverged=True):
    # reads the walker output from crispy. Looks for unconverged file if get_unconverged is true

    def read(fname):
        coords = np.loadtxt(fname, unpack=True)
        return np.expand_dims(coords.T, axis=-1)

    # get name of the unconverged file
    fname_uc = append_suffix(fname)

    if get_unconverged:
        return read(fname), read(fname_uc)

    else:
        return read(fname)


def image2data(image, thres = 0.5, ordXYZ = True, walkerThres=None, overmask=None, min_size=9):
    '''

    :param image:
        <ndarray> the image from which CRISPy runs on

    :param thres:
        <float> the minimal value that a voxel has have to be included in the CRISPy run

    :param ordXYZ:
        <boolean> indicate whether or not the data is ordered in XYZ rather than ZYX. Also work for n-dimensional
         equivalent. If false, the code will assume the indices is in the reverse order

    :param walkerThres:
        <float> the minimal value that a voxel has have to be have a walker placed on it

    :param overmask:
        <boolean ndarray> boolean mask to indicate which voxels to be included in the CRISPy run in addition to the
        thres value criteria

    :return:
    '''
    # convert the input image into the native data format of SCMS
    # i.e., pixel coordinates (X), walker coordinates (G), image weights (weights), number of image dimensions (D)

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
