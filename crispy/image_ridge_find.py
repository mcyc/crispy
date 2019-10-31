import numpy as np
import astropy.io.fits as fits
import scms as scms_mul

########################################################################################################################

def run(fname, h=1, eps=1e-03, maxT=1000, thres=0.135, ordXYZ=True, crdScaling=None, converge_frac=99, ncpu=None,
        walkerThres=None):
    '''
    The wrapper for scmspy_multiproc to identify density ridges in fits images

    :param fname:
        <string> The input fits file name of the image.

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

    image = fits.getdata(fname)
    X, G, weights, D = image2data(image, thres=thres, ordXYZ=ordXYZ, walkerThres=walkerThres)

    if crdScaling is not None:
        crdScaling = np.array(crdScaling)
        X = X[:]/crdScaling[:, None]
        G = G[:]/crdScaling[:, None]

    kwargs = {'eps':eps, 'maxT':maxT, 'wweights':weights, 'converge_frac':converge_frac, 'ncpu':ncpu}
    G = scms_mul.find_ridge(X, G, D, h, 1, **kwargs)

    if crdScaling is not None:
        return G[:]*crdScaling[:, None]
    else:
        return G


def write_output(coords, fname, **kwargs):
    # write the SCMS output as a list of coordinates in text file
    if coords.ndim !=2:
        coords = coords.reshape(coords.shape[0:2])
    np.savetxt(fname, coords, **kwargs)


def image2data(image, thres = 0.5, ordXYZ = True, walkerThres=None):
    # convert the input image into the native data format of SCMS
    # i.e., pixel coordinates (X), walker coordinates (G), image weights (weights), number of image dimensions (D)

    im_shape = image.shape
    D = len(im_shape)
    indices = np.indices(im_shape)


    if walkerThres is None:
        walkerThres = thres * 1.1

    # mask the density field
    mask = image > thres
    Gmask = image > walkerThres

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