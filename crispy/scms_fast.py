import numpy as np
import time
from scipy.stats import multivariate_normal
from multiprocessing import Pool, cpu_count
from itertools import repeat

from scipy.stats import gaussian_kde

def find_ridge(X, G, D=3, h=1, d=1, eps = 1e-06, maxT = 1000, wweights = None, converge_frac = 99):

    G = G.astype('float')
    X = X.astype('float')
    n = len(X)
    m = len(G)  # x and y coordinates 2xN format
    print("n, m: {0}, {1}".format(n,m))
    t = 0

    H = np.eye(D) * h**2
    Hinv = np.eye(D) / h**2
    error = np.full(m, 1e+08)

    if wweights is None:
        weights = 1
    else:
        weights = wweights

    # start timing
    start_time = time.time()

    pct_error = np.percentile(error, converge_frac)

    while ((pct_error > eps) & (t < maxT)):
        # loop through iterations
        t = t + 1
        print("-------- iteration {0} --------".format(t))

        itermask = np.where(error > eps)
        GjList = G[itermask]
        GRes, errorRes = shift_particle_vec(GjList, X, D, h, d, weights, n, H, Hinv)
        G[itermask] = GRes
        error[itermask] = errorRes
        pct_error = np.percentile(error, converge_frac)
        print("{0}%-tile error: {1}".format(converge_frac, pct_error))

        elapsed_time = time.time() - start_time
        # print elapsed_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    return G


def shift_particle_vec(Gj, X, D, h, d, weights, n, H, Hinv):
    # compute probability density of each X point
    #c = multivariate_normal.pdf(X, mean=Gj, cov=H)

    #print(X.shape)
    #print(Gj.shape)

    c = gaussian_kde_scipy(Gj, X, h, weights)
    #c = gaussian_kde_scipy(X, Gj, h)

    print("c: \t {}".format(c.shape))
    print("weights: {}".format(weights.shape))

    # now weight the probability of each X point by the image
    #c *= weights

    # compute mean probability of the test particle
    pj = c.mean(axis=0)
    #pj = c.mean(axis=tuple(range(1, c.ndim)))
    print("pj: \t {}".format(pj.shape))


    # broadcast Gj for vectorization operation
    print("Gj: \t {}".format(Gj.shape))
    print("X: \t {}".format(X.shape))
    print("Hinv: \t {}".format(Hinv.shape))

    Gj = np.broadcast_to(Gj, (X.shape[0],) + Gj.shape)
    Gj = np.swapaxes(Gj, 0, 1)
    print("Gj: \t {}".format(Gj.shape))

    # compute gradient and Hessian
    u = np.matmul(Hinv, (Gj - X))/h**2
    print("u: \t {}".format(u.shape))
    #u = np.swapaxes(u, 0, 1)
    #u = np.einsum('ij, jkl->kil', Hinv, Gj - X)

    cb = np.broadcast_to(c[:, None, None], (Gj.shape[0],) + c[:, None, None].shape)
    #g = -np.sum(c[:, None] * u, axis=0) / n
    g = -np.sum(cb * u, axis=0) / n
    print("g: \t {}".format(g.shape))

    #Hess = np.sum(c[:, None, None] * (np.matmul(u[:, None], T_1D(u)) - Hinv), axis=0) / n
    uT = T_1D(u)
    print("uT: \t {}".format(uT.shape))

    Tmu = np.matmul(u, uT)
    print("Tmu: \t {}".format(Tmu.shape))

    yo = uT - Hinv
    print("yo: \t {}".format(yo.shape))

    cyo = c[None, :, None, None]*yo
    print("cyo: \t {}".format(cyo.shape))

    # sum over all the pixels (but not walkers)
    Hess = np.sum(cyo, axis=1)
    print("Hess: \t {}".format(Hess.shape))

    #yo  = np.matmul(u, Tu - Hinv)
    #print("yo: \t {}".format(yo.shape))
    #Hess = np.sum(c[:, None, None] * , axis=0) / n

    # compute inverse of the covariance matrix
    Sigmainv = -Hess/pj + np.matmul(g, T_1D(g))/pj**2

    # compute shift
    shift0 = Gj + np.matmul(H, g) / pj

    # compute eigenvectors with the largest eigenvalues down to D-d
    EigVal, EigVec = np.linalg.eigh(Sigmainv)
    V = EigVec[:, d:D]

    # shift the test particle
    Gj = np.matmul(V, np.matmul(V.T, shift0 - Gj)) + Gj

    # compute error
    tmp = np.matmul(V.T, g)
    errorj = np.sqrt(np.sum(tmp**2) / np.sum(g**2))
    return np.append(Gj.ravel(), [errorj])


def T_1D(mtxAry):
    # return an array of transposed 1D matrices
    if mtxAry.ndim == 4:
        return np.transpose(mtxAry, axes=(0,1,3,2))
    elif mtxAry.ndim == 3:
        return np.transpose(mtxAry, axes=(0,2,1))
    elif mtxAry.ndim == 2:
        return mtxAry.T

#=======================================================================================================================

from sklearn.neighbors import KernelDensity


def gaussian_kde_scipy(x, data, h, weights=None):
    """
    Perform a multi-dimensional Gaussian kernel density estimate
    at a given position using scipy library.

    Parameters:
    - x: array-like, shape (n_features,)
        The position at which to evaluate the density.
    - data: array-like, shape (n_samples, n_features)
        The data points used to estimate the density.
    - h: array-like, shape (n_features,)
        The bandwidths for each dimension.

    Returns:
    - density: float
        The estimated density at the given position.
    """

    # Remove dimensions of size 1 from ndarray (to be competible with older data format)
    data = np.squeeze(data)
    x = np.squeeze(x)

    '''
    kde = gaussian_kde(data, bw_method='gaussian_sigma', bw_args=h)
    density = kde(x)
    return density
    '''

    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(data, sample_weight=weights)
    # exp is used because score_sample returns log likelihood
    return np.exp(kde.score_samples(x))