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

    print(X.shape)
    print(Gj.shape)

    XX = np.broadcast_to(X, (Gj.shape[0],) + X.shape)
    GG = np.broadcast_to(Gj, (X.shape[0],) + Gj.shape)
    GG = np.swapaxes(GG, 0, 1)

    # note X.shape[-1] should == D
    HH = np.broadcast_to(h**2, (Gj.shape[0],) + (X.shape[0],) + (X.shape[-1],))
    print("HH: \t {}".format(HH.shape))
    c = np.exp(vectorized_gaussian_logpdf(XX, means=GG, covariances=HH))

    print("c: \t {}".format(c.shape))
    print("weights: {}".format(weights.shape))

    # now weight the probability of each X point by the image (i.e., c *= weights)
    c *= np.broadcast_to(weights, c.shape)

    # compute mean probability of the test particle
    pj = c.mean(axis=1)
    print("pj: \t {}".format(pj.shape))

    # broadcast Gj for vectorization operation
    print("Gj: \t {}".format(Gj.shape))
    print("X: \t {}".format(X.shape))
    print("Hinv: \t {}".format(Hinv.shape))

    Gj = np.broadcast_to(Gj, (X.shape[0],) + Gj.shape)
    Gj = np.swapaxes(Gj, 0, 1)

    print("Gj: \t {}".format(Gj.shape))
    print("X: \t {}".format(X.shape))
    print("XX: \t {}".format(XX.shape))

    # compute gradient and Hessian
    # X can probably be fine for the operation below without broadcasting to XX
    u = np.matmul(Hinv, (Gj - XX))/h**2
    print("u: \t {}".format(u.shape))

    cb = np.broadcast_to(c[:, :, None, None], u.shape)
    print("cb: \t {}".format(cb.shape))

    g = -np.sum(cb * u, axis=1) / n
    print("g: \t {}".format(g.shape))

    uT = T_1D(u)
    print("uT: \t {}".format(uT.shape))

    Tmu = np.matmul(u, uT)
    print("Tmu: \t {}".format(Tmu.shape))

    # no broadcasting needed for Hinv?
    yo = Tmu - Hinv
    print("yo: \t {}".format(yo.shape))

    cyo = cb*yo
    print("cyo: \t {}".format(cyo.shape))

    # sum over all the pixels (but not walkers)
    Hess = np.sum(cyo, axis=1)/ n
    print("Hess: \t {}".format(Hess.shape))


    # compute inverse of the covariance matrix
    ppj = np.broadcast_to(pj[:, None, None], Hess.shape)
    print("ppj: \t {}".format(ppj.shape))
    Sigmainv = -Hess / ppj + np.matmul(g, T_1D(g)) / ppj ** 2

    print("Sigmainv: \t {}".format(Sigmainv.shape))

    ho = np.matmul(H, g)
    print("ho: \t {}".format(ho.shape))

    ppj2 = np.broadcast_to(pj[:, None, None], ho.shape)
    print("ppj2: \t {}".format(ppj2.shape))

    # compute shift
    ta = np.matmul(H, g) / ppj2
    ta = np.broadcast_to(ta[:, None], Gj.shape)
    print("ta: \t {}".format(ta.shape))

    shift0 = Gj + ta
    # compute eigenvectors with the largest eigenvalues down to D-d
    EigVal, EigVec = np.linalg.eigh(Sigmainv)
    V = EigVec[:, d:D]
    print("V: \t {}".format(V.shape))
    V = np.broadcast_to(V[:,None], uT.shape)
    print("V: \t {}".format(V.shape))

    # shift the test particle
    VT = T_1D(V)
    print("VT: \t {}".format(VT.shape))
    # I need to check if this reverse order for the vectorization make sense
    VVT = np.matmul(VT, V)

    print("VVT: \t {}".format(VVT.shape))
    print("shift0: \t {}".format(shift0.shape))
    ka = shift0 - Gj
    print("ka: \t {}".format(ka.shape))

    Gj = np.matmul(VVT, ka) + Gj

    # compute error
    g = np.broadcast_to(g[:,None], VT.shape)
    print("g: \t {}".format(g.shape))
    tmp = np.matmul(V, g)
    print("tmp: \t {}".format(tmp.shape))

    # reduce redundancy
    g = g[:,0,:]
    tmp = tmp[:,0,:]

    errorj = np.sqrt(np.sum(tmp**2, axis=(1,2)) / np.sum(g**2, axis=(1,2)))

    print("Gj: \t {}".format(Gj.shape))
    print("errorj: \t {}".format(errorj.shape))
    Gjf = Gj[:,0,:]
    print("Gjf: \t {}".format(Gjf.shape))

    return Gjf, errorj


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



def vectorized_gaussian_logpdf(X, means, covariances):
    """
    Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i
    Note: this assumes the covariance matrices constructed from covariances are diagonal
    The n, m, d in the arguments are the number of X, mean, and data dimesions, respectively
    Args:
        X : shape (n, m, d)
            Data points
        means : shape (n,m, d)
            Mean vectors
        covariances : shape (n,m, d)
            Diagonal covariance matrices
    Returns:
        logpdfs : shape (n,)
            Log probabilities
    """

    # remove the extra axis
    X = np.squeeze(X)
    means = np.squeeze(means)

    # find the dimesions of the data
    d = X.shape[-1]
    constant = d * np.log(2 * np.pi)
    #log_determinants = np.log(np.prod(covariances, axis=1))
    log_determinants = np.log(np.prod(covariances, axis=-1))

    deviations = X - means
    inverses = 1 / covariances

    '''
    print("constant: {}".format(constant.shape))
    print("log_determinants: {}".format(log_determinants.shape))
    print("deviations: {}".format(deviations.shape))
    print("inverses: {}".format(inverses.shape))
    '''

    #return -0.5 * (constant + log_determinants +
    #    np.sum(deviations * inverses * deviations, axis=1))
    return -0.5 * (constant + log_determinants +
        np.sum(deviations * inverses * deviations, axis=-1))
