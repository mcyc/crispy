import numpy as np
import os, time
#from scipy.stats import multivariate_normal
from . import shift_walkers as sw_pyx

def find_ridge(X, G, D=3, h=1.0, d=1, eps = 1e-03, maxT = 1000, wweights = None, converge_frac = 99):

    G = G.astype(np.float64)
    X = X.astype(np.float64)
    h = np.float64(h)
    n = len(X)
    m = len(G)  # x and y coordinates 2xN format
    print("n, m: {0}, {1}".format(n,m))
    t = 0

    H = np.eye(D) * h**2
    Hinv = np.eye(D) / h**2
    error = np.full(m, 1e+08)

    if wweights is None:
        weights = 1.0
    else:
        weights = wweights

    weights = np.float64(weights)

    # start timing
    start_time = time.time()

    pct_error = np.percentile(error, converge_frac)

    while ((pct_error > eps) & (t < maxT)):
        # loop through iterations
        t = t + 1
        print("-------- iteration {0} --------".format(t))

        itermask = np.where(error > eps)
        GjList = G[itermask]
        print("number of walkers remaining: {}".format(len(GjList)))

        #GRes, errorRes = shift_walkers(X, GjList, weights, h, H, Hinv, n, d, D)
        GRes, errorRes = sw_pyx.shift_walkers(X, GjList, weights, h, H, Hinv, n, d, D)
        G[itermask] = GRes
        error[itermask] = errorRes

        pct_error = np.percentile(error, converge_frac)
        print("{0}%-tile error: {1}".format(converge_frac, pct_error))

        elapsed_time = time.time() - start_time
        # print elapsed_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("number of cpu to be used: {}".format(1))

    return G

def shift_walkers(X, G, weights, h, H, Hinv, n, d, D):
    # loop through points in the mesh (i.e., G)

    m = len(G)
    newG = np.zeros(G.shape)
    newErr = np.zeros(m)

    for j, Gj in enumerate(G):
        newG[j], newErr[j] = shift_walker(X, Gj, weights, h, H, Hinv, n, d, D)
    return newG, newErr

def shift_walker(X, Gj, weights, h, H, Hinv, n, d, D):

    #c = multivariate_normal.pdf(X.reshape(X.shape[0:2]), mean=Gj.ravel(), cov=H)
    c = np.exp(gaussian_logpdf(np.squeeze(X), mean=Gj.ravel(), covariance=h**2))

    # now weight the probability of each X point by the image
    c = c * weights
    # reshape c so it can be broadcasted onto 3 dimension arrays
    c = c[:, None, None]
    pj = np.mean(c)

    u = np.matmul(Hinv, (Gj - X)) / h ** 2
    g = -1 * np.sum((c * u), axis=0) / n

    # compute the Hessian matrix
    Hess = np.sum(c * (np.matmul(u, T_1D(u)) - Hinv), axis=0) / n

    Sigmainv = -1 * Hess / pj + np.matmul(g, g.T) / pj ** 2
    shift0 = Gj + np.matmul(H, g) / pj

    # Sigmainv matrices computed here are symmetric, and thus linalg.eigh is preferred
    # note that the eigenvectors in linalg.eigh is already sorted unlike linalg.eig
    EigVal, EigVec = np.linalg.eigh(Sigmainv)

    # get the eigenvectors with the largest eigenvalues down to d-1
    V = EigVec[:, d:D]

    VVT = np.matmul(V, V.T)
    Gj = np.matmul(VVT, shift0 - Gj) + Gj

    tmp = np.matmul(V.T, g)
    errorj = np.sqrt(np.sum(tmp ** 2) / np.sum(g ** 2))
    #return Gj.ravel(), errorj
    return Gj, errorj

def T_1D(mtxAry):
    # return an array of transposed 1D matrices
    return np.transpose(mtxAry, axes=(0,2,1))


def gaussian_logpdf(X, mean, covariance):
    """
    Compute log N(x_i; mu, sigma) for each x_i, mu_i, sigma_i
    Args:
        X : shape (n, d)
            Data points
        means : shape (d)
            Mean vectors
        covariances : float
            The covariance of the gaussian, same for all dimensions
    Returns:
        logpdfs : shape (n,)
            Log probabilities
    """
    d = X.shape[1]
    constant = d * np.log(2 * np.pi)
    #log_determinants = np.log(np.prod(covariance, axis=0)) # this would work for a diagnoal covariance matrix
    log_determinants = np.log(covariance)
    deviations = X - mean
    inverses = 1 / covariance
    return -0.5 * (constant + log_determinants +
        np.sum(deviations * inverses * deviations, axis=1))




