import numpy as np
import time
import sys
import multiprocessing as mp
from sklearn.neighbors import KernelDensity

#======================================================================================================================#

def find_ridge(X, G, D=3, h=1, d=1, eps = 1e-06, maxT = 1000, wweights = None, converge_frac = 99, ncpu = None,
               return_unconverged=True):

    # use float32 to make the operation more efficient (particularly since the precision need isn't too high)
    G = G.astype(np.float32)
    X = X.astype(np.float32)
    h = np.float32(h)
    eps = np.float32(eps)
    wweights = np.float32(wweights)
    converge_frac = np.float32(converge_frac)

    n = len(X)
    m = len(G)  # x and y coordinates 2xN format
    print("n, m: {0}, {1}".format(n,m))
    t = 0

    H = np.eye(D) * h**2
    Hinv = np.eye(D) / h**2
    error = np.full(m, 1e+08, dtype=np.float32)

    if wweights is None:
        weights = np.float32(1)
    else:
        weights = wweights

    # start timing
    start_time = time.time()

    pct_error = np.percentile(error, converge_frac)

    # assign the number of cpus to use if not specified:
    if ncpu is None:
        ncpu = mp.cpu_count() - 1

    while ((pct_error > eps) & (t < maxT)):
        # loop through iterations
        t = t + 1

        itermask = error > eps
        GjList = G[itermask]

        if t%10 == 0:
            print("-------- iteration {0} --------".format(t))
            print("number of walkers remaining: {}".format(len(GjList)))

        GRes, errorRes = shift_particles(GjList, X, D, h, d, weights, n, H, Hinv)

        G[itermask] = GRes
        error[itermask] = errorRes

        pct_error = np.percentile(error, converge_frac)
        #print("{0}%-tile error: {1}".format(converge_frac, pct_error))

    mask = error < eps

    if return_unconverged:
        # return both the converged and unconverged results
        return G[mask], G[~mask]
    else:
        # return only converged results
        return G[mask]



def shift_particles(G, X, D, h, d, weights, n, H, Hinv):
    # shift individual walkers using SCMS

    # compute the gaussian
    c = vectorized_gaussian(X, G, h)
    c = c * weights

    # Compute the mean probability
    pj = np.mean(c, axis=1)

    # Expand dimensions for X and G to enable broadcasting for pairwise differences
    X_expanded = X[None, :, :, :]  # Shape (1, 7507, 2, 1)
    G_expanded = G[:, None, :, :]  # Shape (2179, 1, 2, 1)

    # Compute u for all walker points in G and all points in X
    u = np.matmul(Hinv, (G_expanded - X_expanded)) / h ** 2  # Shape (2179, 7507, 2, 1)

    # Compute g for all walker points in G
    c_expanded = c[:, :, None, None]  # Shape (2179, 7507, 1, 1) for broadcasting with u
    g = -1 * np.sum(c_expanded * u, axis=1) / n  # Shape (2179, 2, 1)

    # Compute the Hessian matrix for all walker points in G
    u_T = np.transpose(u, axes=(0, 1, 3, 2))  # Transpose u for broadcasting, shape (2179, 7507, 1, 2)
    Hess = np.sum(c_expanded * (np.matmul(u, u_T) - Hinv), axis=1) / n  # Shape (2179, 2, 2)

    # Expand dimensions for pj
    pj = pj[:,None, None]

    Sigmainv = -1 * Hess / pj +\
               np.matmul(g, np.transpose(g, axes=(0, 2, 1))) / pj** 2

    # Compute the shift for each walker point
    shift0 = G + np.matmul(H, g) / pj

    # Eigen decomposition for Sigmainv for each walker point
    EigVal, EigVec = np.linalg.eigh(Sigmainv)

    # Get the eigenvectors with the largest eigenvalues for each walker point
    V = EigVec[:, :, d:D]

    # Compute VVT for each walker point
    VVT = np.matmul(V, np.transpose(V, axes=(0, 2, 1)))

    # Update G for each walker point
    G = np.matmul(VVT, (shift0 - G)) + G

    # Compute the error term for each walker point
    tmp = np.matmul(np.transpose(V, axes=(0, 2, 1)), g)
    error = np.sqrt(np.sum(tmp ** 2, axis=(1, 2)) / np.sum(g ** 2, axis=(1, 2)))

    return G, error




def vectorized_gaussian(X, G, h):
    """
    Compute the Gaussian exponential efficiently for each walker in G against each point in X.

    Parameters:
    X : ndarray
        Array of data points with shape (n, 2, 1).
    G : ndarray
        Array of walkers with shape (m, 2, 1).
    h : float
        Scalar value representing the covariance (assumes isotropic covariance).

    Returns:
    c : ndarray
        Array of computed Gaussian exponentials with shape (m, n).
    """
    # Reshape X to shape (n, 2)
    X_squeezed = np.squeeze(X, axis=-1)

    # Reshape G to shape (m, 2)
    G_squeezed = np.squeeze(G, axis=-1)

    # Compute differences for all combinations of G and X
    diff = G_squeezed[:, np.newaxis, :] - X_squeezed[np.newaxis, :, :]  # Shape: (m, n, 2)

    # Compute the inverse covariance (assumes h is scalar)
    inv_cov = 1 / (h**2)

    # Calculate the exponent for the Gaussian function
    exponent = -0.5 * np.sum(diff**2 * inv_cov, axis=-1)  # Shape: (m, n)

    # Compute the Gaussian exponential
    c = np.exp(exponent)  # Shape: (m, n)

    return c


def gaussian(X, mean, covariance):
    inv_cov = 1 / covariance
    diff = X - mean
    return -0.5 * np.sum(diff**2 * inv_cov, axis=-1)
