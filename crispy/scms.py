import numpy as np
import time
import sys

#======================================================================================================================#

def find_ridge(X, G, D=3, h=1, d=1, eps=1e-06, maxT=1000, weights=None, converge_frac=99, ncpu=None,
               return_unconverged=True, f_h=5):
    '''
    Find the density ridge in the data by iteratively moving the walkers towards the ridge
    :param X: Coordinates of the data points in the shape of (n, D, 1)
    :param G: Coordinates of the walkers in the shape of (m, D, 1)
    :param D: The dimension of the data
    :param h: The smoothing bandwidth of the Gaussian kernel
    :param d:
    :param eps: Convergence criteria for individual walkers.
    :param maxT: Maximum iteration for the run
    :param weights: The weight of each data point (for an image, that's the value at each pixel)
    :param converge_frac: The fraction of walkers converged required to end the run, in percent.
    :param ncpu: The number of cpu to use
    :param return_unconverged:
    :param f_h: A factor to filter out X by their shortest distances to all data points.
        X with distances > f_h*h are removed
    :return:
    '''

    # use float32 to make the operation more efficient (particularly since the precision need isn't too high)
    G = G.astype(np.float32)
    X = X.astype(np.float32)
    h = np.float32(h)
    eps = np.float32(eps)
    if weights is None:
        weights = np.float32(1)
    else:
        weights = weights.astype(np.float32)
    converge_frac = np.float32(converge_frac)

    n = len(X)
    m = len(G)
    t = 0

    H = np.eye(D) * h**2
    Hinv = np.eye(D) / h**2
    error = np.full(m, 1e+08, dtype=np.float32)

    print("==========================================================================")
    print(f"Starting the run. The number of image points and walkers  are {n} and {m}")
    print("==========================================================================")

    # start timing
    start_time = time.time()
    last_print_time = start_time

    pct_error = np.percentile(error, converge_frac)

    while ((pct_error > eps) & (t < maxT)):
        # loop through iterations
        t = t + 1

        itermask = error > eps
        GjList = G[itermask]

        # filter out X points that are too far to save computing time and evaluate the weighted Gaussian
        X, c, weights = wgauss_n_filtered_points(X, GjList, h, weights, f_h=f_h)

        ni, mi = len(X), len(GjList)

        current_time = time.time()
        if current_time - last_print_time >= 1:
            elapsed_time = current_time - start_time
            formatted_time = time.strftime("%H:%M:%S ", time.gmtime(elapsed_time))
            sys.stdout.write(f"\rIteration {t}"
                             f" | Data points: {ni}"
                             f" | Walkers remaining: {mi}/{m} ({100 - mi/m*100:0.1f}% complete)"
                             f" | {converge_frac}-percentile error: {pct_error:0.3f}"
                             f" | Total run time: {formatted_time}")
            sys.stdout.flush()

        GRes, errorRes = shift_particles(GjList, X, D, h, d, c, ni, H, Hinv)

        G[itermask], error[itermask] = GRes, errorRes
        pct_error = np.percentile(error, converge_frac)

    sys.stdout.write("\n")
    mask = error < eps

    if return_unconverged:
        # return both the converged and unconverged results
        return G[mask], G[~mask]
    else:
        # return only converged results
        return G[mask]


def wgauss_n_filtered_points(X, G, h, weights, f_h=8):
    '''
    Computed the weighted Gaussian at positions X, with means in G, but only for X positions with distances < f_h*h from  all G
    :param X: Coordinates of the data points in the shape of (n, D, 1)
    :param G: Coordinates of the walkers in the shape of (m, D, 1)
    :param h: The smoothing bandwidth of the Gaussian
    :param weights: the weights of X in the shape of (m, D, 1)
    :param f_h: the distance multiiple cut off for which X with minimum distances < f_h*h from all G points are filtered out
    :return: 
    '''

    # find data points too far from the walkers
    dist, diff = euclidean_dist(X, G)
    toofar = np.all(dist > f_h * h, axis=0)

    # filter out the distant data
    X = X[~toofar, :, :]
    diff = diff[:, ~toofar, :]
    weights = weights[~toofar]

    # Calculate the Gaussian values:
    inv_cov = 1 / (h**2)
    exponent = -0.5 * np.sum(diff**2 * inv_cov, axis=-1)
    c = np.exp(exponent)

    return X, c*weights, weights


def shift_particles(G, X, D, h, d, c, n, H, Hinv):
    """
    Shift individual walkers in G using SCMS towards the density ridges
    note: the old method used to take arguments of: G, X, D, h, d, weights, n, H, Hinv

    :param X: Coordinates of the data points in the shape of (n, D, 1)
    :param G: Coordinates of the walkers in the shape of (m, D, 1)
    :param D: The dimension of the data points
    :param h: The smoothing bandwidth of the Gaussian
    :param d:
    :param c: the weighted Gaussian value computed from G and X
    :param n:
    :param H:
    :param Hinv:
    :return:
    """

    # Compute the mean probability
    pj = np.mean(c, axis=1)

    # Expand dimensions for X and G to enable broadcasting for pairwise differences
    X_expanded = X[None, :, :, :]  # Shape (1, n, 2, 1)
    G_expanded = G[:, None, :, :]  # Shape (m, 1, 2, 1)

    # Compute u for all walker points in G and all points in X
    u = np.matmul(Hinv, (G_expanded - X_expanded)) / h ** 2  # Shape (m, n, 2, 1)

    # Compute g for all walker points in G
    c_expanded = c[:, :, None, None]  # Shape (m, n, 1, 1) for broadcasting with u
    g = -1 * np.sum(c_expanded * u, axis=1) / n  # Shape (m, 2, 1)

    # Compute the Hessian matrix for all walker points in G
    u_T = np.transpose(u, axes=(0, 1, 3, 2))  # Transpose u for broadcasting, shape (m, n, 1, 2)
    Hess = np.sum(c_expanded * (np.matmul(u, u_T) - Hinv), axis=1) / n  # Shape (m, 2, 2)

    # Expand dimensions for pj
    pj = pj[:, None, None]

    Sigmainv = -1 * Hess / pj + \
               np.matmul(g, np.transpose(g, axes=(0, 2, 1))) / pj ** 2

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


def euclidean_dist(X, G):
    """
    Compute the Euclidean distances between X and G points.
    Parameters:
    X : ndarray
        An array of positions (n, D, 1).
    G : ndarray
        An array of the other positions (m, D, 1).
    h : float
        Scalar value representing the covariance (assumes isotropic covariance).

    Returns:
    """
    X_squeezed = np.squeeze(X, axis=-1)
    G_squeezed = np.squeeze(G, axis=-1)

    # Compute differences for all combinations of G and X
    diff = G_squeezed[:, np.newaxis, :] - X_squeezed[np.newaxis, :, :]  # Shape: (m, n, D)
    distances = np.linalg.norm(diff, axis=-1)  # Shape: (m, n)

    return distances, diff



def vectorized_gaussian(X, G, h):
    """
    Compute the Gaussian exponential efficiently for each point in X for each mean position in G
    Parameters:
    X : ndarray
        An array of positions to evaluate with shape (n, D, 1).
    G : ndarray
        An array of mean positions with shape (m, D, 1).
    h : float
        Scalar value representing the covariance (assumes isotropic covariance).

    Returns:
    c : ndarray
        Array of computed Gaussian exponentials with shape (m, n).
    distances : ndarray
        Array of Euclidean distances between each walker in G and each point in X, with shape (m, n).
    """
    # Calculate the Euclidean distances between each walker in G and each point in X
    X_squeezed = np.squeeze(X, axis=-1)
    G_squeezed = np.squeeze(G, axis=-1)

    # Compute differences for all combinations of G and X
    diff = G_squeezed[:, np.newaxis, :] - X_squeezed[np.newaxis, :, :]  # Shape: (m, n, D)

    # Compute the inverse covariance (assumes h is scalar)
    inv_cov = 1 / (h**2)

    # Calculate the exponent for the Gaussian function
    exponent = -0.5 * np.sum(diff**2 * inv_cov, axis=-1)  # Shape: (m, n)

    # Compute the Gaussian exponential
    c = np.exp(exponent)  # Shape: (m, n)

    return c