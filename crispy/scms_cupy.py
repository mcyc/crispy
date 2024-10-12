import cupy as cp
import time
import sys

#======================================================================================================================#

def find_ridge(X, G, D=3, h=1, d=1, eps=1e-06, maxT=1000, wweights=None, converge_frac=99, ncpu=None,
               return_unconverged=True):

    # use float32 to make the operation more efficient (particularly since the precision need isn't too high)
    G = G.astype(cp.float32)
    X = X.astype(cp.float32)
    h = cp.float32(h)
    eps = cp.float32(eps)
    wweights = cp.float32(wweights)
    converge_frac = cp.float32(converge_frac)

    n = len(X)
    m = len(G)  # x and y coordinates 2xN format
    t = 0

    H = cp.eye(D) * h**2
    Hinv = cp.eye(D) / h**2
    error = cp.full(m, 1e+08, dtype=cp.float32)

    if wweights is None:
        weights = cp.float32(1)
    else:
        weights = wweights

    print("==========================================================================")
    print(f"Starting the run. The number of image points and walkers  are {n} and {m}")
    print("==========================================================================")

    # start timing
    start_time = time.time()
    last_print_time = start_time

    pct_error = cp.percentile(error, converge_frac)

    while ((pct_error > eps) & (t < maxT)):
        # loop through iterations
        t = t + 1

        itermask = error > eps
        GjList = G[itermask]

        current_time = time.time()
        if current_time - last_print_time >= 1:
            elapsed_time = current_time - start_time
            formatted_time = time.strftime("%H:%M:%S ", time.gmtime(elapsed_time))
            sys.stdout.write(f"\rIteration {t}"
                             f" | Number of walkers remaining: {len(GjList)}/{m} ({100 - len(GjList)/m*100:0.1f}% complete)"
                             f" | {converge_frac}-percentile error: {pct_error:0.3f}"
                             f" | total run time: {formatted_time}")
            sys.stdout.flush()

        GRes, errorRes = shift_particles(GjList, X, D, h, d, weights, n, H, Hinv)

        G[itermask] = GRes
        error[itermask] = errorRes

        pct_error = cp.percentile(error, converge_frac)

    sys.stdout.write("\n")
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
    pj = cp.mean(c, axis=1)

    # Expand dimensions for X and G to enable broadcasting for pairwise differences
    X_expanded = X[None, :, :, :]  # Shape (1, 7507, 2, 1)
    G_expanded = G[:, None, :, :]  # Shape (2179, 1, 2, 1)

    # Compute u for all walker points in G and all points in X
    u = cp.matmul(Hinv, (G_expanded - X_expanded)) / h ** 2  # Shape (2179, 7507, 2, 1)

    # Compute g for all walker points in G
    c_expanded = c[:, :, None, None]  # Shape (2179, 7507, 1, 1) for broadcasting with u
    g = -1 * cp.sum(c_expanded * u, axis=1) / n  # Shape (2179, 2, 1)

    # Compute the Hessian matrix for all walker points in G
    u_T = cp.transpose(u, axes=(0, 1, 3, 2))  # Transpose u for broadcasting, shape (2179, 7507, 1, 2)
    Hess = cp.sum(c_expanded * (cp.matmul(u, u_T) - Hinv), axis=1) / n  # Shape (2179, 2, 2)

    # Expand dimensions for pj
    pj = pj[:, None, None]

    Sigmainv = -1 * Hess / pj + \
               cp.matmul(g, cp.transpose(g, axes=(0, 2, 1))) / pj ** 2

    # Compute the shift for each walker point
    shift0 = G + cp.matmul(H, g) / pj

    # Eigen decomposition for Sigmainv for each walker point
    EigVal, EigVec = cp.linalg.eigh(Sigmainv)

    # Get the eigenvectors with the largest eigenvalues for each walker point
    V = EigVec[:, :, d:D]

    # Compute VVT for each walker point
    VVT = cp.matmul(V, cp.transpose(V, axes=(0, 2, 1)))

    # Update G for each walker point
    G = cp.matmul(VVT, (shift0 - G)) + G

    # Compute the error term for each walker point
    tmp = cp.matmul(cp.transpose(V, axes=(0, 2, 1)), g)
    error = cp.sqrt(cp.sum(tmp ** 2, axis=(1, 2)) / cp.sum(g ** 2, axis=(1, 2)))

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
    X_squeezed = cp.squeeze(X, axis=-1)

    # Reshape G to shape (m, 2)
    G_squeezed = cp.squeeze(G, axis=-1)

    # Compute differences for all combinations of G and X
    diff = G_squeezed[:, cp.newaxis, :] - X_squeezed[cp.newaxis, :, :]  # Shape: (m, n, 2)

    # Compute the inverse covariance (assumes h is scalar)
    inv_cov = 1 / (h**2)

    # Calculate the exponent for the Gaussian function
    exponent = -0.5 * cp.sum(diff**2 * inv_cov, axis=-1)  # Shape: (m, n)

    # Compute the Gaussian exponential
    c = cp.exp(exponent)  # Shape: (m, n)

    return c

def gaussian(X, mean, covariance):
    inv_cov = 1 / covariance
    diff = X - mean
    return -0.5 * cp.sum(diff**2 * inv_cov, axis=-1)