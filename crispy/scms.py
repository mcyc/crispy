import numpy as np
import time
import sys

from . import numba_func as nb

# ======================================================================================================================#

#@numba.njit(parallel=True, fastmath=True)
def find_ridge(X, G, D=3, h=1, d=1, eps=1e-06, maxT=1000, wweights=None, converge_frac=99.0, return_unconverged=True):
    """
    Find the ridge of data points using the Subspace Constrained Mean Shift (SCMS) algorithm.

    Parameters:
    X : ndarray
        Array of data points with shape (n, D).
    G : ndarray
        Array of initial walkers with shape (m, D).
    D : int, optional
        Dimensionality of the data space (default is 3).
    h : float, optional
        Bandwidth parameter for Gaussian kernel (default is 1).
    d : int, optional
        Dimensionality of the subspace to constrain (default is 1).
    eps : float, optional
        Convergence threshold for error (default is 1e-06).
    maxT : int, optional
        Maximum number of iterations (default is 1000).
    wweights : ndarray, optional
        Weights for each data point in X (default is None).
    converge_frac : float, optional
        Fraction of walkers required to converge (default is 99.0).
    return_unconverged : bool, optional
        Whether to return unconverged walkers (default is True).

    Returns:
    G_converged : ndarray
        Array of converged walker positions.
    G_unconverged : ndarray
        Array of unconverged walker positions (if return_unconverged is True).
    """
    # Convert inputs to float32 for efficiency and ensure they are little-endian

    G = G.astype(np.dtype('float32').newbyteorder('<'), copy=False)
    X = X.astype(np.dtype('float32').newbyteorder('<'), copy=False)
    h = np.float32(h)
    eps = np.float32(eps)
    converge_frac = np.float32(converge_frac)

    n = len(X)  # Number of data points
    m = len(G)  # Number of walkers
    t = 0  # Iteration counter

    # Initialize covariance matrix H and its inverse
    H = np.eye(D, dtype=np.float32) * h ** 2
    Hinv = np.eye(D, dtype=np.float32) / h ** 2
    error = np.full(m, 1e+08, dtype=np.float32)  # Initial large error for all walkers

    # Set weights to 1 if not provided
    weights = np.float32(1) if wweights is None else wweights.astype(np.dtype('float32').newbyteorder('<'), copy=False)

    # Print initial information
    print("==========================================================================")
    print(f"Starting the run. The number of image points and walkers are {n} and {m}")
    print("==========================================================================")

    # Start timing
    start_time = time.time()
    last_print_time = start_time

    pct_error = np.percentile(error, converge_frac)

    # Iterate until convergence or maximum iterations reached
    while ((pct_error > eps) & (t < maxT)):
        # Loop through iterations
        t += 1

        # Identify walkers that have not converged
        itermask = error > eps
        if np.sum(itermask) == 0:
            # All walkers have converged
            break

        # Select only the unconverged walkers
        GjList = G[itermask]

        # Print progress every second
        current_time = time.time()
        if current_time - last_print_time >= 1:
            elapsed_time = current_time - start_time
            formatted_time = time.strftime("%H:%M:%S ", time.gmtime(elapsed_time))
            sys.stdout.write(f"\rIteration {t}"
                             f" | Number of walkers remaining: {len(GjList)}/{m} ({100 - len(GjList) / m * 100:0.1f}% complete)"
                             f" | {converge_frac}-percentile error: {pct_error:0.3f}"
                             f" | total run time: {formatted_time}")
            sys.stdout.flush()
            last_print_time = current_time

        # Shift the unconverged walkers
        GRes, errorRes = shift_particles(GjList, X, D, h, d, weights, n, H, Hinv)

        # Update the positions and errors of the unconverged walkers
        G[itermask] = GRes
        error[itermask] = errorRes

        # Calculate the error percentile for convergence check
        pct_error = np.percentile(error, converge_frac)
        if pct_error <= eps:
            # Convergence criteria met
            break

    sys.stdout.write("\n")

    # Mask for converged walkers
    mask = error < eps

    if return_unconverged:
        # Return both converged and unconverged walkers
        return G[mask], G[~mask]
    else:
        # Return only converged walkers
        return G[mask]

def shift_particles(G, X, D, h, d, weights, n, H, Hinv):
    """
    Shift individual walkers using the SCMS update rule.

    Parameters:
    G : ndarray
        Array of walkers with shape (m, D).
    X : ndarray
        Array of data points with shape (n, D).
    D : int
        Dimensionality of the data space.
    h : float
        Bandwidth parameter for Gaussian kernel.
    d : int
        Dimensionality of the subspace to constrain.
    weights : ndarray
        Weights for each data point in X.
    n : int
        Number of data points.
    H : ndarray
        Covariance matrix.
    Hinv : ndarray
        Inverse of the covariance matrix.

    Returns:
    G : ndarray
        Updated walker positions.
    error : ndarray
        Error values for each walker.
    """

    c = nb.vectorized_gaussian(X, G, h)
    c = c * weights
    pj = np.mean(c, axis=1)

    # Expand dimensions for broadcasting
    G_expanded = G[:, None, :]
    X_expanded = X[None, :, :]

    # Compute u, the gradient of the log-density estimate for each walker
    u = np.matmul(Hinv, (G_expanded - X_expanded)) / h ** 2

    # Compute g, the direction vector for each walker
    c_expanded = c[:, :, None, None]

    g = -np.sum(c_expanded * u, axis=1) / n

    # Compute the Hessian matrix for each walker
    u_T = u.transpose(0, 1, 3, 2)
    Hess = np.sum(c_expanded * (np.matmul(u, u_T) - Hinv), axis=1) / n

    # Expand dimensions for broadcasting
    pj = pj[:, None, None]
    # Compute the inverse of the covariance matrix for each walker
    Sigmainv = -1 * Hess / pj + \
               np.matmul(g, np.transpose(g, axes=(0, 2, 1))) / pj ** 2

    # Compute the shift for each walker
    shift0 = G + np.matmul(H, g) / pj

    # Perform eigen decomposition to find the principal directions
    EigVal, EigVec = np.linalg.eigh(Sigmainv)
    V = EigVec[:, :, d:D]
    # Project the shift onto the subspace defined by the principal directions
    VVT = np.matmul(V, V.transpose(0, 2, 1))

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