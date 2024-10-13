import numpy as np
import numba


import numpy as np
import time
import sys

#from . import numba_func as nb
from . import scms_numba

# ======================================================================================================================#

#@numba.njit(parallel=True, fastmath=True)
def find_ridge(X, G, D=3, h=1, d=1, eps=1e-06, maxT=1000, wweights=None, converge_frac=99.0, return_unconverged=True,
               numba=True):
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
        GRes, errorRes = move_walkers(GjList, X, D, h, d, weights, n, H, Hinv)

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


@numba.njit(parallel=True, fastmath=True)
def move_walkers(G, X, D, h, d, weights, n, H, Hinv):
    errors = np.full(G.shape[0], 1e+08, dtype=np.float32)
    for j, Gj in enumerate(G):
        G[j], errors[j] = shift_particle(Gj, X, D, h, d, weights, n, H, Hinv)

    return G, errors

@numba.njit(parallel=False, fastmath=True)
def shift_particle(Gj, X, D, h, d, weights, n, H, Hinv):
    # shift individual walkers using SCMS
    c = np.zeros(weights.shape, dtype=weights.dtype) #(n,)
    cu = np.zeros(X.shape, dtype=weights.dtype) #(n, D, 1)
    hh = np.zeros((n, D, D), dtype=weights.dtype) #(n,D,D)

    for i, Xi in enumerate(X):  # Xi has shape (D,)
        # Evaluate the Gaussian value of Xi at Gj
        #c_i = np.exp(gaussian(Xi, mean=Gj, covariance=h ** 2))  # c is a scalar
        c_i = evaluate_gaussian(Xi, mean=Gj, covariance=h**2)

        # Now weight the probability of Xi by the image
        c[i] = c_i* weights[i]  # c is a scalar

        # Compute u, which is used for shifting
        u_i = np.dot(Hinv, (Gj - Xi)) / h**2  # u has shape (D, 1)
        cu[i] = c[i]*u_i # (D,1)
        hh[i] = c[i]*np.dot(u_i, np.transpose(u_i)) - Hinv #(D,D)

    pj = np.mean(c)  # pj is a scalar
    pj = np.float32(pj)
    g = -1 * np.sum(cu, axis=0) / n  # g has shape (D, 1)

    # Compute the Hessian matrix
    Hess = np.sum(hh, axis=0) / n  # Hess has shape (D, D)

    # Compute Sigmainv, which is an approximation of the inverse covariance matrix
    Sigmainv = -1 * Hess / pj + np.dot(g, np.transpose(g)) / pj ** 2  # Sigmainv has shape (D, D)

    # Update shift0 based on gradient
    shift0 = Gj + np.dot(H, g) / pj  # shift0 has shape (D,1)

    # Eigen decomposition of Sigmainv to find the principal directions
    EigVal, EigVec = np.linalg.eigh(Sigmainv)  # EigVal has shape (D,), EigVec has shape (D, D)

    # Get the eigenvectors with the largest eigenvalues down to D-d
    V = np.ascontiguousarray(EigVec[:, d:D])  # V has shape (D, D-d) .astype(np.dtype('float32'), copy=False)

    # Compute V * V^T
    VT = np.transpose(V)
    VVT = np.dot(V, VT)  # VVT has shape (D, D)

    # Update Gj based on the projection
    Gj = np.dot(VVT, shift0 - Gj) + Gj  # Gj has shape (D,)

    # Compute error
    tmp = np.dot(VT, g)  # tmp has shape (D-d,)
    errorj = np.sqrt(np.sum(tmp ** 2) / np.sum(g ** 2))  # errorj is a scalar

    return Gj, errorj

@numba.njit(parallel=False, fastmath=True)
def gaussian(X, mean, covariance):
    """
    Compute N(X; mean, covariance) for a given point X in D-dimensional space
    Args:
        X : shape (D, 1)
            Data point in D-dimensional space
        mean : shape (D, 1)
            A mean vector in D-dimensional space
        covariance : float
            A single covariance of the Gaussian, same for all dimensions.
            This calculation assumes a diagonal covariance matrix with identical elements
    Returns:
        pdf : float
            Probability density function evaluated at X
    """
    D = X.shape[0]
    constant = D * np.log(2 * np.pi)
    log_determinants = np.log(covariance)
    deviations = X - mean
    inverses = 1 / covariance
    return -0.5 * (constant + log_determinants + np.sum(deviations * inverses * deviations))

@numba.njit(parallel=False, fastmath=True)
def evaluate_gaussian(X, mean, covariance):
    """
    Evaluate a D-dimensional Gaussian function at a given position.

    Parameters:
    X (numpy.ndarray): Position at which to evaluate the Gaussian, shape (D, 1).
    mean (numpy.ndarray): Mean of the Gaussian, shape (D, 1).
    covariance (float): Scalar covariance value (isotropic covariance, `covariance`).

    Returns:
    float: Value of the Gaussian function at position X.
    """
    D = mean.shape[0]
    diff = X - mean
    inv_covariance = 1 / covariance  # Since covariance is scalar, the inverse covariance matrix is just 1/covariance
    normalization_factor = 1 / ((2 * np.pi * covariance) ** (D / 2))
    exponent = -0.5 * np.dot(diff.T, diff) * inv_covariance
    value = normalization_factor * np.exp(exponent)
    return value.item()  # Return as a scalar