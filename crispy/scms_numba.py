import numpy as np
import numba

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