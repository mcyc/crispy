
import numpy as np
import numba
@numba.njit(parallel=True, fastmath=True)
def vectorized_gaussian(X, G, h):
    """
    Compute the Gaussian kernel values efficiently for each walker in G against each point in X.

    Parameters:
    X : ndarray
        Array of data points with shape (n, D, 1).
    G : ndarray
        Array of walkers with shape (m, D, 1).
    h : float
        Bandwidth parameter for Gaussian kernel.

    Returns:
    c : ndarray
        Array of computed Gaussian kernel values with shape (m, n).
    """
    # Remove the last dimension using slicing and ensure arrays are contiguous
    X_squeezed = X[:, :, 0].copy()  # Resulting shape will be (n, D), with a contiguous copy
    G_squeezed = G[:, :, 0].copy()  # Resulting shape will be (m, D), with a contiguous copy

    # Expand dimensions manually using reshaping for broadcasting compatibility
    # Reshape G_squeezed to (m, 1, D)
    G_expanded = G_squeezed.reshape(G_squeezed.shape[0], 1, G_squeezed.shape[1])
    # Reshape X_squeezed to (1, n, D)
    X_expanded = X_squeezed.reshape(1, X_squeezed.shape[0], X_squeezed.shape[1])

    # Now G_expanded and X_expanded are compatible for broadcasting
    # G_expanded: (m, 1, D), X_expanded: (1, n, D)
    # Compute the pairwise differences between walkers and data points
    diff = G_expanded - X_expanded  # Shape (m, n, D)

    # Compute the inverse covariance (assumes isotropic covariance)
    inv_cov = 1 / (h ** 2)

    # Calculate the exponent for the Gaussian kernel
    exponent = -0.5 * np.sum(diff ** 2 * inv_cov, axis=-1)

    # Compute the Gaussian kernel values
    c = np.exp(exponent)  # Shape (m, n)

    return c