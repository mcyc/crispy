
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




@numba.njit(parallel=True, fastmath=True)
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

    c = vectorized_gaussian(X, G, h)
    c = c * weights # (m, n)

    # Compute the mean in numba
    # pj = np.mean(c, axis=1)
    sum_c = np.sum(c, axis=1)
    count = c.shape[1]
    pj = sum_c / count # (611,)

    # X (2179, 2, 1)
    # G (611, 2, 1)

    # Remove the last dimension using slicing and ensure arrays are contiguous
    X_squeezed = X[:, :, 0].copy()  # Resulting shape will be (n, D), with a contiguous copy
    G_squeezed = G[:, :, 0].copy()  # Resulting shape will be (m, D), with a contiguous copy

    # Reshape G_squeezed to (m, 1, D)
    G_expanded = G_squeezed.reshape(G_squeezed.shape[0], 1, G_squeezed.shape[1])
    # Reshape X_squeezed to (1, n, D)
    X_expanded = X_squeezed.reshape(1, X_squeezed.shape[0], X_squeezed.shape[1])

    diff = (G_expanded - X_expanded) # (m,n,D)

    u = matmul(Hinv, diff) / h ** 2  # (m,n,D)

    # get c into (m,n,D) shape
    c_expanded = np.empty(u.shape, dtype=c.dtype)
    for d in range(D):
        c_expanded[:, :, d] = c

    g = -np.sum(c_expanded * u, axis=1) / n

    # equivalent of np.matmul(u, u_T), but with u having the dimensions of (m,n,D,1)
    uuT = batch_outer_product_direct(u) #(m,n,D)

    diff2 = uuT - Hinv
    return diff2


    # get c_expanded into (m,n,D,D) shape
    c_expanded2 = np.empty(c_expanded.shape, dtype=c_expanded.dtype)
    for d in range(D):
        c_expanded2[:, :, :, d] = c_expanded

    cuuT = c_expanded*(uuT- Hinv)

    #cuuT = c_expanded*uuT
    return cuuT, Hinv

    #Hess = np.sum(cuuT - Hinv, axis=1) / n

    #return Hess



def others():
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

@numba.njit(parallel=False)
def matmul(S, M):
    # Get the shape of M
    m, n, D = M.shape

    # Create an output array to store the result
    result = np.zeros((m, n, D), dtype=M.dtype)

    # Perform the matrix multiplication using np.dot for efficiency
    for i in numba.prange(m):
        for j in range(n):
            # Extract the D-dimensional vector from M
            vector = M[i, j, :]  # Shape (D,)

            # Perform the matrix multiplication with S (D, D) and vector (D,)
            res_vector = np.dot(S, vector)  # Resulting shape (D,)

            # Store the result in the output array
            result[i, j, :] = res_vector

    return result


@numba.njit(parallel=False)
def batch_outer_product_direct(u_i):
    # Get the shape of u_i
    m, n, D = u_i.shape

    # Create an output array to store the result
    result = np.zeros((m, n, D, D), dtype=u_i.dtype)

    # Perform the batch outer product computation
    for i in numba.prange(m):
        for j in range(n):
            # Extract the vector of shape (D,)
            u_slice = u_i[i, j, :]  # Shape (D,)

            # Compute the outer product using np.outer, which results in a (D, D) matrix
            outer_result = np.outer(u_slice, u_slice)  # Shape (D, D)

            # Store the result
            result[i, j, :, :] = outer_result

    return result
