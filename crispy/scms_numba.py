import numpy as np

def shift_particle(Gj, X, D, h, d, weights, n, H, Hinv):
    # shift individual walkers using SCMS
    c = np.zeros(weights.shape, dtype=weights.dtype) #(n,)
    cu = np.zeros(X.shape, dtype=weights.dtype) #(n, D, 1)
    hh = np.zeros((n, D, D), dtype=weights.dtype) #(n,D,D)

    for i, Xi in enumerate(X):  # Xi has shape (D,)
        # Evaluate the Gaussian value of Xi at Gj
        c_i = np.exp(gaussian(Xi, mean=Gj, covariance=h ** 2))  # c is a scalar

        # Now weight the probability of Xi by the image
        c[i] = c_i* weights[i]  # c is a scalar

        # Compute u, which is used for shifting
        u_i = np.dot(Hinv, (Gj - Xi)) / h**2  # u has shape (D, 1)
        cu[i] = c[i]*u_i # (D,1)
        hh[i] = c[i]*np.dot(u_i, np.transpose(u_i)) - Hinv #(D,D)

    pj = np.mean(c)  # pj is a scalar
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
    V = EigVec[:, d:D]  # V has shape (D, D-d)

    # Compute V * V^T
    VT = np.transpose(V)
    VVT = np.dot(V, np.transpose(V))  # VVT has shape (D, D)

    # Update Gj based on the projection
    Gj = np.dot(VVT, shift0 - Gj) + Gj  # Gj has shape (D,)

    # Compute error
    tmp = np.dot(VT, g)  # tmp has shape (D-d,)
    errorj = np.sqrt(np.sum(tmp ** 2) / np.sum(g ** 2))  # errorj is a scalar

    return Gj, errorj, pj, g, shift0


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