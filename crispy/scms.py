import numpy as np
try:
    from cuml.neighbors import KernelDensity
    gpu_available = True
except ImportError:
    from sklearn.neighbors import KernelDensity
    gpu_available = False
from cuml.metrics import pairwise_distances
from multiprocessing import Pool


def find_ridge(data, bandwidth=1.0, max_iter=100, tol=1e-5, n_jobs=1, wweights=None, return_unconverged=False):
    """
    Find ridges in the given data using the Subspace Constrained Mean Shift (SCMS) algorithm.

    Parameters:
    data (X) : array-like, shape (n_samples, n_features)
        The input data points. Equivalent to `X` in the original implementation.
    bandwidth (h) : float, optional, default=1.0
        The bandwidth parameter for the kernel density estimation. Equivalent to `h` in the original implementation.
    max_iter (maxT) : int, optional, default=100
        The maximum number of iterations for the SCMS algorithm. Equivalent to `maxT` in the original implementation.
    tol (converge_frac) : float, optional, default=1e-5
        The tolerance for convergence. The algorithm stops if the change in walker positions is below this value.
         Equivalent to `converge_frac` in the original implementation.
    n_jobs (ncpu) : int, optional, default=1
        The number of jobs to run in parallel for shifting walkers. Equivalent to `ncpu` in the original implementation.
    wweights : array-like, shape (n_samples,), optional, default=None
        Weights for each data point. If None, equal weights are assumed. Equivalent to `wweights` in
         the original implementation.
    return_unconverged : bool, optional, default=False
        If True, returns the data points that did not converge within the maximum number of iterations.
         Equivalent to `return_unconverged` in the original implementation.

    Returns:
    walkers (G, D) : array-like, shape (n_samples, n_features)
        The converged positions of the data points (walkers) after applying SCMS. Equivalent to `G` and
         `D` in the original implementation.
    unconverged : array-like, shape (n_unconverged, n_features), optional
        The positions of the data points that did not converge if return_unconverged is True.
    """
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(data)

    walkers = data.copy()
    prev_walkers = np.zeros_like(walkers)

    if wweights is None:
        wweights = np.ones(len(data))

    for _ in range(max_iter):
        if np.linalg.norm(walkers - prev_walkers) < tol:
            break
        prev_walkers = walkers.copy()

        with Pool(n_jobs) as pool:
            walkers = np.vstack(pool.map(lambda x: shift_particle(x, kde, wweights), walkers))

    if return_unconverged:
        converged_mask = np.linalg.norm(walkers - prev_walkers, axis=1) < tol
        unconverged = walkers[~converged_mask]
        return walkers, unconverged

    return walkers


def shift_particle(particle, kde, wweights):
    """
    Shift a single particle along the ridge using the SCMS algorithm.

    Parameters:
    particle : array-like, shape (n_features,)
        The position of the particle to be shifted.
    kde : cuml.neighbors.KernelDensity or sklearn.neighbors.KernelDensity
        The kernel density estimator fitted to the data.
    wweights : array-like, shape (n_samples,)
        Weights for each data point.

    Returns:
    new_position : array-like, shape (n_features,)
        The new position of the particle after shifting.
    """
    grad = compute_gradient(particle, kde, wweights)
    hessian_inv = np.linalg.pinv(compute_hessian(particle, kde, wweights))
    return particle + hessian_inv @ grad


def compute_gradient(particle, kde, wweights):
    """
    Compute the gradient of the kernel density estimate at a given particle position.

    Parameters:
    particle : array-like, shape (n_features,)
        The position of the particle.
    kde : cuml.neighbors.KernelDensity or sklearn.neighbors.KernelDensity
        The kernel density estimator fitted to the data.
    wweights : array-like, shape (n_samples,)
        Weights for each data point.

    Returns:
    grad : array-like, shape (n_features,)
        The gradient vector of the density estimate at the particle's position.
    """
    eps = 1e-5
    d = len(particle)
    grad = np.zeros(d)

    for i in range(d):
        step = np.zeros(d)
        step[i] = eps
        diff = (kde.score_samples([particle + step]) - kde.score_samples([particle - step])) * wweights[i]
        grad[i] = diff / (2 * eps)

    return grad


def compute_hessian(particle, kde, wweights):
    """
    Compute the Hessian matrix of the kernel density estimate at a given particle position.

    Parameters:
    particle : array-like, shape (n_features,)
        The position of the particle.
    kde : cuml.neighbors.KernelDensity or sklearn.neighbors.KernelDensity
        The kernel density estimator fitted to the data.
    wweights : array-like, shape (n_samples,)
        Weights for each data point.

    Returns:
    hessian : array-like, shape (n_features, n_features)
        The Hessian matrix of the density estimate at the particle's position.
    """
    eps = 1e-5
    d = len(particle)
    hessian = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            step_i = np.zeros(d)
            step_j = np.zeros(d)
            step_i[i] = eps
            step_j[j] = eps

            f_ij = kde.score_samples([particle + step_i + step_j]) * wweights[i] * wweights[j]
            f_i = kde.score_samples([particle + step_i - step_j]) * wweights[i]
            f_j = kde.score_samples([particle - step_i + step_j]) * wweights[j]
            f_ij_neg = kde.score_samples([particle - step_i - step_j])

            hessian[i, j] = (f_ij - f_i - f_j + f_ij_neg) / (4 * eps ** 2)

    return hessian