import numpy as np
from skimage import io, filters
from scms import find_ridge


def image_ridge_find(image_path, bandwidth=1.0, max_iter=100, tol=1e-5, n_jobs=1, wweights=None, return_unconverged=False):
    """
    Identify ridges in an image using the Subspace Constrained Mean Shift (SCMS) algorithm.

    Parameters:
    image_path : str
        Path to the input image file.
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
        Weights for each data point. If None, equal weights are assumed. Equivalent to `wweights` in the
        original implementation.
    return_unconverged : bool, optional, default=False
        If True, returns the data points that did not converge within the maximum number of iterations.
         Equivalent to `return_unconverged` in the original implementation.

    Returns:
    ridges : array-like, shape (n_samples, 2)
        The coordinates of the ridge points found in the image.
    unconverged : array-like, shape (n_unconverged, 2), optional
        The positions of the data points that did not converge if return_unconverged is True.
    """
    # Load and preprocess the image
    image = io.imread(image_path, as_gray=True)
    edges = filters.sobel(image)
    y, x = np.nonzero(edges)
    data = np.vstack((x, y)).T

    # Find ridges using SCMS
    if return_unconverged:
        ridges, unconverged = find_ridge(data, bandwidth=bandwidth, max_iter=max_iter, tol=tol, n_jobs=n_jobs, wweights=wweights, return_unconverged=True)
        return ridges, unconverged
    else:
        ridges = find_ridge(data, bandwidth=bandwidth, max_iter=max_iter, tol=tol, n_jobs=n_jobs, wweights=wweights, return_unconverged=False)
        return ridges


# Example usage
if __name__ == "__main__":
    ridges = image_ridge_find("path/to/image.png", bandwidth=1.5, max_iter=150, tol=1e-4, n_jobs=4)
    print("Ridge points found:", ridges)