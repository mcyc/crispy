import numpy as np
from joblib import Parallel, delayed, cpu_count
import time
import sys

#======================================================================================================================#

def find_ridge(X, G, D=3, h=1, d=1, eps=1e-06, maxT=1000, weights=None, converge_frac=99, ncpu=None,
               return_unconverged=True, f_h=5):
    """
    Finds the density ridge in the data by iteratively moving walkers towards the ridge.

    Parameters:
    X : ndarray
        Coordinates of the data points, shape (n, D, 1).
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1).
    D : int, optional
        Dimension of the data (default is 3).
    h : float, optional
        Smoothing bandwidth for the Gaussian kernel (default is 1).
    d : int, optional
        Number of dimensions for projection (default is 1).
    eps : float, optional
        Convergence criteria for individual walkers (default is 1e-06).
    maxT : int, optional
        Maximum number of iterations (default is 1000).
    weights : ndarray, optional
        Weight of each data point, or value at each pixel if dealing with images (default is None).
    converge_frac : float, optional
        Fraction of walkers that need to converge for the algorithm to stop, in percent (default is 99).
    ncpu : int, optional
        Number of CPUs to use (default of None means use all the cpus).
    return_unconverged : bool, optional
        If True, returns both converged and unconverged walkers (default is True).
    f_h : float, optional
        Factor used to filter out data points based on distance to all other data points (default is 5).

    Returns:
    G_converged : ndarray
        Coordinates of converged walkers.
    G_unconverged : ndarray, optional
        Coordinates of unconverged walkers, if return_unconverged is True.
    """

    # Convert data to float32 for efficiency
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

    error = np.full(m, 1e+08, dtype=np.float32)

    print("==========================================================================")
    print(f"Starting the run. Number of data points: {n}, Number of walkers: {m}")
    print("==========================================================================")

    # Start timing
    start_time = time.time()
    last_print_time = start_time

    pct_error = np.percentile(error, converge_frac)

    while ((pct_error > eps) & (t < maxT)):
        # Loop through iterations
        t += 1

        itermask = error > eps
        GjList = G[itermask]

        # Filter out data points too far away to save computation time
        X, c, weights, dist = wgauss_n_filtered_points(X, GjList, h, weights, f_h=f_h)
        mask = dist < f_h * h

        ni, mi = len(X), len(GjList)

        current_time = time.time()
        if current_time - last_print_time >= 1:
            elapsed_time = current_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            sys.stdout.write(
                f"\rIteration {t} | Data points: {ni} | Walkers remaining: {mi}/{m} ({100 - mi / m * 100:0.1f}% complete) | {converge_frac}-percentile error: {pct_error:0.3f} | Total run time: {formatted_time}")
            sys.stdout.flush()

        GRes, errorRes = shift_wakers_multiproc(GjList, X, h, d, c, mask, ncpu)
        G[itermask], error[itermask]  = GRes, errorRes

        pct_error = np.percentile(error, converge_frac)

    sys.stdout.write("\n")
    mask = error < eps

    ncpu = cpu_count() if ncpu is None else ncpu
    print("the number of cpu used: ", ncpu)

    if return_unconverged:
        return G[mask], G[~mask]
    else:
        return G[mask]


def wgauss_n_filtered_points(X, G, h, weights, f_h=5):
    """
    Computes the weighted Gaussian at positions X, with means in G, filtering out points based on distance.

    Parameters:
    X : ndarray
        Coordinates of the data points, shape (n, D, 1).
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1).
    h : float
        Smoothing bandwidth of the Gaussian kernel.
    weights : ndarray
        Weights of the data points.
    f_h : float, optional
        Distance multiple cutoff for filtering points (default is 8).

    Returns:
    X_filtered : ndarray
        Filtered coordinates of the data points.
    c : ndarray
        Weighted Gaussian values for each data point.
    weights_filtered : ndarray
        Filtered weights for the data points.
    """

    # Find data points that are too far from walkers
    dist, diff = euclidean_dist(X, G)
    toofar = np.all(dist > f_h * h, axis=0)

    # Filter out distant data
    X = X[~toofar, :, :]
    diff = diff[:, ~toofar, :]
    dist = dist[:, ~toofar]
    weights = weights[~toofar]

    # Calculate the Gaussian values
    inv_cov = 1 / (h**2)
    exponent = -0.5 * np.sum(diff**2 * inv_cov, axis=-1)
    c = np.exp(exponent)

    return X, c * weights, weights, dist


    # Use available_cpus as the default if ncpu is -1
def chunk_data(ncpu, data_list, data_size):
    # break data up into chunks for multiprocessing
    ncpu = cpu_count() if ncpu is None else ncpu
    chunk_size = max(1, data_size // ncpu) if ncpu > 0 else data_size
    chunks = ()
    for data in data_list:
        chunks += ([data[i:i + chunk_size] for i in range(0, data_size, chunk_size)],)
    return chunks

def shift_wakers_multiproc(G, X, h, d, c, mask, ncpu):
    # multiprocessing wrapper for shift_walkers

    # Split GjList into chunks for parallel processing
    chunks = chunk_data(ncpu, [G, c, mask], len(G))

    results = Parallel(n_jobs=ncpu)(delayed(shift_walkers)(G_chunk, X, h, d, c_chunk, mask_chunk)
                                    for G_chunk, c_chunk, mask_chunk in zip(*chunks))
    GRes, errorRes = zip(*results)
    GRes, errorRes = np.concatenate(GRes, axis=0), np.concatenate(errorRes, axis=0)
    return GRes, errorRes


def shift_walkers(G, X, h, d, c, mask):
    """
    Shifts the walkers in G towards the density ridges using SCMS,
    excluding elements outside the mask

    Parameters:
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1).
    X : ndarray
        Coordinates of the data points, shape (n, D, 1).
    h : float
        Smoothing bandwidth of the Gaussian kernel.
    d : int
        Target number of dimensions for projection.
    c : ndarray
        Weighted Gaussian values computed from G and X.

    Returns:
    G_updated : ndarray
        Updated coordinates of the walkers.
    error : ndarray
        Error term for each walker.
    """

    m, D = G.shape[0], G.shape[1]
    n = X.shape[0]

    # Compute internally to make parallel processing easier. They take incredibly little space
    H = np.eye(D) * h**2
    Hinv = np.eye(D) / h**2

    # Compute mean probability
    pj = np.mean(c, axis=1)  # (m,)

    # Compute u for selected elements only
    mask_indices = np.argwhere(mask)  # (k, 2) where k is the number of True values in mask
    diff_selected = G[mask_indices[:, 0]] - X[mask_indices[:, 1]]  # (k, D, 1)

    # Compute u directly without creating the intermediate step
    u_diff = np.einsum('ij,njk->nik', Hinv, diff_selected)  # (k, D, 1) #/h**2 still needed?

    # Compute g for selected walker points using the mask
    c_selected = c[mask_indices[:, 0], mask_indices[:, 1]]  # (k,)
    g = np.zeros(G.shape)  # (m, D, 1)
    np.add.at(g, mask_indices[:, 0], -1 * c_selected[:, None, None] * u_diff / n)

    # Compute u_diff.T @ u_diff using einsum to avoid explicit transpose and matmul
    product = np.einsum('nik,njk->nij', u_diff, u_diff) - Hinv  # (k, D, D)

    # Update Hessian matrix
    Hess = np.zeros((m, D, D))  # (m, D, D)
    np.add.at(Hess, mask_indices[:, 0], c_selected[:, None, None] * product / n)

    # Expand dimensions for pj
    pj = pj[:, None, None]  # (m, 1, 1)

    # Compute Sigmainv
    Sigmainv = -1 * Hess/pj + np.einsum('mik,mil->mkl', g, g)/pj** 2  # (m, D, D)

    # Compute the shift for each walker
    shift0 = G + np.matmul(H, g)/pj  # (m, D, 1)
    #shift0 = G + np.einsum('ij,jk->ik', H, g) / pj

    # Eigen decomposition for Sigmainv
    EigVal, EigVec = np.linalg.eigh(Sigmainv) # (m, D), (m, D, D)

    # Get the eigenvectors with the largest eigenvalues
    V = EigVec[:, :, d:D]  # (m, D, D-d)

    # Update G for each walker
    G = np.einsum('mij,mjk->mik', np.einsum('mik,mjk->mij', V, V), (shift0 - G)) + G  # (m, D, 1)

    # Compute the error term
    tmp = np.einsum('mji,mjk->mik', V, g)  # (m, D, 1)
    error = np.sqrt(np.einsum('mik,mik->m', tmp, tmp) / np.einsum('mik,mik->m', g, g))  # (m,)

    return G, error


def shift_particles(G, X, D, h, d, c, n, H, Hinv):
    """
    Shifts the walkers in G towards the density ridges using SCMS.

    Parameters:
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1).
    X : ndarray
        Coordinates of the data points, shape (n, D, 1).
    D : int
        Dimension of the data points.
    h : float
        Smoothing bandwidth of the Gaussian kernel.
    d : int
        Target number of dimensions for projection.
    c : ndarray
        Weighted Gaussian values computed from G and X.
    n : int
        Number of data points.
    H : ndarray
        Covariance matrix for the Gaussian kernel.
    Hinv : ndarray
        Inverse of the covariance matrix.

    Returns:
    G_updated : ndarray
        Updated coordinates of the walkers.
    error : ndarray
        Error term for each walker.
    """

    # Compute mean probability
    pj = np.mean(c, axis=1)

    # Expand dimensions for broadcasting
    X_expanded = X[None, :, :, :] # (n, 1, D, 1)
    G_expanded = G[:, None, :, :] # (1, m, D, 1)

    # Compute u for all walker points
    u = np.matmul(Hinv, (G_expanded - X_expanded)) #/ h**2

    # Compute g for all walker points
    c_expanded = c[:, :, None, None]
    g = -1 * np.sum(c_expanded * u, axis=1) / n

    # Compute the Hessian matrix for all walker points
    u_T = np.transpose(u, axes=(0, 1, 3, 2))
    Hess = np.sum(c_expanded * (np.matmul(u, u_T) - Hinv), axis=1) / n

    # Expand dimensions for pj
    pj = pj[:, None, None]

    Sigmainv = -1 * Hess / pj + np.matmul(g, np.transpose(g, axes=(0, 2, 1))) / pj**2

    # Compute the shift for each walker
    shift0 = G + np.matmul(H, g) / pj

    # Eigen decomposition for Sigmainv
    EigVal, EigVec = np.linalg.eigh(Sigmainv)

    # Get the eigenvectors with the largest eigenvalues
    V = EigVec[:, :, d:D]

    # Compute VVT
    VVT = np.matmul(V, np.transpose(V, axes=(0, 2, 1)))

    # Update G for each walker
    G = np.matmul(VVT, (shift0 - G)) + G

    # Compute the error term
    tmp = np.matmul(np.transpose(V, axes=(0, 2, 1)), g)
    error = np.sqrt(np.sum(tmp**2, axis=(1, 2)) / np.sum(g**2, axis=(1, 2)))

    return G, error


def euclidean_dist(X, G):
    """
    Computes the Euclidean distances between points in X and G.

    Parameters:
    X : ndarray
        Coordinates of the data points, shape (n, D, 1).
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1).

    Returns:
    distances : ndarray
        Euclidean distances between each pair of points in X and G, shape (m, n).
    diff : ndarray
        Pairwise differences between each point in G and each point in X, shape (m, n, D).
    """
    X_squeezed = np.squeeze(X, axis=-1)
    G_squeezed = np.squeeze(G, axis=-1)

    # Compute differences for all combinations of G and X
    diff = G_squeezed[:, np.newaxis, :] - X_squeezed[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    return distances, diff


def vectorized_gaussian(X, G, h):
    """
    Computes the Gaussian exponential for each point in X for each mean position in G.

    Parameters:
    X : ndarray
        Coordinates of the data points, shape (n, D, 1).
    G : ndarray
        Mean positions for the Gaussian, shape (m, D, 1).
    h : float
        Smoothing bandwidth of the Gaussian kernel.

    Returns:
    c : ndarray
        Gaussian exponentials computed for each pair of points in X and G, shape (m, n).
    distances : ndarray
        Euclidean distances between each walker in G and each point in X, shape (m, n).
    """
    # Calculate Euclidean distances between each walker in G and each point in X
    X_squeezed = np.squeeze(X, axis=-1)
    G_squeezed = np.squeeze(G, axis=-1)

    # Compute differences for all combinations of G and X
    diff = G_squeezed[:, np.newaxis, :] - X_squeezed[np.newaxis, :, :]

    # Compute the inverse covariance (assumes h is scalar)
    inv_cov = 1 / (h**2)

    # Calculate the exponent for the Gaussian function
    exponent = -0.5 * np.sum(diff**2 * inv_cov, axis=-1)

    # Compute the Gaussian exponential
    c = np.exp(exponent)

    return c