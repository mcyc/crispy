"""
Subspace Constrained Mean Shift (SCMS) algorithm for density ridge estimation.

This module provides functions to identify density ridges in high-dimensional data using the SCMS algorithm,
including support for parallel computation and filtering to improve efficiency. Core functionalities include
walker initialization, Gaussian kernel evaluation, ridge-shifting processes, and multiprocessing utilities.
"""

import numpy as np
from joblib import Parallel, delayed, cpu_count
import time
import sys
import gc

#======================================================================================================================#

def find_ridge(X, G, D=3, h=1, d=1, eps=1e-2, maxT=1000, weights=None, converge_frac=99, ncpu=None,
               return_unconverged=True, f_h=5):
    """
    Identify density ridges in data using the Subspace Constrained Mean Shift (SCMS) algorithm.

    Optimized for:
    - Memory efficiency: Avoid unnecessary copies of large arrays.
    - Computational efficiency: Reduce redundant operations in loops.
    - Parallelism: Efficient chunking and multiprocessing.
    """

    # Convert data to float32 efficiently (avoid unnecessary copying)
    G = G.astype(np.float32, copy=False)
    X = X.astype(np.float32, copy=False)
    h = np.float32(h)
    eps = np.float32(eps)

    # Handle weights efficiently
    weights = np.full(len(X), 1.0, dtype=np.float32) if weights is None else weights.astype(np.float32, copy=False)
    converge_frac = np.float32(converge_frac)

    n, m = len(X), len(G)
    t = 0

    # Preallocate error array (initialize to a large value)
    error = np.full(m, eps*1e2, dtype=np.float32)

    print("==========================================================================")
    print(f"Starting the run. Number of data points: {n}, Number of walkers: {m}")
    print("==========================================================================")

    # Start timing
    start_time = time.time()
    last_print_time = start_time

    # Get CPU count
    ncpu = cpu_count() if ncpu is None else ncpu

    # Compute initial convergence fraction
    pct_error = np.percentile(error, converge_frac)

    while pct_error > eps and t < maxT:
        t += 1

        # Identify unconverged walkers
        itermask = error > eps
        GjList = G[itermask]

        # Apply filtering to data points to optimize computations
        X, c, weights, dist = wgauss_n_filtered_points_multiproc(X, GjList, h, weights, f_h=f_h, ncpu=ncpu)

        ni, mi = len(X), len(GjList)

        # **Efficient logging**
        current_time = time.time()
        if current_time - last_print_time >= 1:  # Only print every second to reduce I/O overhead
            elapsed_time = current_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            sys.stdout.write(
                f"\rIteration {t} | Data points: {ni} | Walkers remaining: {mi}/{m} "
                f"({100 - mi / m * 100:.1f}% complete) | {converge_frac}-percentile error: {pct_error:.3f} | "
                f"Total run time: {formatted_time}")
            sys.stdout.flush()
            last_print_time = current_time
        # clear memory evey 10 seconds
        if current_time - last_print_time >= 5:
            gc.collect()

        # Perform walker shift in parallel
        GRes, errorRes = shift_wakers_multiproc(GjList, X, h, d, c, dist < f_h * h, ncpu)

        # Update G and error **in-place**
        G[itermask] = GRes
        error[itermask] = errorRes

        # Update error percentiles **efficiently**
        pct_error = np.percentile(error, converge_frac)

    sys.stdout.write("\n")
    print(f"Number of CPUs used: {ncpu}")

    # Determine which walkers converged
    mask = error < eps

    return (G[mask], G[~mask]) if return_unconverged else G[mask]


def wgauss_n_filtered_points(X, G, h, weights, f_h=5):
    """
    Compute weighted Gaussian values for data points relative to walker positions,
    filtering out distant points to optimize computation.

    This function calculates the Gaussian weights for data points (`X`) centered at walker
    positions (`G`) using a Gaussian kernel with bandwidth `h`. Data points farther than
    `f_h * h` from all walkers are excluded to reduce computational cost.

    Parameters
    ----------
    X : ndarray
        Coordinates of the data points, shape (n, D, 1), where `n` is the number of points
        and `D` is the dimensionality.
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1), where `m` is the number of walkers.
    h : float
        Smoothing bandwidth of the Gaussian kernel.
    weights : ndarray
        Weights of the data points, shape (n,).
    f_h : float, optional, default=5
        Distance multiplier cutoff for filtering points. Data points farther than
        `f_h * h` from all walkers are excluded.

    Returns
    -------
    X_filtered : ndarray
        Filtered coordinates of the data points, shape (k, D, 1), where `k` is the number of
        points that passed the filtering.
    c : ndarray
        Weighted Gaussian values for each data point, shape (k,).
    weights_filtered : ndarray
        Filtered weights corresponding to `X_filtered`, shape (k,).
    dist : ndarray
        Distances between remaining data points and walker positions, shape (m, k).

    Notes
    -----
    - The filtering step significantly reduces the number of data points to consider,
      which improves the efficiency of subsequent calculations.

    Examples
    --------
    Filter and compute Gaussian weights for a dataset:

    >>> import numpy as np
    >>> from crispy import scms
    >>> data = np.random.random((100, 3, 1))  # 3D data points
    >>> walkers = np.random.random((10, 3, 1))  # 3D walker positions
    >>> weights = np.ones(100)  # Equal weights for data points
    >>> X_filtered, c, weights_filtered, dist = scms.wgauss_n_filtered_points(data, walkers, h=0.5, weights=weights)
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
    inv_cov = 1 / (h ** 2)
    # Optimized computation of exponent
    exponent = -0.5 * np.einsum('ijk,ijk->ij', diff, diff) * inv_cov
    c = np.exp(exponent)

    return X, c * weights, weights, dist


def chunk_data(ncpu, data_list, data_size):
    """
    Divide data into chunks for multiprocessing.

    This function splits data into approximately equal-sized chunks to facilitate
    parallel processing across multiple CPUs.

    Parameters
    ----------
    ncpu : int
        Number of CPUs to use for parallel processing. If `ncpu` is negative, the entire
        dataset is treated as a single chunk.

    data_list : list of ndarray
        List of data arrays to be chunked. Each array should have the same size along
        the first axis (`data_size`).

    data_size : int
        The total number of data points (size of the first dimension of arrays in `data_list`).

    Returns
    -------
    chunks : tuple of lists of ndarray
        A tuple where each element corresponds to a list of chunks for a particular array in
        `data_list`. The total number of chunks is determined by `ncpu`.

    Notes
    -----
    - The function computes the chunk size as `data_size // ncpu` to ensure chunks are of
      approximately equal size.
    - If `ncpu` is negative, the entire dataset is returned as a single chunk.

    Examples
    --------
    Divide data into chunks for parallel processing:

    >>> import numpy as np
    >>> from crispy import scms
    >>> data1 = np.random.random((100, 3))  # Dataset 1
    >>> data2 = np.random.random((100, 3))  # Dataset 2
    >>> ncpu = 4
    >>> chunks = scms.chunk_data(ncpu, [data1, data2], data_size=100)
    >>> for chunk1, chunk2 in zip(*chunks):
    ...     print(chunk1.shape, chunk2.shape)
    """
    chunk_size = max(1, data_size // ncpu) if ncpu > 0 else data_size
    chunks = ()
    for data in data_list:
        chunks += ([data[i:i + chunk_size] for i in range(0, data_size, chunk_size)],)
    return chunks


def wgauss_n_filtered_points_multiproc(X, G, h, weights, f_h, ncpu=None):
    """
    Compute weighted Gaussian values for data points relative to walker positions
    in parallel, filtering out distant points to optimize computation.

    This function extends `wgauss_n_filtered_points` to support multiprocessing for
    efficient computation on large datasets.

    Parameters
    ----------
    X : ndarray
        Coordinates of the data points, shape (n, D, 1), where `n` is the number of points
        and `D` is the dimensionality.
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1), where `m` is the number of walkers.
    h : float
        Smoothing bandwidth of the Gaussian kernel.
    weights : ndarray
        Weights of the data points, shape (n,).
    f_h : float
        Distance multiplier cutoff for filtering points. Data points farther than
        `f_h * h` from all walkers are excluded.
    ncpu : int
        Number of CPUs to use for parallel processing. If set to `None`, defaults
        to the number of available CPUs.

    Returns
    -------
    X_filtered : ndarray
        Filtered coordinates of the data points, shape (k, D, 1), where `k` is the number of
        points that passed the filtering.
    c : ndarray
        Weighted Gaussian values for each filtered data point, shape (k,).
    weights_filtered : ndarray
        Filtered weights corresponding to `X_filtered`, shape (k,).
    dist : ndarray
        Distances between remaining data points and walker positions, shape (m, k).
    """
    if ncpu is None:
        ncpu = -1  # Use all available cores

    # Convert data to float32 only if necessary
    X = X.astype(np.float32, copy=False)
    G = G.astype(np.float32, copy=False)
    weights = weights.astype(np.float32, copy=False)
    h = np.float32(h)
    f_h = np.float32(f_h)

    # Efficiently split data
    X_chunks = np.array_split(X, ncpu)
    weights_chunks = np.array_split(weights, ncpu)

    # Parallel processing
    results = Parallel(n_jobs=ncpu)(
        delayed(wgauss_n_filtered_points)(X_chunk, G, h, weights_chunk, f_h)
        for X_chunk, weights_chunk in zip(X_chunks, weights_chunks)
    )

    # Extract results and concatenate efficiently
    X_filtered, c, weights_filtered, dist = zip(*results)

    X_filtered = np.concatenate(X_filtered, axis=0)
    c = np.hstack(c)  # Use hstack for better performance
    weights_filtered = np.concatenate(weights_filtered, axis=0)
    dist = np.hstack(dist)  # Use hstack for better performance

    return X_filtered, c, weights_filtered, dist


def shift_wakers_multiproc(G, X, h, d, c, mask, ncpu):
    """
    Shift walkers towards density ridges using the SCMS algorithm with multiprocessing.

    This function parallelizes the SCMS walker-shifting process for improved efficiency
    on large datasets. It divides the walkers into chunks and processes them concurrently
    across multiple CPUs.

    Parameters
    ----------
    G : ndarray
        Initial coordinates of the walkers, shape (m, D, 1), where `m` is the number of walkers
        and `D` is the dimensionality.

    X : ndarray
        Coordinates of the data points, shape (n, D, 1), where `n` is the number of data points.

    h : float
        Smoothing bandwidth for the Gaussian kernel.

    d : int
        Target dimensionality of the ridge subspace.

    c : ndarray
        Weighted Gaussian values computed for the data points and walkers, shape (m, n).

    mask : ndarray of bool
        Boolean mask indicating valid (True) data points for each walker. Shape is (m, n).

    ncpu : int
        Number of CPUs to use for parallel processing. If set to `None`, defaults to the number
        of available CPUs.

    Returns
    -------
    G_updated : ndarray
        Updated coordinates of the walkers after the SCMS shift, shape (m, D, 1).

    error : ndarray
        Convergence error for each walker, shape (m,). The error represents the displacement
        of each walker and is used to determine convergence.

    Notes
    -----
    - The walkers (`G`) are divided into chunks, and each chunk is processed independently
      on a separate CPU.
    - Internally, this function calls `shift_walkers` for each chunk, ensuring consistency
      with the SCMS algorithm.
    - Multiprocessing is particularly beneficial when the number of walkers or data points
      is large.

    Examples
    --------
    Perform a parallel SCMS shift for walkers:

    >>> import numpy as np
    >>> from crispy import scms
    >>> data = np.random.random((100, 3, 1))  # 3D data points
    >>> walkers = np.random.random((10, 3, 1))  # Initial walker positions
    >>> c = np.random.random((10, 100))  # Weighted Gaussian values
    >>> mask = np.random.choice([True, False], size=(10, 100))  # Boolean mask
    >>> h = 1.0
    >>> d = 1
    >>> ncpu = 4  # Use 4 CPUs
    >>> G_updated, error = scms.shift_wakers_multiproc(walkers, data, h, d, c, mask, ncpu)
    """
    ncpu = cpu_count() if ncpu is None else ncpu
    chunk_size = max(1, len(G) // ncpu)

    # Initialize storage with minimal memory allocations
    GRes = np.empty_like(G, dtype=np.float32)
    errorRes = np.empty(len(G), dtype=np.float32)

    # Define chunked data for parallel processing
    chunks = [
        (G[i:i + chunk_size], c[i:i + chunk_size], mask[i:i + chunk_size])
        for i in range(0, len(G), chunk_size)
    ]

    results = Parallel(n_jobs=ncpu)(
        delayed(shift_walkers)(G_chunk, X, h, d, c_chunk, mask_chunk)
        for G_chunk, c_chunk, mask_chunk in chunks
    )

    # Store results directly in preallocated arrays
    start = 0
    for res_G, res_error in results:
        end = start + len(res_G)
        GRes[start:end] = res_G
        errorRes[start:end] = res_error
        start = end

    return GRes, errorRes


def shift_walkers(G, X, h, d, c, mask):
    """
    Shift walkers towards density ridges using the Subspace Constrained Mean Shift (SCMS) algorithm.

    This function updates the positions of walkers (`G`) based on local density estimates
    from data points (`X`) and a Gaussian kernel with bandwidth `h`. The shift is constrained
    to the subspace defined by the eigenvectors of the Hessian matrix with the largest eigenvalues.

    Parameters
    ----------
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1), where `m` is the number of walkers
        and `D` is the dimensionality.

    X : ndarray
        Coordinates of the data points, shape (n, D, 1), where `n` is the number of points.

    h : float
        Smoothing bandwidth for the Gaussian kernel.

    d : int
        Target dimensionality of the ridge subspace.

    c : ndarray
        Weighted Gaussian values computed for the data points and walkers, shape (m, n).

    mask : ndarray of bool
        Boolean mask indicating valid (True) data points for each walker. Shape is (m, n).

    Returns
    -------
    G_updated : ndarray
        Updated coordinates of the walkers after the SCMS shift, shape (m, D, 1).

    error : ndarray
        Convergence error for each walker, shape (m,). The error represents the displacement
        of each walker and is used to determine convergence.

    Notes
    -----
    - The SCMS algorithm shifts walkers towards regions of high density and projects their
      movement onto the subspace spanned by the eigenvectors of the Hessian matrix with the
      largest eigenvalues.
    - The convergence error is calculated as the magnitude of the shift relative to the
      density gradient.

    Examples
    --------
    Perform a single SCMS shift for walkers:

    >>> import numpy as np
    >>> from crispy import scms
    >>> data = np.random.random((100, 3, 1))  # 3D data points
    >>> walkers = np.random.random((10, 3, 1))  # Walker positions
    >>> c = np.random.random((10, 100))  # Weighted Gaussian values
    >>> mask = np.random.choice([True, False], size=(10, 100))  # Boolean mask
    >>> h = 1.0
    >>> d = 1
    >>> G_updated, error = scms.shift_walkers(walkers, data, h, d, c, mask)
    """
    m, D = G.shape[0], G.shape[1]
    n = X.shape[0]

    # Compute internally to make parallel processing easier. They take incredibly little space
    H = np.eye(D, dtype=np.float32) * h ** 2
    Hinv = np.eye(D, dtype=np.float32) / h**2

    # Compute mean probability
    pj = np.mean(c, axis=1)[:, None, None]  # (m, 1, 1)

    # Compute u for selected elements only
    mask_indices = np.argwhere(mask)  # (k, 2) where k is the number of True values in mask
    diff_selected = G[mask_indices[:, 0]] - X[mask_indices[:, 1]]  # (k, D, 1)

    # Compute u directly without creating the intermediate step
    u_diff = np.einsum('ij,njk->nik', Hinv, diff_selected)  # (k, D, 1) #/h**2 still needed?

    # Compute g for selected walker points using the mask
    c_selected = c[mask_indices[:, 0], mask_indices[:, 1]]  # (k,)
    g = np.zeros_like(G, dtype=np.float64)  # (m, D, 1) # use dtype=np.float32 breaks the code for some reasons
    np.add.at(g, mask_indices[:, 0], -c_selected[:, None, None] * u_diff / n)

    # Compute u_diff.T @ u_diff using einsum to avoid explicit transpose and matmul
    product = np.einsum('nik,njk->nij', u_diff, u_diff) - Hinv  # (k, D, D)

    # Update Hessian matrix
    Hess = np.zeros((m, D, D), dtype=np.float32)  # (m, D, D)
    np.add.at(Hess, mask_indices[:, 0], c_selected[:, None, None] * product / n)

    # Compute Sigmainv
    Sigmainv = (-Hess + np.einsum('mik,mil->mkl', g, g)/pj)/pj  # (m, D, D)

    # Compute the shift for each walker
    shift0 = G + np.einsum('ij,mjk->mik', H, g) / pj # (m, D, 1)

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
    Shift walkers toward density ridges using the Subspace Constrained Mean Shift (SCMS) algorithm.

    This function updates the positions of walkers (`G`) based on local density estimates
    computed from data points (`X`) and projects their movement onto the subspace of
    interest, defined by eigenvectors of the Hessian matrix.

    Parameters
    ----------
    G : ndarray
        Initial coordinates of the walkers, shape (m, D, 1), where `m` is the number of walkers
        and `D` is the dimensionality.

    X : ndarray
        Coordinates of the data points, shape (n, D, 1), where `n` is the number of points.

    D : int
        Dimensionality of the data points.

    h : float
        Smoothing bandwidth of the Gaussian kernel.

    d : int
        Target dimensionality of the ridge subspace.

    c : ndarray
        Weighted Gaussian values computed for the data points and walkers, shape (m, n).

    n : int
        Number of data points (`n = X.shape[0]`).

    H : ndarray
        Covariance matrix for the Gaussian kernel, shape (D, D).

    Hinv : ndarray
        Inverse of the covariance matrix, shape (D, D).

    Returns
    -------
    G_updated : ndarray
        Updated coordinates of the walkers, shape (m, D, 1).

    error : ndarray
        Convergence error for each walker, shape (m,). The error represents the displacement
        of each walker and is used to determine convergence.

    Notes
    -----
    - The SCMS algorithm shifts walkers toward regions of high density by iteratively
      estimating gradients and projecting movements onto the ridge subspace.
    - The eigen decomposition of the Hessian matrix is used to constrain movement to the
      subspace defined by the largest eigenvalues.
    - The convergence error is computed as the displacement magnitude of each walker
      relative to the density gradient.

    Examples
    --------
    Perform a single SCMS shift for walkers:

    >>> import numpy as np
    >>> from crispy import scms
    >>> data = np.random.random((100, 3, 1))  # 3D data points
    >>> walkers = np.random.random((10, 3, 1))  # Walker positions
    >>> h = 1.0
    >>> d = 1
    >>> c = np.random.random((10, 100))  # Weighted Gaussian values
    >>> n = data.shape[0]
    >>> H = np.eye(3) * h**2  # Covariance matrix
    >>> Hinv = np.linalg.inv(H)  # Inverse covariance matrix
    >>> G_updated, error = scms.shift_particles(walkers, data, D=3, h=h, d=d, c=c, n=n, H=H, Hinv=Hinv)
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
    Compute the Euclidean distances and differences between data points and walkers.

    This function calculates pairwise Euclidean distances between points in `X` and `G` and
    returns both the distances and the differences in their coordinates.

    Parameters
    ----------
    X : ndarray
        Coordinates of the data points, shape (n, D, 1), where `n` is the number of points
        and `D` is the dimensionality.
    G : ndarray
        Coordinates of the walkers, shape (m, D, 1), where `m` is the number of walkers.

    Returns
    -------
    distances : ndarray
        Pairwise Euclidean distances between each point in `G` and each point in `X`,
        shape (m, n).
    diff : ndarray
        Pairwise coordinate differences between points in `G` and points in `X`,
        shape (m, n, D).

    Notes
    -----
    - This function is useful for calculating distances and displacements required
      in SCMS-based ridge detection.
    """
    X_squeezed = np.squeeze(X, axis=-1)
    G_squeezed = np.squeeze(G, axis=-1)

    diff = G_squeezed[:, np.newaxis, :] - X_squeezed[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    return distances, diff


def vectorized_gaussian(X, G, h):
    """
    Compute Gaussian kernel values for data points relative to walker positions.

    This function calculates the Gaussian kernel values for each pair of points in `X` and
    `G`, based on the Euclidean distances between them, and returns both the kernel values
    and the distances.

    Parameters
    ----------
    X : ndarray
        Coordinates of the data points, shape (n, D, 1), where `n` is the number of points
        and `D` is the dimensionality.

    G : ndarray
        Mean positions (walker coordinates) for the Gaussian kernel, shape (m, D, 1),
        where `m` is the number of walkers.

    h : float
        Smoothing bandwidth of the Gaussian kernel.

    Returns
    -------
    c : ndarray
        Gaussian kernel values for each pair of data point and walker, shape (m, n).

    distances : ndarray
        Pairwise Euclidean distances between each walker in `G` and each point in `X`,
        shape (m, n).

    Notes
    -----
    This function is optimized to handle pairwise distance calculations and kernel
    evaluations efficiently.

    Examples
    --------
    Compute Gaussian kernel values and distances:

    >>> import numpy as np
    >>> from crispy import scms
    >>> data = np.random.random((100, 3, 1))  # 3D data points
    >>> walkers = np.random.random((10, 3, 1))  # Walker positions
    >>> h = 1.0  # Bandwidth
    >>> c, distances = scms.vectorized_gaussian(data, walkers, h)
    >>> print(c.shape)  # Should be (10, 100)
    >>> print(distances.shape)  # Should be (10, 100)
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