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

import functools
import tracemalloc

def peak_memory_usage(func):
    """Decorator to measure and print the peak memory usage (in MB) of a function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"[{func.__name__}] Peak memory usage: {peak / (1024 * 1024):.2f} MB")

        return result

    return wrapper


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

        # Inside your find_ridge() loop:
        # (Assuming you want to use the boolean mask version to pass into shift_wakers_multiproc.)
        X, c, weights, mask = wgauss_n_filtered_points_multiproc(
            X, GjList, h, weights, f_h=f_h, ncpu=ncpu, return_distances=False
        )


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
        GRes, errorRes = shift_wakers_multiproc(GjList, X, h, d, c, mask, ncpu)

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


def wgauss_n_filtered_points_multiproc(X, G, h, weights, f_h, ncpu=None,
                                             target_chunk_size=1000, min_chunk_size=100,
                                             return_distances=False):
    """
    Multiprocessing wrapper for wgauss_n_filtered_points_opt.

    The function splits X (and weights) into chunks to limit memory usage
    when computing the full pairwise differences, then runs the filtering
    in parallel. The number of chunks is determined by both the target chunk
    size and the number of CPUs.

    Parameters
    ----------
    X : ndarray
        Data points, shape (n, D, 1).
    G : ndarray
        Walker positions, shape (m, D, 1).
    h : float
        Gaussian kernel bandwidth.
    weights : ndarray
        Weights for the data points, shape (n,).
    f_h : float
        Cutoff multiplier for filtering.
    ncpu : int, optional
        Number of CPUs to use (defaults to all available).
    target_chunk_size : int, optional
        Target number of data points per chunk.
    min_chunk_size : int, optional
        Minimum chunk size to avoid over‑splitting.
    return_distances : bool, optional
        Passed to wgauss_n_filtered_points_opt (see its docstring).

    Returns
    -------
    X_filtered : ndarray
        Filtered data points.
    c : ndarray
        Weighted Gaussian values for the filtered points.
    weights_filtered : ndarray
        Filtered weights.
    out : ndarray
        Either distances or a boolean mask (see wgauss_n_filtered_points_opt).
    """
    if ncpu is None:
        ncpu = cpu_count()

    n_points = X.shape[0]*G.shape[0]
    target_chunk_size = target_chunk_size**2
    # Determine number of chunks; ensure at least ncpu chunks but avoid too many small chunks.
    num_chunks = max(ncpu, n_points // target_chunk_size)
    num_chunks = min(num_chunks, n_points // min_chunk_size**2) if n_points >= target_chunk_size else 1
    #print(f"n chunks {num_chunks}")
    # Split X and weights into roughly equal chunks
    X_chunks = np.array_split(X, num_chunks)
    weights_chunks = np.array_split(weights, num_chunks)

    results = Parallel(n_jobs=ncpu)(
        delayed(wgauss_n_filtered_points)(X_chunk, G, h, w_chunk, f_h, return_distances)
        for X_chunk, w_chunk in zip(X_chunks, weights_chunks)
    )

    # Unpack and efficiently concatenate results
    X_filtered_list, c_list, weights_filtered_list, out_list = zip(*results)
    X_filtered = np.concatenate(X_filtered_list, axis=0)
    # For c, if each chunk’s result is (m, k_chunk), hstack along the second axis gives (m, total_k)
    c = np.hstack(c_list)
    weights_filtered = np.concatenate(weights_filtered_list, axis=0)
    # out (distances or mask) will be hstacked similarly.
    out = np.hstack(out_list)

    return X_filtered, c, weights_filtered, out


def wgauss_n_filtered_points(X, G, h, weights, f_h=5, return_distances=False):
    """
    Optimized weighted Gaussian evaluation and filtering.

    This version avoids computing the square-root (which saves both time and memory)
    by working with squared distances. It also squeezes out the trailing singleton
    dimension so that temporary arrays are smaller.

    Parameters
    ----------
    X : ndarray
        Data points, shape (n, D, 1).
    G : ndarray
        Walker positions, shape (m, D, 1).
    h : float
        Gaussian kernel bandwidth.
    weights : ndarray
        Weights of the data points, shape (n,).
    f_h : float, optional
        Multiplier for the bandwidth cutoff.
    return_distances : bool, optional
        If True, return the distances (square-root of squared distances); otherwise,
        return a boolean mask (which uses much less memory).

    Returns
    -------
    X_filtered : ndarray
        Filtered data points (only those within f_h * h of at least one walker).
    c : ndarray
        Weighted Gaussian values (each multiplied by the corresponding weight).
    weights_filtered : ndarray
        Filtered weights corresponding to X_filtered.
    out : ndarray
        Either the distances (if return_distances=True) or a boolean mask (if False)
        indicating, for each walker × filtered point, whether that point is within
        f_h * h.
    """
    # Work with 2D views (n, D) and (m, D)
    Xs = np.squeeze(X, axis=-1)  # shape: (n, D)
    Gs = np.squeeze(G, axis=-1)  # shape: (m, D)

    # Compute squared Euclidean distances (without computing sqrt)
    squared_diff = np.sum((Gs[:, None, :] - Xs[None, :, :])**2, axis=-1)  # shape: (m, n)

    # Identify points that are “too far” from all walkers.
    # (For each data point, if every walker is farther than the cutoff, mark it as too far.)
    keep = ~np.all(squared_diff > (f_h * h) ** 2, axis=0)  # shape: (n,)

    # Filter the data and corresponding weights
    weights_filtered = weights[keep]  # shape: (k,)

    # Also filter the computed differences and squared distances along the data axis.
    squared_diff_filtered = squared_diff[:, keep]  # shape: (m, k)

    # Compute Gaussian kernel values (using squared distances)
    inv_cov = 1 / (h ** 2)
    c = np.exp(-0.5 * squared_diff_filtered * inv_cov)*weights_filtered  # shape: (m, k)

    # Prepare output: either distances or just a boolean mask
    if return_distances:
        # (This does require an extra sqrt computation; if not needed, set return_distances=False)
        distances = np.sqrt(squared_diff_filtered)
        out = distances
    else:
        # A boolean mask: True if the (squared) distance is below the cutoff
        out = squared_diff_filtered < (f_h * h) ** 2

    return X[keep], c, weights_filtered, out


def shift_wakers_multiproc(G, X, h, d, c, mask, ncpu=None, target_chunk_size=5000, min_chunk_size=500):
    """
    Optimized Parallel Walker Shifting.

    - Uses `joblib.Parallel` for multiprocessing.
    - **Chunks `G` (walkers), not `X`**, ensuring full visibility of `X`.
    - Preallocates `GRes` and `errorRes` to **avoid excessive memory operations**.
    - Uses **adaptive chunking** to **balance memory usage and CPU workload**.

    Parameters
    ----------
    G : ndarray
        Initial walker positions, shape `(m, D, 1)`.
    X : ndarray
        Data points, shape `(n, D, 1)`.
    h : float
        Smoothing bandwidth.
    d : int
        Target ridge subspace dimensionality.
    c : ndarray
        Weighted Gaussian values for `X` and `G`, shape `(m, n)`.
    mask : ndarray
        Boolean mask indicating valid data points for each walker, shape `(m, n)`.
    ncpu : int, optional
        Number of CPUs for parallel processing.
    target_chunk_size : int, optional
        Target number of computations per chunk (default: `5000`).
    min_chunk_size : int, optional
        Minimum number of walkers per chunk (default: `500`).

    Returns
    -------
    G_updated : ndarray
        Updated walker positions after shifting, shape `(m, D, 1)`.
    error : ndarray
        Convergence error for each walker, shape `(m,)`.
    """

    if ncpu is None:
        ncpu = -1  # Use all available CPU cores

    # Convert to float32 for efficiency
    G = G.astype(np.float32, copy=False)
    X = X.astype(np.float32, copy=False)
    c = c.astype(np.float32, copy=False)
    mask = mask.astype(bool, copy=False)
    h = np.float32(h)

    m, D = G.shape[0], G.shape[1]

    # **Compute the number of chunks based on `m × n`**
    total_size = m * X.shape[0]  # Total number of pairwise operations
    num_chunks = max(total_size // (target_chunk_size**2), ncpu)  # Ensure at least `ncpu` chunks

    # **Ensure chunk size is reasonable**
    num_chunks = max(num_chunks, m // min_chunk_size)  # Prevent over-splitting `G`
    #print(f"shift walker n chunks: {num_chunks}")
    G_chunks = np.array_split(G, num_chunks)
    c_chunks = np.array_split(c, num_chunks)
    mask_chunks = np.array_split(mask, num_chunks)

    # **Preallocate result storage to avoid concatenation overhead**
    GRes = np.empty_like(G, dtype=np.float32)
    errorRes = np.empty(m, dtype=np.float32)

    # **Parallel processing**
    results = Parallel(n_jobs=ncpu)(
        delayed(shift_walkers)(G_chunk, X, h, d, c_chunk, mask_chunk)
        for G_chunk, c_chunk, mask_chunk in zip(G_chunks, c_chunks, mask_chunks)
    )

    # **Store results in preallocated arrays**
    start = 0
    for res_G, res_error in results:
        end = start + len(res_G)
        GRes[start:end] = res_G
        errorRes[start:end] = res_error
        start = end

    return GRes, errorRes


def shift_walkers(G, X, h, d, c, mask, cleanup_threshold=1e6):
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

    # determine if gc.collect() is needed
    clean = m * n > cleanup_threshold

    # Compute H and Hinv matrices
    H = np.eye(D, dtype=np.float32) * h ** 2
    Hinv = np.eye(D, dtype=np.float32) / h ** 2

    # Compute mean probability
    pj = np.mean(c, axis=1)[:, None, None]  # (m, 1, 1)

    # Compute u for selected elements only
    mask_indices = np.argwhere(mask)  # (k, 2) where k is the number of True values in mask
    diff_selected = np.take(G, mask_indices[:, 0], axis=0) - np.take(X, mask_indices[:, 1], axis=0)

    # Compute u directly without creating the intermediate step
    u_diff = np.einsum('ij,njk->nik', Hinv, diff_selected)  # (k, D, 1) #/h**2 still needed?

    if clean:
        del diff_selected
        gc.collect()

    # Compute g for selected walker points using the mask
    c_selected = c[mask_indices[:, 0], mask_indices[:, 1]]  # (k,)
    g = np.zeros_like(G, dtype=np.float64)  # (m, D, 1) # use dtype=np.float32 breaks the code for some reasons
    np.add.at(g, mask_indices[:, 0], -c_selected[:, None, None] * u_diff / n)

    # Compute Hessian update
    product = np.einsum('nik,njk->nij', u_diff, u_diff) - Hinv  # (k, D, D)

    # **Adaptive Memory Cleanup**
    if clean:
        del u_diff
        gc.collect()

    # Update Hessian matrix
    Hess = np.zeros((m, D, D), dtype=np.float32)  # (m, D, D)
    np.add.at(Hess, mask_indices[:, 0], c_selected[:, None, None] * product / n)

    # **Adaptive Memory Cleanup**
    if m * n > cleanup_threshold:
        del product
        del c_selected
        gc.collect()

    # Compute Sigmainv
    Sigmainv = (-Hess + np.einsum('mik,mil->mkl', g, g)/pj)/pj  # (m, D, D)


    # Compute the shift for each walker
    shift0 = G + np.einsum('ij,mjk->mik', H, g) / pj # (m, D, 1)

    # **Adaptive Memory Cleanup**
    if m * n > cleanup_threshold:
        del Hess
        gc.collect()

    # Eigen decomposition
    EigVal, EigVec = np.linalg.eigh(Sigmainv) # (m, D), (m, D, D)

    # Get the eigenvectors with the largest eigenvalues
    V = EigVec[:, :, d:D]  # (m, D, D-d)

    # **Adaptive Memory Cleanup**
    if m * n > cleanup_threshold:
        del Sigmainv
        del EigVal
        del EigVec
        gc.collect()

    # Update G for each walker
    G = np.einsum('mij,mjk->mik', np.einsum('mik,mjk->mij', V, V), (shift0 - G)) + G  # (m, D, 1)

    # Compute error
    tmp = np.einsum('mji,mjk->mik', V, g)  # (m, D, 1)
    error = np.sqrt(np.einsum('mik,mik->m', tmp, tmp) / np.einsum('mik,mik->m', g, g))  # (m,)

    # **Adaptive Memory Cleanup**
    if m * n > cleanup_threshold:
        del g
        del V
        del tmp
        gc.collect()

    return G, error

