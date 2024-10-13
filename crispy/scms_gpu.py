from numba import cuda
import numpy as np
import time
import sys


# GPU kernel for shifting particles
@cuda.jit
def move_walkers_gpu(G, X, D, h, d, weights, n, H, Hinv, errors):
    j = cuda.grid(1)
    if j < G.shape[0]:
        Gj = G[j]
        c = cuda.local.array(1024, dtype=np.float32)  # Assuming maximum of 1024 points, adjust as needed
        cu = cuda.local.array((1024, D), dtype=np.float32)
        hh = cuda.local.array((1024, D, D), dtype=np.float32)

        for i in range(n):
            Xi = X[i]
            c_i = evaluate_gaussian(Xi, Gj, h ** 2)
            c[i] = c_i * weights[i]
            u_i = np.dot(Hinv, (Gj - Xi)) / h ** 2
            cu[i] = c[i] * u_i
            hh[i] = c[i] * np.dot(u_i, np.transpose(u_i)) - Hinv

        pj = np.mean(c)
        g = -1 * np.sum(cu, axis=0) / n
        Hess = np.sum(hh, axis=0) / n
        Sigmainv = -1 * Hess / pj + np.dot(g, np.transpose(g)) / pj ** 2
        shift0 = Gj + np.dot(H, g) / pj

        EigVal, EigVec = np.linalg.eigh(Sigmainv)
        V = np.ascontiguousarray(EigVec[:, d:D])
        VT = np.transpose(V)
        VVT = np.dot(V, VT)
        Gj = np.dot(VVT, shift0 - Gj) + Gj
        tmp = np.dot(VT, g)
        errorj = np.sqrt(np.sum(tmp ** 2) / np.sum(g ** 2))

        G[j] = Gj
        errors[j] = errorj


# Main function to invoke GPU kernel
def find_ridge_gpu(X, G, D=3, h=1, d=1, eps=1e-06, maxT=1000, wweights=None, converge_frac=99.0,
                   return_unconverged=True):
    # Prepare data for GPU
    G_device = cuda.to_device(G)
    X_device = cuda.to_device(X)
    errors_device = cuda.device_array(G.shape[0], dtype=np.float32)

    threads_per_block = 128
    blocks_per_grid = (G.shape[0] + (threads_per_block - 1)) // threads_per_block

    # Start timing
    start_time = time.time()
    last_print_time = start_time

    print("==========================================================================")
    print(f"Starting the run. The number of image points and walkers are {len(X)} and {len(G)}")
    print("==========================================================================")

    for t in range(maxT):
        # Launch GPU kernel
        move_walkers_gpu[blocks_per_grid, threads_per_block](
            G_device, X_device, D, h, d, wweights, len(X), np.eye(D), np.eye(D), errors_device
        )

        # Copy errors back to host for progress checking
        errors = errors_device.copy_to_host()

        # Calculate the convergence percentile
        pct_error = np.percentile(errors, converge_frac)

        # Print progress every second
        current_time = time.time()
        if current_time - last_print_time >= 1:
            elapsed_time = current_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            remaining_walkers = np.sum(errors > eps)
            sys.stdout.write(f"\rIteration {t + 1}"
                             f" | Number of walkers remaining: {remaining_walkers}/{len(G)}"
                             f" ({100 - remaining_walkers / len(G) * 100:0.1f}% complete)"
                             f" | {converge_frac}-percentile error: {pct_error:0.3f}"
                             f" | total run time: {formatted_time}")
            sys.stdout.flush()
            last_print_time = current_time

        # Check for convergence
        if pct_error <= eps:
            break

    sys.stdout.write("\n")

    # Copy converged walker positions back to host
    G = G_device.copy_to_host()

    # Mask for converged walkers
    mask = errors < eps

    if return_unconverged:
        # Return both converged and unconverged walkers
        return G[mask], G[~mask]
    else:
        # Return only converged walkers
        return G[mask]


# Helper function (adjust as needed for GPU compatibility)
@cuda.jit(device=True)
def evaluate_gaussian(X, mean, covariance):
    """
    Evaluate a D-dimensional Gaussian function at a given position.
    GPU-compatible version.
    """
    D = mean.shape[0]
    diff = X - mean
    inv_covariance = 1 / covariance  # Since covariance is scalar, the inverse covariance matrix is just 1/covariance
    normalization_factor = 1 / ((2 * np.pi * covariance) ** (D / 2))
    exponent = -0.5 * np.dot(diff.T, diff) * inv_covariance
    return normalization_factor * cuda.math.exp(exponent)  # Return as a scalar
