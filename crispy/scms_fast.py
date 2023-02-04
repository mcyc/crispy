import numpy as np
import time
import gaussian as gs_pyx

def find_ridge(X, G, D=3, h=1, d=1, eps = 1e-06, maxT = 1000, wweights = None, converge_frac = 99):

    G = G.astype('float')
    X = X.astype('float')
    n = len(X)
    m = len(G)  # x and y coordinates 2xN format
    print("n, m: {0}, {1}".format(n,m))
    t = 0

    H = np.eye(D) * h**2
    Hinv = np.eye(D) / h**2
    error = np.full(m, 1e+08)

    if wweights is None:
        weights = 1
    else:
        weights = wweights

    # start timing
    start_time = time.time()

    pct_error = np.percentile(error, converge_frac)

    while ((pct_error > eps) & (t < maxT)):
        # loop through iterations
        t = t + 1
        print("-------- iteration {0} --------".format(t))

        itermask = np.where(error > eps)
        GjList = G[itermask]
        GRes, errorRes = shift_particle_vec(GjList, X, D, h, d, weights, n, H, Hinv)
        G[itermask] = GRes
        error[itermask] = errorRes
        pct_error = np.percentile(error, converge_frac)
        print("{0}%-tile error: {1}".format(converge_frac, pct_error))

        elapsed_time = time.time() - start_time
        # print elapsed_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    return G


def shift_particle_vec(Gj, X, D, h, d, weights, n, H, Hinv):

    def get_c(X,Gj,h):
        # find Gaussian provability centered around Gj, evaulated at X
        # broadcast to a common shape
        XX = np.broadcast_to(X, (Gj.shape[0],) + X.shape)
        GG = np.broadcast_to(Gj[:,None], (Gj.shape[0],) + X.shape)

        # remove the additional dimension
        XX = np.squeeze(XX)
        GG = np.squeeze(GG)

        # broadcast the covariance to the right dimensions
        HH = np.broadcast_to(h**2, (GG.shape[0], GG.shape[1]))
        return np.exp(vectorized_gaussian_logpdf(XX, means=GG, covariances=HH))

    # compute probability density of each X point
    c = get_c(X,Gj,h)

    # now weight the probability of each X point by the image (i.e., c *= weights)
    c *= np.broadcast_to(weights, c.shape)

    # compute mean probability of the test particle for each walker G
    pj = c.mean(axis=1)

    # broadcast Gj for vectorization operation
    Gj = np.broadcast_to(Gj[:, None], (Gj.shape[0],) + X.shape)

    # compute gradient and Hessian
    u = np.matmul(Hinv, (Gj - X)) / h ** 2

    def get_g_n_Hess(c, u, n, Hinv):
        cb = np.broadcast_to(c[:, :, None, None], u.shape)
        g = -np.sum(cb * u, axis=1) / n
        # sum over all the pixels (but not walkers)
        Hess = np.sum(cb*(np.matmul(u, T_1D(u)) - Hinv), axis=1)/ n
        return g, Hess

    g, Hess = get_g_n_Hess(c, u, n, Hinv)

    # compute inverse of the covariance matrix
    def get_Sigmainv(Hess, g, pj):
        ppj = np.broadcast_to(pj[:, None, None], Hess.shape)
        return -Hess / ppj + np.matmul(g, T_1D(g)) / ppj ** 2

    Sigmainv = get_Sigmainv(Hess, g, pj)

    # compute shift
    def get_shift0(Gj, H, g, pj):
        Gxg = np.matmul(H, g)
        ppj2 = np.broadcast_to(pj[:, None, None], Gxg.shape)
        return Gj + np.broadcast_to((Gxg / ppj2)[:, None], Gj.shape)

    shift0 = get_shift0(Gj, H, g, pj)

    # compute eigenvectors with the largest eigenvalues down to D-d
    EigVal, EigVec = np.linalg.eigh(Sigmainv)
    V = EigVec[:, d:D]

    # shift the test particle
    # Note, the order of matrix multiplication is reverse from the non-vectorized version
    VVT = np.matmul(T_1D(V), V)
    VVT = np.broadcast_to(VVT[:, None], (VVT.shape[0],) + (X.shape[0],)+ (VVT.shape[1], VVT.shape[2]))
    Gj = np.matmul(VVT, shift0 - Gj) + Gj

    # compute error
    tmp = np.matmul(V, g)
    errorj = np.sqrt(np.sum(tmp**2, axis=(1,2)) / np.sum(g**2, axis=(1,2)))

    return Gj[:,0,:], errorj


def T_1D(mtxAry):
    # return an array of transposed 1D matrices
    if mtxAry.ndim == 4:
        return np.transpose(mtxAry, axes=(0,1,3,2))
    elif mtxAry.ndim == 3:
        return np.transpose(mtxAry, axes=(0,2,1))
    elif mtxAry.ndim == 2:
        return mtxAry.T

#=======================================================================================================================


def vectorized_gaussian_logpdf(X, means, covariances, cython=True):
    if cython:
        return gs_pyx.vectorized_gaussian_logpdf_py(X, means, covariances)
    else:
        return vectorized_gaussian_logpdf_py(X, means, covariances)

def vectorized_gaussian_logpdf_py(X, means, covariances):
    """
    Compute log N(x_i; mu_i, sigma_i) for each x_i, mu_i, sigma_i
    Note: this assumes the covariance matrices constructed from covariances are diagonal
    The n, m, d in the arguments are the number of X, mean, and data dimesions, respectively
    Args:
        X : shape (n, m, d)
            Data points
        means : shape (n, m, d)
            Mean vectors
        covariances : shape (n, m)
            Diagonal covariance matrices
    Returns:
        logpdfs : shape (n,)
            Log probabilities
    """

    # add another axis to covariances to be compitable with X and mean
    covariances = covariances[:,:,None]

    # find the dimesions of the data
    d = X.shape[-1]
    constant = d * np.log(2 * np.pi)
    log_determinants = np.log(np.prod(covariances, axis=-1))

    deviations = X - means
    inverses = 1 / covariances

    return -0.5 * (constant + log_determinants +
        np.sum(deviations * inverses * deviations, axis=-1))
