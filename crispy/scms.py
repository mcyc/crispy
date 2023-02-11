import numpy as np
import time
import multiprocessing as mp

#======================================================================================================================#

def find_ridge(X, G, D=3, h=1, d=1, eps = 1e-06, maxT = 1000, wweights = None, converge_frac = 99, ncpu = None,
               return_all_G=False):

    # use float32 to make the operation more efficient (particularly since the precision need isn't too high)
    G = G.astype(np.float32)
    X = X.astype(np.float32)
    h = np.float32(h)
    eps = np.float32(eps)
    wweights = np.float32(wweights)
    converge_frac = np.float32(converge_frac)

    n = len(X)
    m = len(G)  # x and y coordinates 2xN format
    print("n, m: {0}, {1}".format(n,m))
    t = 0

    H = np.eye(D) * h**2
    Hinv = np.eye(D) / h**2
    error = np.full(m, 1e+08, dtype=np.float32)

    if wweights is None:
        weights = np.float32(1)
    else:
        weights = wweights

    # start timing
    start_time = time.time()

    pct_error = np.percentile(error, converge_frac)

    # assign the number of cpus to use if not specified:
    if ncpu is None:
        ncpu = mp.cpu_count() - 1

    while ((pct_error > eps) & (t < maxT)):
        # loop through iterations
        t = t + 1
        print("-------- iteration {0} --------".format(t))

        itermask = np.where(error > eps)
        GjList = G[itermask]
        print("number of active walkers: {0}".format(len(GjList)))
        GRes, errorRes = shift_walkers_multi(X, GjList, weights, h, H, Hinv, n, d, D, ncpu)

        G[itermask] = GRes
        error[itermask] = errorRes

        pct_error = np.percentile(error, converge_frac)
        print("{0}%-tile error: {1}".format(converge_frac, pct_error))

        elapsed_time = time.time() - start_time
        # print elapsed_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("number of cpu to be used: {}".format(ncpu))

    if return_all_G:
        return G
    else:
        mask = np.where(error < eps)
        return G[mask]


def shift_walkers_multi(X, G, weights, h, H, Hinv, n, d, D, ncpu):
    # run shift_walkers using multiprocessing
    with mp.Manager() as manager:
        shared_dict = manager.dict()
        shared_dict['X'] = X
        shared_dict['D'] = D
        shared_dict['h'] = h
        shared_dict['d'] = d
        shared_dict['weights'] = weights
        shared_dict['n'] = n
        shared_dict['H'] = H
        shared_dict['Hinv'] = Hinv

        with mp.Pool(processes=ncpu) as pool:
            results = [pool.apply_async(shift, args=(Gj, shared_dict)) for Gj in
                       np.array_split(G, ncpu)]
            GRes_list = [r.get()[0] for r in results]
            errorRes_list = [r.get()[1] for r in results]

        GRes = np.concatenate(GRes_list)
        errorRes = np.concatenate(errorRes_list)

        return GRes, errorRes

def shift(G, shared_dict):
    # a wrapper function around shift_walkers to be used for multi-processing
    X = shared_dict['X']
    D = shared_dict['D']
    h = shared_dict['h']
    d = shared_dict['d']
    weights = shared_dict['weights']
    n = shared_dict['n']
    H = shared_dict['H']
    Hinv = shared_dict['Hinv']
    return shift_walkers(X, G, weights, h, H, Hinv, n, d, D)

def shift_walkers(X, G, weights, h, H, Hinv, n, d, D):
    # Loop through each walker. A more effecient vectorization has yet to be found when the loop is excuted
    # via multi-processing

    m = len(G)
    newG = np.zeros(G.shape, dtype=np.float32)
    newErr = np.zeros(m, dtype=np.float32)

    for j, Gj in enumerate(G):
        newG[j], newErr[j] = shift_particle(Gj, X, D, h, d, weights, n, H, Hinv)
    return newG, newErr

def shift_particle(Gj, X, D, h, d, weights, n, H, Hinv):
    # shift individual walkers using SCMS

    # evulate the Gaussian value of all the X points at Gj
    c = np.exp(gaussian(np.squeeze(X), mean=Gj.ravel(), covariance=h**2))

    # now weight the probability of each X point by the image
    c = c*weights

    # reshape c so it can be broadcasted onto 3 dimension arrays
    c = c[slice(None), None, None]
    pj = np.mean(c)

    u = np.matmul(Hinv, (Gj - X))/h**2
    g = -1*np.sum((c * u),axis=0)/n

    # compute the Hessian matrix
    Hess = np.sum(c * (np.matmul(u, T_1D(u)) - Hinv), axis=0) / n

    Sigmainv = -1*Hess/pj + np.matmul(g, g.T)/pj**2
    shift0 = Gj + np.matmul(H, g) / pj

    # Sigmainv matrices computed here are symmetric, and thus linalg.eigh is preferred
    # note that the eigenvectors in linalg.eigh is already sorted unlike linalg.eig
    EigVal, EigVec = np.linalg.eigh(Sigmainv)

    # get the eigenvectors with the largest eigenvalues down to D-d (e.g., D-1 for ridge finding)
    V = EigVec[slice(None), d:D]

    VVT= np.matmul(V, V.T)
    Gj = np.matmul(VVT, shift0 - Gj) + Gj

    tmp = np.matmul(V.T, g)
    errorj = np.sqrt(np.sum(tmp**2) / np.sum(g**2))
    #return np.append(Gj.ravel(), [errorj])
    return Gj, errorj

def gaussian(X, mean, covariance):
    """
    Compute log N(x_i; mu, sigma) for each x_i, mu, covariance
    Args:
        X : shape (n, d)
            Data points
        means : shape (d)
            A mean vector
        covariances : float
            A sigle covariance of the gaussian, same for all dimensions.
            This calculation assume a diagnal covariance matrix with identifical elements
    Returns:
        logpdfs : shape (n,)
            Log probabilities
    """
    d = X.shape[1]
    constant = d * np.log(2 * np.float32(np.pi))
    log_determinants = np.log(covariance)
    deviations = X - mean
    inverses = 1 / covariance
    return -0.5 * (constant + log_determinants +
        np.sum(deviations * inverses * deviations, axis=1))

def T_1D(mtxAry):
    # return an array of transposed 1D matrices
    return np.transpose(mtxAry, axes=(0,2,1))
