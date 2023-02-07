import numpy as np
import os, time
from scipy.stats import multivariate_normal

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
        print("number of walkers remaining: {}".format(len(GjList)))
        GRes, errorRes = shift_walkers(X, GjList, weights, h, H, Hinv, n, d, D)

        G[itermask] = GRes
        #print("GRes: {}".format(GRes))
        error[itermask] = errorRes
        #print("errorRes: {}".format(errorRes))
        pct_error = np.percentile(error, converge_frac)
        print("{0}%-tile error: {1}".format(converge_frac, pct_error))

        elapsed_time = time.time() - start_time
        # print elapsed_time
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    print("number of cpu to be used: {}".format(1))

    return G

def shift_walkers(X, G, weights, h, H, Hinv, n, d, D):
    # loop through points in the mesh (i.e., G)

    m = len(G)
    newG = np.zeros(G.shape)
    newErr = np.zeros(m)

    for j, Gj in enumerate(G):
        newG[j], newErr[j] = shift_walker(X, Gj, weights, h, H, Hinv, n, d, D)
        '''
        results = shift_walker(X, Gj, weights, h, H, Hinv, n, d, D)
        print(results[0].shape)
        newG[j] = results[0]
        newErr[j] = results[1]
        '''

    return newG, newErr

def shift_walker(X, Gj, weights, h, H, Hinv, n, d, D):

    c = multivariate_normal.pdf(X.reshape(X.shape[0:2]), mean=Gj.ravel(), cov=H)

    # now weight the probability of each X point by the image
    c = c * weights
    # reshape c so it can be broadcasted onto 3 dimension arrays
    c = c[:, None, None]
    pj = np.mean(c)

    u = np.matmul(Hinv, (Gj - X)) / h ** 2
    g = -1 * np.sum((c * u), axis=0) / n

    # compute the Hessian matrix
    Hess = np.sum(c * (np.matmul(u, T_1D(u)) - Hinv), axis=0) / n

    Sigmainv = -1 * Hess / pj + np.matmul(g, g.T) / pj ** 2
    shift0 = Gj + np.matmul(H, g) / pj

    # Sigmainv matrices computed here are symmetric, and thus linalg.eigh is preferred
    # note that the eigenvectors in linalg.eigh is already sorted unlike linalg.eig
    EigVal, EigVec = np.linalg.eigh(Sigmainv)

    # get the eigenvectors with the largest eigenvalues down to d-1
    V = EigVec[:, d:D]

    VVT = np.matmul(V, V.T)
    Gj = np.matmul(VVT, shift0 - Gj) + Gj

    tmp = np.matmul(V.T, g)
    errorj = np.sqrt(np.sum(tmp ** 2) / np.sum(g ** 2))
    #return Gj.ravel(), errorj
    return Gj, errorj

def T_1D(mtxAry):
    # return an array of transposed 1D matrices
    return np.transpose(mtxAry, axes=(0,2,1))




