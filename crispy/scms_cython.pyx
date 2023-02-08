import numpy as np
import astropy.io.fits as fits
import os, time
from scipy.stats import multivariate_normal

def find_ridge_sing(X, G, D=3, h=1, d=1, eps = 1e-06, maxT = 1000, weights = None, converge_frac = 99):

    X = X.astype('float')
    G = G.astype('float')

    n = len(X)
    m = len(G)  # x and y coordinates 2xN format
    print "n, m: {0}, {1}".format(n,m)
    t = 0
    H = np.eye(D) * h**2
    Hinv = np.eye(D) / h**2
    error = np.full(m, 1e+08)

    if weights is None:
        weights = 1

    def T_1D(mtxAry):
        # return an array of transposed 1D matrices
        return np.transpose(mtxAry, axes=(0,2,1))

    # start timing
    start_time = time.time()

    pct_error = np.percentile(error, converge_frac)
    while ((pct_error > eps) & (t < maxT)):
        # loop through iterations
        t = t + 1
        print("-------- iteration {0} --------".format(t))

        for j in np.where(error > eps)[0]:
            # loop through points in the mesh (i.e., G)
            c = multivariate_normal.pdf(X.reshape(X.shape[0:2]), mean=G[j].ravel(), cov=H)

            # now weight the probability of each X point by the image
            c = c*weights
            # reshape c so it can be broadcasted onto 3 dimension arrays
            c = c[:, None, None]
            pj = np.mean(c)

            u = np.matmul(Hinv, (G[j] - X))/h**2
            g = -1*np.sum((c * u),axis=0)/n

            # compute the Hessian matrix
            Hess = np.sum(c * (np.matmul(u, T_1D(u)) - Hinv), axis=0) / n

            Sigmainv = -1*Hess/pj + np.matmul(g, g.T)/pj**2
            shift0 = G[j] + np.matmul(H, g) / pj

            # Sigmainv matrices computed here are symmetric, and thus linalg.eigh is preferred
            # note that the eigenvectors in linalg.eigh is already sorted unlike linalg.eig
            EigVal, EigVec = np.linalg.eigh(Sigmainv)

            # get the eigenvectors with the largest eigenvalues down to d-1
            #V = EigVec[:, 0:D - d]
            V = EigVec[:, d:D]

            VVT= np.matmul(V, V.T)
            G[j] = np.matmul(VVT, shift0 - G[j]) + G[j]

            tmp = np.matmul(V.T, g)
            error[j] = np.sqrt(np.sum(tmp**2) / np.sum(g**2))

        #print "maximum error: {0}".format(error.max())
        pct_error = np.percentile(error, converge_frac)
        print "{0}%-tile error: {1}".format(converge_frac, pct_error)

        elapsed_time = time.time() - start_time
        #print elapsed_time
        print time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    return G

