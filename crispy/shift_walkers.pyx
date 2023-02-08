import numpy as np
cimport numpy as np

def shift_walkers(np.ndarray[double, ndim=3] X,
                  np.ndarray[double, ndim=3] G,
                  np.ndarray[double, ndim=1] weights,
                  double h,
                  np.ndarray[double, ndim=2] H,
                  np.ndarray[double, ndim=2] Hinv,
                  int n, int d, int D):

    cdef int m, j
    cdef np.ndarray[double, ndim=3] newG
    cdef np.ndarray[double, ndim=2] Gj
    cdef np.ndarray[double, ndim=1] newErr

    # loop through points in the mesh (i.e., G)

    m = G.shape[0]
    newG = np.zeros(np.shape(G))
    newErr = np.zeros(m)

    for j in range(m):
        Gj = G[j]
        newG[j], newErr[j] = shift_walker(X, Gj, weights, h, H, Hinv, n, d, D)

    return newG, newErr



def shift_walker(np.ndarray[double, ndim=3] X,
                 np.ndarray[double, ndim=2] Gj,
                 np.ndarray[double, ndim=1] weights,
                 double h,
                 np.ndarray[double, ndim=2] H,
                 np.ndarray[double, ndim=2] Hinv,
                 int n, int d, int D):

    cdef double pj, errorj
    cdef np.ndarray[double, ndim=3] u, cc
    cdef np.ndarray[double, ndim=2] g, Hess, Sigmainv, shift0, V, VVT, EigVec, tmp
    cdef np.ndarray[double, ndim=1] EigVal, c

    c = np.exp(gaussian_logpdf(np.squeeze(X), mean=Gj.ravel(), covariance=h**2))

    # now weight the probability of each X point by the image
    c = c * weights

    pj = np.mean(c)
    u = np.matmul(Hinv, (Gj - X)) / h ** 2

    # reshape c so it can be broadcasted onto 3 dimension arrays
    cc = c[:, None, None]

    g = -1 * np.sum((cc * u), axis=0) / n

    # compute the Hessian matrix

    Hess = np.sum(cc * (np.matmul(u, np.transpose(u, axes=(0,2,1))) - Hinv), axis=0) / n
    Sigmainv = -1 * Hess / pj + np.matmul(g, np.transpose(g)) / pj ** 2
    shift0 = Gj + np.matmul(H, g) / pj

    # Sigmainv matrices computed here are symmetric, and thus linalg.eigh is preferred
    # note that the eigenvectors in linalg.eigh is already sorted unlike linalg.eig
    EigVal, EigVec = np.linalg.eigh(Sigmainv)

    # get the eigenvectors with the largest eigenvalues down to d-1
    V = EigVec[:, d:D]
    VVT = np.matmul(V, np.transpose(V))
    Gj = np.matmul(VVT, shift0 - Gj) + Gj
    tmp = np.matmul(np.transpose(V), g)
    errorj = np.sqrt(np.sum(tmp ** 2) / np.sum(g ** 2))
    return Gj, errorj



def gaussian_logpdf(np.ndarray[double, ndim=2] X, np.ndarray[double, ndim=1] mean, double covariance):
    cdef int d
    cdef double constant, log_determinants, inverses
    cdef np.ndarray[double, ndim=2] deviations

    d = np.shape(X)[1]
    constant = float(d) * np.log(2 * np.pi)
    log_determinants = np.log(covariance)
    deviations = X - mean
    inverses = 1 / covariance
    return -0.5 * (constant + log_determinants +
        np.sum(deviations * inverses * deviations, axis=1))


def T_1D(np.ndarray[double, ndim=3] mtxAry):
    # return an array of transposed 1D matrices
    return np.transpose(mtxAry, axes=(0,2,1))
