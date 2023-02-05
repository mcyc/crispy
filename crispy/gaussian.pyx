import numpy as np
cimport numpy as np

def vectorized_gaussian_logpdf(np.ndarray[double, ndim=3] X, np.ndarray[double, ndim=3] means, np.ndarray[double, ndim=2] covariances):
    cdef int d
    cdef np.ndarray[double, ndim=3] covariances_expanded, deviations, inverses
    cdef np.ndarray[double, ndim=2] log_determinants
    cdef double constant = 0

    # add another axis to covariances to be compitable with X and mean
    covariances_expanded = covariances[:,:,None]

    # find the dimesions of the data (note, indexing of -1 doesn work the same as python)
    d = X.shape[2]

    constant = float(d) * np.log(2 * np.pi)
    log_determinants = np.log(np.prod(covariances_expanded, axis=-1))
    deviations = X - means
    inverses = 1 / covariances_expanded

    return -0.5 * (constant + log_determinants +
        np.sum(deviations * inverses * deviations, axis=-1))
