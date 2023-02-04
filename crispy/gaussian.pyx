import numpy as np
cimport numpy as np

def vectorized_gaussian_logpdf(np.ndarray[double, ndim=3] X, np.ndarray[double, ndim=3] means, np.ndarray[double, ndim=2] covariances):
    cdef int d
    cdef np.ndarray[np.float64_t, ndim=3] covariances_expanded
    cdef np.ndarray[np.float64_t, ndim=2] log_determinants, inverses, deviations
    cdef double constant = 0

    # add another axis to covariances to be compitable with X and mean
    covariances_expanded = np.expand_dims(covariances, axis=-1)

    # find the dimesions of the data
    d = X.shape[-1]
    constant = d * np.log(2 * np.pi)
    log_determinants = np.log(np.prod(covariances_expanded, axis=-1))

    deviations = X - means
    inverses = 1 / covariances_expanded

    return -0.5 * (constant + log_determinants +
        np.sum(deviations * inverses * deviations, axis=-1))
