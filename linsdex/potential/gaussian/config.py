"""
This controls the backend algorithm we use for sampling from a Gaussian.
This controls whether we use the Cholesky decomposition or the SVD to compute the
matrix square root of the covariance matrix when sampling.  This results in two
different parametrizations of the Gaussian.  The Cholesky parametrization is
differentiable whereas the SVD parametrization is not, but the SVD parametrization
works when the covariance matrix is low rank while the Cholesky parametrization
does not.  TODO: Make SVD differentiable.
"""
USE_CHOLESKY_SAMPLING = False