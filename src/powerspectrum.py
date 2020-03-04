import numpy
from scipy import signal


def dft_matrix(nu):
    dft = numpy.exp(-2 * numpy.pi * 1j / len(nu)) ** numpy.arange(0, len(nu), 1)
    dftmatrix = numpy.vander(dft, increasing=True) / numpy.sqrt(len(nu))

    eta = numpy.arange(0, len(nu), 1) / (nu.max() - nu.min())
    return dftmatrix, eta


def blackman_harris_taper(frequency_range):
    window = signal.blackmanharris(len(frequency_range))
    return window


def compute_ps_variance(taper1, taper2, covariance, dft_matrix):
    tapered_cov = covariance * taper1 * taper2
    eta_cov = numpy.dot(numpy.dot(dft_matrix.conj().T, tapered_cov), dft_matrix)
    variance = numpy.diag(numpy.real(eta_cov))

    return variance