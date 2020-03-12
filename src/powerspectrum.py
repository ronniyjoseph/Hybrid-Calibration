import numpy
from scipy import signal


def discrete_fourier_transform_matrix(nu):
    dft = numpy.exp(-2 * numpy.pi * 1j / len(nu)) ** numpy.arange(0, len(nu), 1)
    dftmatrix = numpy.vander(dft, increasing=True) / numpy.sqrt(len(nu))

    return dftmatrix


def from_frequency_to_eta(nu):
    eta = numpy.arange(0, len(nu), 1) / (nu.max() - nu.min())
    return eta[:int(len(nu) / 2)]


def blackman_harris_taper(frequency_range):
    window = signal.blackmanharris(len(frequency_range))
    return window


def compute_power(nu, covariance):
    dft_matrix = discrete_fourier_transform_matrix(nu)
    window = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window, window)

    tapered_cov = covariance * taper1 * taper2
    eta_cov = numpy.dot(numpy.dot(dft_matrix.conj().T, tapered_cov), dft_matrix)
    power = numpy.diag(numpy.real(eta_cov))[:int(len(nu) / 2)]

    return power

