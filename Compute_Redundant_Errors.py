import numpy
import powerbox
from scipy.constants import c
from scipy import signal
from src.generaltools import symlog_bounds
from src.radiotelescope import beam_width

from src.plottools import colorbar
from src.generaltools import from_eta_to_k_par
from src.generaltools import from_u_to_k_perp
from src.generaltools import from_jansky_to_milikelvin
from src.covariance import position_covariance
from src.covariance import beam_covariance
from src.covariance import sky_covariance
from src.powerspectrum import blackman_harris_taper
from src.powerspectrum import dft_matrix
import matplotlib.colors as colors
from matplotlib import pyplot

def gain_error_covariance(u_range, frequency_range, residuals='both', weights=None, broken_baseline_weight = 1):
    model_variance = numpy.diag(sky_covariance(0, 0, frequency_range, S_low=1, S_high=10))
    model_normalisation = numpy.sqrt(numpy.outer(model_variance, model_variance))
    gain_error_covariance = numpy.zeros((len(u_range), len(frequency_range), len(frequency_range)))

    # Compute all residual to model ratios at different u scales
    for u_index in range(len(u_range)):
        if residuals == "position":
            residual_covariance = position_covariance(u=u_range[u_index], v=0, nu=frequency_range)
        elif residuals == "beam":
            residual_covariance = broken_baseline_weight **2*beam_covariance(u=u_range[u_index], v=0, nu=frequency_range)
        elif residuals == 'both':
            residual_covariance = position_covariance(u=u_range[u_index], v=0, nu=frequency_range) + \
                                  broken_baseline_weight **2*beam_covariance(u_range[u_index], v=0, nu=frequency_range)
        gain_error_covariance[u_index, :, :] = residual_covariance / model_normalisation

    if weights is None:
        gain_averaged_covariance = numpy.sum(gain_error_covariance, axis=0) * (1/(127*len(u_range))) ** 2
    else:
        gain_averaged_covariance = gain_error_covariance.copy()
        for u_index in range(len(u_range)):
            u_weight_reshaped = numpy.tile(weights[u_index, :].flatten(), (len(frequency_range), len(frequency_range), 1)).T
            gain_averaged_covariance[u_index, ...] = numpy.sum(gain_error_covariance * u_weight_reshaped, axis=0)
    return gain_averaged_covariance

def compute_ps_variance(taper1, taper2, covariance, dft_matrix):
    tapered_cov = covariance * taper1 * taper2
    eta_cov = numpy.dot(numpy.dot(dft_matrix.conj().T, tapered_cov), dft_matrix)
    variance = numpy.diag(numpy.real(eta_cov))

    return variance


def residual_ps_error(u_range, frequency_range, residuals='both', broken_baselines_weight = 1, weights = None, path="./", plot=True):
    cal_variance = numpy.zeros((len(u_range), len(frequency_range)))
    raw_variance = numpy.zeros((len(u_range), len(frequency_range)))

    window_function = blackman_harris_taper(frequency_range)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)
    dftmatrix, eta = dft_matrix(frequency_range)

    gain_averaged_covariance = gain_error_covariance(u_range, frequency_range, residuals=residuals, weights= weights,
                                                     broken_baseline_weight = broken_baselines_weight)

    # Compute the gain corrected residuals at all u scales
    if residuals == "position":
        residual_variance = position_covariance(u=0, v=0, nu=frequency_range)
    elif residuals == "beam":
        residual_variance = broken_baselines_weight **2*beam_covariance(u=0, v=0, nu=frequency_range)
    elif residuals == 'both':
        residual_variance = position_covariance(0, 0, frequency_range) + \
                            broken_baselines_weight **2*beam_covariance(u=0, v=0, nu=frequency_range)

    gain = residual_variance / sky_covariance(u=0, v=0, nu=frequency_range)
    for i in range(len(u_range)):
        if residuals == "position":
            residual_covariance = sky_covariance(u=u_range[i], v=0, nu=frequency_range)
            blaah = 0
        elif residuals == "beam":
            residual_covariance = broken_baselines_weight**2*beam_covariance(u=u_range[i], v=0, nu=frequency_range)
            blaah = 0
        elif residuals == 'both':
            residual_covariance = position_covariance(u=u_range[i], v=0, nu=frequency_range) + \
                                  broken_baselines_weight**2*beam_covariance(u=u_range[i], v=0, nu=frequency_range)
            blaah = 0

        model_covariance = sky_covariance(u=u_range[i], v=0, nu=frequency_range, S_low=1, S_high=10)
        scale = numpy.diag(numpy.zeros_like(frequency_range) ) + 1 + blaah
        if weights is None:
            nu_cov = 2*gain_averaged_covariance*model_covariance + \
                     (scale + 2*gain_averaged_covariance)*residual_covariance
        else:
            nu_cov = 2*gain_averaged_covariance[i, ...]*model_covariance + \
                     (scale + 2*gain_averaged_covariance[i, ...])*residual_covariance

        cal_variance[i, :] = compute_ps_variance(taper1, taper2, nu_cov, dftmatrix)
        raw_variance[i, :] = compute_ps_variance(taper1, taper2, residual_covariance, dftmatrix)

    pyplot.plot(gain_averaged_covariance)
    pyplot.show()
    return eta[:int(len(eta) / 2)], raw_variance[:, :int(len(eta) / 2)], cal_variance[:, :int(len(eta) / 2)]


if __name__ == "__main__":
    from matplotlib import pyplot

    u = numpy.logspace(-1, 2.5, 100)
    nu = numpy.linspace(140, 160, 101) * 1e6

    eta, sky_only_raw, sky_only_cal = residual_ps_error(u, nu, residuals="both")
    # pyplot.show()

