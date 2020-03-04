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
from src.covariance import import sky

import matplotlib.colors as colors

def gain_error_covariance(u_range, frequency_range, residuals='both', weights=None, broken_baseline_weight = 1):
    model_variance = numpy.diag(sky_covariance(0, 0, frequency_range, S_low=1, S_high=10))
    model_normalisation = numpy.sqrt(numpy.outer(model_variance, model_variance))
    gain_error_covariance = numpy.zeros((len(u_range), len(frequency_range), len(frequency_range)))

    # Compute all residual to model ratios at different u scales
    for u_index in range(len(u_range)):
        if residuals == "sky":
            residual_covariance = sky_covariance(u_range[u_index], v=0, nu=frequency_range)
        elif residuals == "beam":
            residual_covariance = broken_baseline_weight **2*beam_covariance(u_range[u_index], v=0, nu=frequency_range)
        elif residuals == 'both':
            residual_covariance = sky_covariance(u_range[u_index], v=0, nu=frequency_range) + \
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
