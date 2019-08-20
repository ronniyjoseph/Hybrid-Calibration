import sys
import numpy

sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")

from skymodel import apparent_fluxes_numba
from analytic_covariance import sky_covariance

def split_visibility(data):
    data_real = numpy.real(data)
    data_imag = numpy.imag(data)

    data_split = numpy.hstack((data_real, data_imag)).reshape((1, 2 * len(data_real)), order="C")
    return data_split[0, :]


def find_sky_model_sources(sky_realisation, frequency_range, antenna_size = 4):
    apparent_flux = apparent_fluxes_numba(sky_realisation, frequency_range, antenna_diameter = antenna_size )
    rms = numpy.sqrt(numpy.mean(sky_realisation.fluxes**2))
    model_source_indices = numpy.where(apparent_flux[:, 0] > 3*rms)
    sky_model = sky_realisation.select_sources(model_source_indices)

    return sky_model


def generate_sky_model_vectors(sky_model_sources, baseline_table, frequency_range, antenna_size, sorting_indices = None):
    number_of_sources = len(sky_model_sources.fluxes)
    sky_vectors = numpy.zeros((number_of_sources, baseline_table.number_of_baselines*2))
    for i in range(number_of_sources):
        single_source = sky_model_sources.select_sources(i)
        source_visibilities = single_source.create_visibility_model(baseline_table, frequency_range, antenna_size)
        if sorting_indices is not None:
            sky_vectors[i, :] = split_visibility(source_visibilities[sorting_indices])

    return sky_vectors


def generate_covariance_vectors(number_of_baselines, frequency_range):
    covariance_vectors = numpy.zeros((2, number_of_baselines*2))
    covariance_vectors[0::2] = 1
    covariance_vectors[1::2] = 1
    covariance_vectors *= sky_covariance(0, 0, frequency_range)

    return covariance_vectors
