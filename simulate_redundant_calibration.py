import sys
from scipy.constants import c
import numpy
import matplotlib
from matplotlib import pyplot

# Import local codes
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
sys.path.append("../corrcal2")

from radiotelescope import AntennaPositions
from radiotelescope import BaselineTable
from radiotelescope import ideal_gaussian_beam

from skymodel import SkyRealisation
from skymodel import apparent_fluxes_numba

from corrcal2 import grid_data

from util import split_visibility
from util import find_sky_model_sources
from util import generate_sky_model_vectors
from util import generate_covariance_vectors

from calibrate import hybrid_calibration

def main():
    # 12 500 sources within 2.5 primary beam width
    # about 10 calibrator sources with primary beam weighted flux > 3x RMS
    # beam width 13 wavelength dishes
    # 8x8 array with 20 wavelength spacing
    # position noise of 0.04 wavelengths
    # per visibility noise 0.1*brightes source

    frequency_range = numpy.array([150]) * 1e6
    n_antenna_grid_points = 8
    separation_scale = 20
    antenna_scale = 13  # wavelengths
    noise_fraction_brightest_source = 0.1

    wavelength = c / frequency_range[0]
    array_size = (n_antenna_grid_points - 1) * wavelength * separation_scale
    antenna_size = antenna_scale * wavelength

    antenna_table = AntennaPositions(load=False, shape=['square', array_size / 2, n_antenna_grid_points, 0, 0])
    antenna_table.antenna_ids = numpy.arange(0, len(antenna_table.antenna_ids), 1)

    baseline_table = BaselineTable(position_table=antenna_table, frequency_channels=frequency_range)

    # We go down to 40 mili-Jansky to get about 10 calibration sources
    sky_realisation = SkyRealisation(sky_type="random", flux_low=40e-3, flux_high=10)
    sky_model_sources = find_sky_model_sources(sky_realisation, frequency_range, antenna_size=antenna_size)

    # Create thermal noise
    noise_level = noise_fraction_brightest_source * sky_model_sources.fluxes.max()
    # now compute the visibilities
    data_complex = sky_realisation.create_visibility_model(baseline_table, frequency_range, antenna_size=antenna_size)
    thermal_noise = numpy.random.normal(scale=noise_level, size=data_complex.shape)


    data_sorted, u_sorted, v_sorted, noise_sorted, ant1_sorted, ant2_sorted, edges_sorted, sorting_indices, \
    conjugation_flag = grid_data(data_complex + thermal_noise,
                                 baseline_table.u_coordinates,
                                 baseline_table.v_coordinates,
                                 thermal_noise,
                                 baseline_table.antenna_id1.astype(int),
                                 baseline_table.antenna_id2.astype(int))

    data_split = split_visibility(data_sorted)
    model_split = generate_sky_model_vectors(sky_model_sources, baseline_table, frequency_range, antenna_size)[0, :]

    covariance_split = generate_covariance_vectors(baseline_table.number_of_baselines, frequency_range)
    noise_split = numpy.zeros(data_split.shape[0]) + noise_level

    gain_solutions = hybrid_calibration(data_split, noise_split, covariance_split, model_split, edges_sorted,
                                        ant1_sorted, ant2_sorted, gain_guess = None, scale_factor = 1000)

    return gain_solutions


def plot_sky_counts(fluxes):
    bins = numpy.logspace(numpy.log10(fluxes.min()), numpy.log10(fluxes.max()), 100)
    hist, edges = numpy.histogram(fluxes, bins=bins)

    bin_centers = (edges[:len(edges) - 1] + edges[1:]) / 2.
    pyplot.plot(bin_centers, hist)
    pyplot.xscale('log')
    pyplot.yscale('log')

    return


if "__main__" == __name__:
    main()
