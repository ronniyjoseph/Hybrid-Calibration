import sys
import numpy
import argparse
import matplotlib
from matplotlib import pyplot
from scipy.constants import c

# Import local codes
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
sys.path.append("../corrcal2")

from radiotelescope import AntennaPositions
from radiotelescope import BaselineTable
from skymodel import SkyRealisation
from corrcal2 import grid_data

from util import split_visibility
from util import find_sky_model_sources
from util import generate_sky_model_vectors
from util import generate_covariance_vectors

from calibrate import hybrid_calibration

def main(mode = "run"):
    # 12 500 sources within 2.5 primary beam width
    # about 10 calibrator sources with primary beam weighted flux > 3x RMS
    # beam width 13 wavelength dishes
    # 8x8 array with 20 wavelength spacing
    # position noise of 0.04 wavelengths
    # per visibility noise 0.1*brightes source

    output_path = "/data/rjoseph/Hybrid_Calibration/correlation_cal_simulations/redundant_calibration/"
    frequency_range = numpy.array([150]) * 1e6
    n_antenna_grid_points = 8
    separation_scale = 20
    antenna_scale = 13  # wavelengths
    noise_fraction_brightest_source = 0.1
    n_realisations = 2000

    if mode == "run":
        calibration_simulation(frequency_range, n_antenna_grid_points, separation_scale, antenna_scale,
                               noise_fraction_brightest_source, n_realisations, output_path)
    elif mode == 'process':
        simulation_processing(output_path)

    return

def simulation_processing(output_path):
    gain_realisations = numpy.load(output_path + "gain_solutions.npy")
    realisation_mean = numpy.mean(gain_realisations, axis=0)
    deviations = gain_realisations - numpy.tile(realisation_mean, (8*8,1))
    converged_results = numpy.tile(numpy.isnan(realisation_mean), (8*8,1))

    figure, axes = pyplot.subplots(1, 2, figsize=(10, 5), subplot_kw= dict(yscale = "log"))
    axes[0].hist(numpy.abs(deviations[converged_results == False]).flatten(), bins = 100)
    axes[1].hist(numpy.angle(deviations[converged_results == False]).flatten(), bins = 100)

    pyplot.show()
    return

def calibration_simulation(frequency_range, n_antenna_grid_points, separation_scale, antenna_scale,
                           noise_fraction_brightest_source, n_realisations, output_path):
    wavelength = c / frequency_range[0]
    array_size = (n_antenna_grid_points - 1) * wavelength * separation_scale
    antenna_size = antenna_scale * wavelength

    antenna_table = AntennaPositions(load=False, shape=['square', array_size / 2, n_antenna_grid_points, 0, 0])
    antenna_table.antenna_ids = numpy.arange(0, len(antenna_table.antenna_ids), 1)

    solutions = numpy.zeros((len(antenna_table.antenna_ids), n_realisations), dtype=complex)
    print(f"Completed \r", )
    for i in range(n_realisations):
        if (i / n_realisations * 100 % 10) == 0.0:
            print(f"{int(i / n_realisations * 100)}% ... \r", )
        try:
            solutions[:, i] = calibration_realisation(frequency_range, antenna_table, noise_fraction_brightest_source,
                                                      antenna_size, seed=i)
        except:
            solutions[:, i] = numpy.nan

    numpy.save(output_path + "gain_solutions", solutions)
    print("")

    return

def calibration_realisation(frequency_range, antenna_table,  noise_fraction_brightest_source, antenna_size, seed=0):
    numpy.random.seed(seed)

    position_errors = numpy.random.normal(0.04*c/frequency_range[0], size = 2*antenna_table.antenna_ids.shape[0])

    antenna_table.antenna_ids = numpy.arange(0, len(antenna_table.antenna_ids), 1)
    antenna_table.x_coordinates += position_errors[:len(antenna_table.antenna_ids)]
    antenna_table.y_coordinates += position_errors[len(antenna_table.antenna_ids):]

    baseline_table = BaselineTable(position_table=antenna_table, frequency_channels=frequency_range)

    # We go down to 40 mili-Jansky to get about 10 calibration sources
    sky_realisation = SkyRealisation(sky_type="random", flux_low=40e-3, flux_high=10, seed=seed)
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

    data_vector = split_visibility(data_sorted)
    model_vectors = generate_sky_model_vectors(sky_model_sources, baseline_table, frequency_range, antenna_size)[0, :]

    covariance_vectors = generate_covariance_vectors(baseline_table.number_of_baselines, frequency_range)
    noise_split = numpy.zeros(data_vector.shape[0]) + noise_level

    gain_solutions = hybrid_calibration(data_vector, noise_split, covariance_vectors, model_vectors, edges_sorted,
                                        ant1_sorted, ant2_sorted, gain_guess=None, scale_factor=1000)
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
    parser = argparse.ArgumentParser(description='Redundant Calibration Simulation set up')
    parser.add_argument("-r", action = "store_true", default = True)
    parser.add_argument("-p", action="store_true", default=False)
    args = parser.parse_args()

    if args.p:
        main(mode = "process")
    elif args.r:
        main(mode ="run")