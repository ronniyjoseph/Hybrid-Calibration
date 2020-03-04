import sys
import os
import numpy
import argparse

from scipy.constants import c

# Import local codes
# sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
sys.path.append("../../CorrCal_UKZN_Development/corrcal")
from corrcal import grid_data

from src.covariance import thermal_noise
from src.radiotelescope import AntennaPositions
from src.radiotelescope import BaselineTable
from src.skymodel import SkyRealisation

from src.util import split_visibility
from src.util import find_sky_model_sources
from src.util import generate_sky_model_vectors
from src.util import generate_covariance_vectors

from src.calibrate import hybrid_calibration
from src.plottools import colorbar


def main(mode="run"):
    # 12 500 sources within 2.5 primary beam width
    # about 10 calibrator sources with primary beam weighted flux > 3x RMS
    # beam width 13 wavelength dishes
    # 8x8 array with 20 wavelength spacing
    # position noise of 0.04 wavelengths
    # per visibility noise 0.1*brightest source

    output_path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/Square_Large_Array_100Jy_Noise_No_Models/"
    frequency_range = numpy.array([150])*1e6
    tile_size = 4  # wavelengths
    noise_fraction_brightest_source = 0.1
    position_precision = 0
    broken_tile_fraction = 0
    sky_model_limit = 12
    n_realisations = 1000

    if mode == "run":
        print(f"Running Simulation")
        calibration_simulation(frequency_range=frequency_range, antenna_diameter=tile_size,
                               noise_fraction_brightest_source=noise_fraction_brightest_source,
                               position_error=position_precision, broken_tile_fraction=broken_tile_fraction,
                               sky_model_limit=sky_model_limit, n_realisations=n_realisations, output_path=output_path)
    elif mode == 'process':
        simulation_processing(output_path)

    return


def simulation_processing(output_path):
    gain_realisations = numpy.load(output_path + "gain_solutions.npy")
    realisation_mean = numpy.mean(gain_realisations, axis=0)
    deviations = gain_realisations - numpy.tile(realisation_mean, (8 * 8, 1))
    converged_results = numpy.tile(numpy.isnan(realisation_mean), (8 * 8, 1))

    figure, axes = pyplot.subplots(1, 2, figsize=(10, 5), subplot_kw=dict(yscale="log"))
    axes[0].hist(numpy.abs(deviations[converged_results == False]).flatten(), bins=100)
    axes[1].hist(numpy.angle(deviations[converged_results == False]).flatten(), bins=100)

    pyplot.show()
    return


def calibration_simulation(frequency_range, antenna_diameter, noise_fraction_brightest_source, position_error,
                           broken_tile_fraction, sky_model_limit, n_realisations, output_path, save_inputs=True):
    if save_inputs:
        setup_simulation_directory(output_path)
        input_parameters = numpy.array([[position_error], [broken_tile_fraction], [sky_model_limit], [n_realisations]])
        header_string = "Position_Precision[m]  Broken_Tile[Fraction]   Sky_Model_Depth[Jy] , Realisations"
        numpy.savetxt(output_path + "input_parameters.txt", input_parameters.T, header=header_string)

    # antenna_table = AntennaPositions(load=False, shape=['linear', 1000, 20])
    antenna_table = AntennaPositions(load=False, shape=['square', 100, 4, 0, 0 ])

    antenna_table.antenna_ids = numpy.arange(0, len(antenna_table.antenna_ids), 1)
    antenna_table.antenna_gains[2] = 2

    print(f"Progress: \r", )
    for i in range(n_realisations):
        if (i / n_realisations * 100 % 10) == 0.0:
            print(f"{int(i / n_realisations * 100)}% ... \r", )
        calibration_realisation(frequency_range=frequency_range, antenna_table=antenna_table,
                                                  noise_fraction_brightest_source=noise_fraction_brightest_source,
                                                  antenna_size=antenna_diameter, position_precision=position_error,
                                                  broken_tile_fraction=broken_tile_fraction,
                                                  sky_model_limit=sky_model_limit, seed=i, save_inputs=True,
                                                  output_path=output_path)

    print("")

    return


def calibration_realisation(frequency_range, antenna_table, noise_fraction_brightest_source, antenna_size,
                            position_precision=0, broken_tile_fraction=0.0, sky_model_limit=1e-1, seed=0,
                            save_inputs=True, output_path=None):
    numpy.random.seed(seed)

    position_errors = numpy.random.normal(loc=0, scale=position_precision, size=2 * antenna_table.antenna_ids.shape[0])
    antenna_table.antenna_ids = numpy.arange(0, len(antenna_table.antenna_ids), 1)
    antenna_table.x_coordinates += position_errors[:len(antenna_table.antenna_ids)]
    antenna_table.y_coordinates += position_errors[len(antenna_table.antenna_ids):]
    baseline_table = BaselineTable(position_table=antenna_table, frequency_channels=frequency_range)

    # We go down to 40 mili-Jansky to get about 10 calibration sources
    sky_realisation = SkyRealisation(sky_type="random", flux_low=40e-3, flux_high=10, seed=seed)
    # sky_realisation = SkyRealisation(sky_type="point", fluxes=numpy.array([10, 1]), l_coordinates=numpy.array([0.1, 0.2]),
    #                                  m_coordinates=numpy.array([0, 0.15]), spectral_indices=numpy.array([0.8, 0.8]))
    # print(sky_realisation.l_coordinates.shape)

    sky_model_sources = find_sky_model_sources(sky_realisation, frequency_range, antenna_size=antenna_size,
                                               sky_model_depth=sky_model_limit)
    print(f"Including {len(sky_model_sources.l_coordinates)} sources in the sky model")
    if save_inputs:
        if not os.path.exists(output_path + f"realisation_{seed}"):
            print(f"Creating folder for realisation {seed}")
            os.makedirs(output_path + f"realisation_{seed}")
        sky_realisation.save_table(output_path + f"realisation_{seed}/", "sky_realisation")
        antenna_table.save_position_table(output_path + f"realisation_{seed}/", "telescope_positions")
        antenna_table.save_gain_table(output_path + f"realisation_{seed}/", "telescope_gains")
        baseline_table.save_table(output_path + f"realisation_{seed}/", "baseline_table")

    # Create thermal noise
    noise_level = 1e2#thermal_noise()
    # now compute the visibilities
    ideal_visibilities = sky_realisation.create_visibility_model(baseline_table, frequency_range,
                                                                 antenna_size=antenna_size)
    noise_realisation = numpy.random.normal(scale=noise_level, size=(ideal_visibilities.shape[0], 2))
    noise_visibilities = numpy.zeros_like(ideal_visibilities)
    noise_visibilities[:, 0] = noise_realisation[:, 0] + 1j*noise_realisation[:, 1]

    measured_visibilities = baseline_table.baseline_gains*ideal_visibilities + noise_visibilities

    if save_inputs:
        numpy.save(output_path + f"realisation_{seed}/" + "ideal_visibilities", ideal_visibilities)
        numpy.save(output_path + f"realisation_{seed}/" + "noise_visibilities", noise_visibilities)
        numpy.save(output_path + f"realisation_{seed}/" + "measured_visibilities", measured_visibilities)

    data_sorted, u_sorted, v_sorted, noise_sorted, ant1_sorted, ant2_sorted, edges_sorted, sorting_indices, \
    conjugation_flag = grid_data(measured_visibilities,
                                 baseline_table.u_coordinates,
                                 baseline_table.v_coordinates,
                                 noise_visibilities,
                                 baseline_table.antenna_id1.astype(int),
                                 baseline_table.antenna_id2.astype(int))

    data_vector = split_visibility(data_sorted)
    model_vectors = generate_sky_model_vectors(sky_model_sources, baseline_table, frequency_range, antenna_size)
    covariance_vectors = 1.5*generate_covariance_vectors(baseline_table.number_of_baselines, frequency_range,
                                                     10)
    noise_split = numpy.zeros(data_vector.shape[0]) + noise_level**2

    print("Calibrating the Sky")
    gain_solutions = hybrid_calibration(data_vector, noise_split, covariance_vectors, model_vectors, edges_sorted,
                                        ant1_sorted, ant2_sorted, gain_guess=None, scale_factor=1000)
    numpy.save(output_path + f"realisation_{seed}/" + "gain_solutions", gain_solutions)

    return gain_solutions


def plot_sky_counts(fluxes):
    bins = numpy.logspace(numpy.log10(fluxes.min()), numpy.log10(fluxes.max()), 100)
    hist, edges = numpy.histogram(fluxes, bins=bins)

    bin_centers = (edges[:len(edges) - 1] + edges[1:]) / 2.
    pyplot.plot(bin_centers, hist)
    pyplot.xscale('log')
    pyplot.yscale('log')

    return


def setup_simulation_directory(output_path):
    if not os.path.exists(output_path + "/"):
        print("Creating Project folder at output destination!")
        os.makedirs(output_path)
    return


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Redundant Calibration Simulation set up')
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    parser.add_argument("-r", action="store_true", default=True)
    parser.add_argument("-p", action="store_true", default=False)
    args = parser.parse_args()

    import matplotlib

    if args.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    if args.p:
        main(mode="process")
    elif args.r:
        main(mode="run")
