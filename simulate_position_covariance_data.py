import os
import sys
import numpy
import copy
from numba import prange, njit
from matplotlib import pyplot
from src.util import redundant_baseline_finder

from src.radiotelescope import BaselineTable
from src.skymodel import SkyRealisation
from src.skymodel import create_visibilities_analytic
from simulate_beam_covariance_data import compute_baseline_covariance
from simulate_beam_covariance_data import create_hex_telescope
from simulate_beam_covariance_data import plot_covariance_data
import time


def position_covariance_simulation(array_size=3, create_signal=False, compute_covariance=True, plot_covariance=True,
                                   show_plot = True):
    output_path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/"
    project_path = "position_covariance_numerical/"
    n_realisations = 1400
    position_precision = 1e-3
    if not os.path.exists(output_path + project_path + "/"):
        print("Creating Project folder at output destination!")
        os.makedirs(output_path + project_path)
    hex_telescope = create_hex_telescope(array_size)
    if create_signal:
        create_visibility_data(hex_telescope, position_precision, n_realisations, output_path + project_path,
                               output_data=True)

    if compute_covariance:
        compute_baseline_covariance(hex_telescope, output_path + project_path, n_realisations, data_type='model')
        compute_baseline_covariance(hex_telescope, output_path + project_path, n_realisations, data_type='perturbed')
        compute_baseline_covariance(hex_telescope, output_path + project_path, n_realisations, data_type='residual')

    if plot_covariance:
        plot_covariance_data(output_path + project_path, simulation_type = "Position")
        if show_plot:
            pyplot.show()

    return


def create_visibility_data(telescope_object, position_precision, n_realisations, path, output_data=False):
    print("Creating Signal Realisations")
    if not os.path.exists(path + "/" + "Simulated_Visibilities") and output_data:
        print("Creating realisation folder in Project path")
        os.makedirs(path + "/" + "Simulated_Visibilities")

    ideal_baselines = telescope_object.baseline_table

    for i in range(n_realisations):
        print(f"Realisation {i}")
        source_population = SkyRealisation(sky_type='random', flux_high=1, seed = i)

        perturbed_telescope = copy.copy(telescope_object)
        number_antennas = len(perturbed_telescope.antenna_positions.x_coordinates)
        x_offsets = numpy.random.normal(0, position_precision, number_antennas)
        y_offsets = numpy.random.normal(0, position_precision, number_antennas)
        perturbed_telescope.antenna_positions.x_coordinates += x_offsets
        perturbed_telescope.antenna_positions.y_coordinates += y_offsets

        perturbed_telescope.baseline_table = BaselineTable(position_table = perturbed_telescope.antenna_positions)
        perturbed_baselines = perturbed_telescope.baseline_table

        # if perturbed_baselines.number_of_baselines < ideal_baselines.number_of_baselines:
        #     map = perturbed_to_original_mapper(ideal_baselines, perturbed_baselines)
        # else:
        #     map = numpy.arange(0, perturbed_baselines.number_of_baselines, 1, dtype = int)
        model_visibilities = source_population.create_visibility_model(ideal_baselines,
                                                          frequency_channels = numpy.array([150e6]))
        #
        # perturbed_visibilities = numpy.zeros_like(model_visibilities)
        perturbed_visibilities = source_population.create_visibility_model(perturbed_baselines,
                                                           frequency_channels = numpy.array([150e6]))

        # residual_visibilities = numpy.zeros_like(model_visibilities)
        residual_visibilities = model_visibilities - perturbed_visibilities

        numpy.save(path + "/" + "Simulated_Visibilities/" + f"model_realisation_{i}", model_visibilities.flatten())
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"perturbed_realisation_{i}", perturbed_visibilities.flatten())
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"residual_realisation_{i}", residual_visibilities.flatten())
    return


def perturbed_to_original_mapper(original_baselines, perturbed_baselines):
    perturbed_to_original_mapping = numpy.zeros(perturbed_baselines.number_of_baselines)
    for i in range(perturbed_baselines.number_of_baselines):
        antenna1_indices = numpy.where(original_baselines.antenna_id1 == perturbed_baselines.antenna_id1[i])
        antenna2_indices = numpy.where(original_baselines.antenna_id2 == perturbed_baselines.antenna_id2[i])
        perturbed_to_original_mapping[i] = numpy.intersect1d(antenna1_indices, antenna2_indices)[0]

    return perturbed_to_original_mapping.astype(int)


if __name__ == "__main__":
    position_covariance_simulation()
