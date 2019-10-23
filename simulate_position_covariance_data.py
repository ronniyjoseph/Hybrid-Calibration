import os
import sys
import numpy
import copy
from numba import prange, njit
from matplotlib import pyplot
from src.util import hexagonal_array

sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")

from radiotelescope import BaselineTable
from skymodel import SkyRealisation
from skymodel import create_visibilities_analytic
from cramer_rao_bound import redundant_baseline_finder
from simulate_beam_covariance_data import compute_baseline_covariance
from simulate_beam_covariance_data import create_hex_telescope

import time


def position_covariance_simulation(array_size=3, create_signal=False, compute_covariance=True, plot_covariance=True,
                                   show_plot = True):
    output_path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/"
    project_path = "redundant_based_position_covariance/"
    n_realisations = 10000
    position_precision = 1e-2
    if not os.path.exists(output_path + project_path + "/"):
        print("Creating Project folder at output destination!")
        os.makedirs(output_path + project_path)
    hex_telescope = create_hex_telescope(array_size)
    if create_signal:
        create_visibility_data(hex_telescope, position_precision, n_realisations, output_path + project_path, output_data=True)

    if compute_covariance:
        covariance = compute_baseline_covariance(hex_telescope, output_path + project_path, n_realisations)
    if plot_covariance:
        figure, axes = pyplot.subplots(1,2, figsize = (10,5))
        axes[0].imshow(numpy.real(covariance))
        axes[0].set_title("Real - Baseline Position Covariance ")
        axes[0].set_xlabel("Baseline Index")
        axes[0].set_ylabel("Baseline Index")

        axes[1].imshow(numpy.imag(covariance))
        axes[1].set_title("Imaginary - Baseline Position Covariance ")
        axes[1].set_xlabel("Baseline Index")
        figure.savefig(output_path + project_path + "Position_Covariance_Plot.pdf")

        if show_plot:
            pyplot.show()

    return


def create_visibility_data(telescope_object, position_precision, n_realisations, path, output_data=False):
    print("Creating Signal Realisations")
    if not os.path.exists(path + "/" + "Simulated_Visibilities") and output_data:
        print("Creating realisation folder in Project path")
        os.makedirs(path + "/" + "Simulated_Visibilities")

    for i in range(n_realisations):
        print(f"Realisation {i}")
        source_population = SkyRealisation(sky_type='random', flux_high=1)
        ideal_baselines = redundant_table(telescope_object.baseline_table)

        perturbed_telescope = copy.deepcopy(telescope_object)
        number_antennas = len(perturbed_telescope.antenna_positions.x_coordinates)
        perturbed_telescope.antenna_positions.x_coordinates += numpy.random.normal(0, position_precision, number_antennas)
        perturbed_telescope.antenna_positions.y_coordinates += numpy.random.normal(0, position_precision, number_antennas)

        perturbed_telescope.baseline_table = BaselineTable(position_table = perturbed_telescope.antenna_positions)

        perturbed_baselines = redundant_table(perturbed_telescope.baseline_table)

        model_visibilities = create_visibilities_analytic(source_population, ideal_baselines,
                                                           frequency_range = numpy.array([150e6]))

        perturbed_visibilities = create_visibilities_analytic(source_population, perturbed_baselines,
                                                           frequency_range = numpy.array([150e6]))

        residual_visibilities = model_visibilities.flatten() - perturbed_visibilities.flatten()

        numpy.save(path + "/" + "Simulated_Visibilities/" + f"model_realisation_{i}", model_visibilities)
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"perturbed_realisation_{i}", perturbed_visibilities)
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"residual_realisation_{i}", residual_visibilities)
    return


def redundant_table(original_table):
    redundant_baselines = redundant_baseline_finder(original_table.antenna_id1, original_table.antenna_id2,
                                                    original_table.u_coordinates, original_table.v_coordinates,
                                                    original_table.w_coordinates, verbose=False)
    redundant_table = BaselineTable()
    redundant_table.antenna_id1 = redundant_baselines[:, 0]
    redundant_table.antenna_id2 = redundant_baselines[:, 1]
    redundant_table.u_coordinates = redundant_baselines[:, 2]
    redundant_table.v_coordinates = redundant_baselines[:, 3]
    redundant_table.w_coordinates = redundant_baselines[:, 4]
    redundant_table.reference_frequency = 150e6
    redundant_table.number_of_baselines = len(redundant_baselines[:, 0])
    return redundant_table

if __name__ == "__main__":
    position_covariance_simulation()