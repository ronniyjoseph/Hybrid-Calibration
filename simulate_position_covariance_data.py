import os
import numpy
import copy
import argparse

from matplotlib import pyplot
from src.radiotelescope import RadioTelescope
from src.radiotelescope import BaselineTable
from src.skymodel import SkyRealisation
from simulate_beam_covariance_data import compute_baseline_covariance
from simulate_beam_covariance_data import create_hex_telescope
from simulate_beam_covariance_data import plot_covariance_data
import time


def position_covariance_simulation(array_size=3, create_signal=True, compute_covariance=True, plot_covariance=True,
                                   show_plot=True):
    output_path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/"
    project_path = "linear_position_covariance_numerical_point_fixed/"
    n_realisations = 100000
    position_precision = 1e-3

    if not os.path.exists(output_path + project_path + "/"):
        print("Creating Project folder at output destination!")
        os.makedirs(output_path + project_path)

    telescope = RadioTelescope(load=False, shape=['linear', 14, 5])#create_hex_telescope(array_size)

    if create_signal:
        create_visibility_data(telescope, position_precision, n_realisations, output_path + project_path,
                               output_data=True)

    if compute_covariance:
        compute_baseline_covariance(telescope, output_path + project_path, n_realisations, data_type='model')
        compute_baseline_covariance(telescope, output_path + project_path, n_realisations, data_type='perturbed')
        compute_baseline_covariance(telescope, output_path + project_path, n_realisations, data_type='residual')

    if plot_covariance:
        figure, axes = pyplot.subplots(1, 3, figsize=(18, 5))
        plot_covariance_data(output_path + project_path, simulation_type="Position", figure=figure, axes=axes)
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
        if i % int(n_realisations/100) == 0:
            print(f"Realisation {i}")
        # source_population = SkyRealisation(sky_type='random', flux_high=1, seed=i)

        # l_coordinate = numpy.random.uniform(-1, 1, 1)
        # m_coordinate = numpy.random.uniform(-1, 1, 1)
        #
        # source_population = SkyRealisation(sky_type="point", fluxes=numpy.array([100]), l_coordinates=l_coordinate,
        #                                    m_coordinates=m_coordinate, spectral_indices=numpy.array([0.8]))

        source_population = SkyRealisation(sky_type="point", fluxes=numpy.array([100]), l_coordinates=0.3,
                                            m_coordinates=0.0, spectral_indices=numpy.array([0.8]))

        perturbed_telescope = copy.copy(telescope_object)
        # Compute position perturbations
        number_antennas = len(perturbed_telescope.antenna_positions.x_coordinates)
        x_offsets = numpy.random.normal(0, position_precision, number_antennas)
        y_offsets = numpy.random.normal(0, position_precision, number_antennas)
        # print(ideal_baselines.u_coordinates)

        perturbed_telescope.antenna_positions.x_coordinates += x_offsets
        perturbed_telescope.antenna_positions.y_coordinates += y_offsets

        # Compute uv coordinates
        perturbed_telescope.baseline_table = BaselineTable(position_table=perturbed_telescope.antenna_positions)

        perturbed_baselines = perturbed_telescope.baseline_table

        # Compute visibilities for the ideal case and the perturbed case
        model_visibilities = source_population.create_visibility_model(ideal_baselines,
                                                                       frequency_channels=numpy.array([150e6]))
        perturbed_visibilities = source_population.create_visibility_model(perturbed_baselines,
                                                                           frequency_channels=numpy.array([150e6]))
        residual_visibilities = model_visibilities - perturbed_visibilities

        numpy.save(path + "/" + "Simulated_Visibilities/" + f"model_realisation_{i}", model_visibilities.flatten())
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"perturbed_realisation_{i}",
                   perturbed_visibilities.flatten())
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"residual_realisation_{i}",
                   residual_visibilities.flatten())
    return


def perturbed_to_original_mapper(original_baselines, perturbed_baselines):
    perturbed_to_original_mapping = numpy.zeros(perturbed_baselines.number_of_baselines)
    for i in range(perturbed_baselines.number_of_baselines):
        antenna1_indices = numpy.where(original_baselines.antenna_id1 == perturbed_baselines.antenna_id1[i])
        antenna2_indices = numpy.where(original_baselines.antenna_id2 == perturbed_baselines.antenna_id2[i])
        perturbed_to_original_mapping[i] = numpy.intersect1d(antenna1_indices, antenna2_indices)[0]

    return perturbed_to_original_mapping.astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")

    from matplotlib import pyplot

    position_covariance_simulation()
