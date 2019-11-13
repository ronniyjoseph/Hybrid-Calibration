import os
import sys
import numpy
from numba import prange, njit
from matplotlib import pyplot
from src.util import hexagonal_array
from src.util import redundant_baseline_finder

from src.radiotelescope import RadioTelescope
from src.radiotelescope import AntennaPositions
from src.radiotelescope import BaselineTable
from src.radiotelescope import broken_mwa_beam_loader
from src.skymodel import SkyRealisation
from src.skymodel import create_visibilities_analytic
from src.generaltools import from_lm_to_theta_phi
from src.plottools import colorbar


def beam_covariance_simulation(array_size=3, create_signal=False, compute_covariance=True, plot_covariance=True,
                               show_plot=True):
    output_path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/"
    project_path = "redundant_based_beam_covariance/"
    n_realisations = 10000

    if not os.path.exists(output_path + project_path + "/"):
        print("Creating Project folder at output destination!")
        os.makedirs(output_path + project_path)
    hex_telescope = create_hex_telescope(array_size)
    if create_signal:
        create_visibility_data(hex_telescope, n_realisations, output_path + project_path, output_data=True)

    if compute_covariance:
        compute_baseline_covariance(hex_telescope, output_path + project_path, n_realisations, data_type='model')
        compute_baseline_covariance(hex_telescope, output_path + project_path, n_realisations, data_type='perturbed')
        compute_baseline_covariance(hex_telescope, output_path + project_path, n_realisations, data_type='residual')

    if plot_covariance:
        plot_covariance_data(output_path + project_path, simulation_type = "Beam")
        if show_plot:
            pyplot.show()

    return


def broken_tiles(telescope, fraction=25 / 128, seed=None, number_dipoles=16):
    if seed is not None:
        numpy.random.seed((seed))
    number_antennas = len(telescope.antenna_positions.x_coordinates)
    # Determine number of broken tiles
    number_broken_tiles = numpy.random.binomial(n=number_antennas, p=fraction, size=1)
    # Select which tiles are broken
    broken_flags = numpy.zeros(number_antennas, dtype=int)
    broken_tile_indices = numpy.random.randint(0, number_antennas, number_broken_tiles)
    broken_dipole_indices = numpy.random.randint(0, number_dipoles, number_broken_tiles)
    broken_flags[broken_tile_indices] = broken_dipole_indices

    return broken_flags


def create_visibility_data(telescope_object, n_realisations, path, output_data=False):
    print("Creating Signal Realisations")
    if not os.path.exists(path + "/" + "Simulated_Visibilities") and output_data:
        print("Creating realisation folder in Project path")
        os.makedirs(path + "/" + "Simulated_Visibilities")

    original_baselines = telescope_object.baseline_table
    redundant_baselines = redundant_baseline_finder(original_baselines)

    for i in range(n_realisations):
        print(f"Realisation {i}")
        broken_flags = broken_tiles(telescope_object, seed=i)
        source_population = SkyRealisation(sky_type='random', flux_high=1, seed = i)

        model_visibilities = create_visibilities_analytic(source_population, redundant_baselines,
                                                           frequency_range = numpy.array([150e6]))
        perturbed_visibilities = create_perturbed_visibilities(source_population, redundant_baselines, broken_flags)

        residual_visibilities = model_visibilities.flatten() - perturbed_visibilities

        numpy.save(path + "/" + "Simulated_Visibilities/" + f"model_realisation_{i}", model_visibilities)
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"perturbed_realisation_{i}", perturbed_visibilities)
        numpy.save(path + "/" + "Simulated_Visibilities/" + f"residual_realisation_{i}", residual_visibilities)
    return


def create_hex_telescope(size):
    hex_telescope = RadioTelescope(load=False)

    antenna_positions = hexagonal_array(size)
    antenna_table = AntennaPositions(load=False)
    antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
    antenna_table.x_coordinates = antenna_positions[:, 0]
    antenna_table.y_coordinates = antenna_positions[:, 1]
    antenna_table.z_coordinates = antenna_positions[:, 2]

    hex_telescope.antenna_positions = antenna_table
    hex_telescope.baseline_table = BaselineTable(position_table=antenna_table)

    return hex_telescope


def apparent_flux_possibilities(source_population, number_of_dipoles=16, nu=150e6):
    number_of_sources = len(source_population.fluxes)
    theta, phi = from_lm_to_theta_phi(source_population.l_coordinates, source_population.m_coordinates)

    beam_response = numpy.zeros((number_of_sources, number_of_dipoles + 1), dtype=complex)
    flux_beam_product = numpy.zeros_like(beam_response)
    for i in range(number_of_dipoles + 1):
        if i == 0:
            faulty_dipole_i = None
        else:
            faulty_dipole_i = i - 1

        beam_response[:, i] = broken_mwa_beam_loader(theta, phi, frequency=nu, faulty_dipole=faulty_dipole_i,
                                                     load=False)
        flux_beam_product[:, i] = beam_response[:, i] * source_population.fluxes
    apparent_fluxes = numpy.einsum('ij,ik->ijk', flux_beam_product, numpy.conj(beam_response))
    return apparent_fluxes


def create_perturbed_visibilities(source_population, baseline_table, broken_flags, frequency = 150e6):
    observations = numpy.zeros(baseline_table.number_of_baselines, dtype = complex)
    apparent_fluxes = apparent_flux_possibilities(source_population, nu = frequency)

    flags_antenna1 = broken_flags[baseline_table.antenna_id1.astype(int)]
    flags_antenna2 = broken_flags[baseline_table.antenna_id2.astype(int)]
    numba_perturbed_loop(observations, apparent_fluxes, source_population.l_coordinates, source_population.m_coordinates
                         , baseline_table.u(frequency), baseline_table.v(frequency), flags_antenna1, flags_antenna2)

    return observations


@njit(parallel=True)
def numba_perturbed_loop(observations, fluxes, l_source, m_source, u_baselines, v_baselines, broken_flags1,
                         broken_flags2):
    for source_index in prange(len(fluxes)):
        for baseline_index in range(u_baselines.shape[0]):
            kernel = numpy.exp(-2j * numpy.pi * (u_baselines[baseline_index] * l_source[source_index] +
                                                 v_baselines[baseline_index] * m_source[source_index]))

            observations[baseline_index] += fluxes[source_index, broken_flags1[baseline_index],
                                                   broken_flags2[baseline_index]] * kernel


def compute_baseline_covariance(telescope_object, path, n_realisations, data_type = "residual"):
    original_table = telescope_object.baseline_table
    redundant_baselines = redundant_baseline_finder(original_table)

    if not os.path.exists(path + "/" + "Simulated_Covariance"):
        print("Creating Covariance folder in Project path")
        os.makedirs(path + "/" + "Simulated_Covariance")

    residuals = numpy.zeros((2*redundant_baselines.number_of_baselines, n_realisations), dtype = complex)
    for i in range(n_realisations):
        residuals_realisation = numpy.load(path + "Simulated_Visibilities/" + f"{data_type}_realisation_{i}.npy").flatten()
        residuals[0::2, i] = numpy.real(residuals_realisation)
        residuals[1::2, i] = numpy.imag(residuals_realisation)
    baseline_covariance = numpy.cov(residuals)
    numpy.save(path + "Simulated_Covariance/" + f"baseline_{data_type}_covariance", baseline_covariance)

    return


def plot_covariance_data(path, simulation_type = "Unspecified"):
    if not os.path.exists(path + "/" + "Plots"):
        print("Creating realisation folder in Project path")
        os.makedirs(path + "/" + "Plots")
    data_labels =['Ideal', 'Perturbed', 'Residual']

    data = []
    data.append(numpy.load(path + "Simulated_Covariance/" + f"baseline_model_covariance.npy"))
    data.append(numpy.load(path + "Simulated_Covariance/" + f"baseline_perturbed_covariance.npy"))
    data.append(numpy.load(path + "Simulated_Covariance/" + f"baseline_residual_covariance.npy"))

    figure, axes = pyplot.subplots(3, 2, figsize=(8, 12))
    figure.suptitle(f"Baseline {simulation_type} Covariance")
    for i in range(3):
        realplot = axes[i, 0].imshow(numpy.real(data[i]))
        imagplot = axes[i, 1].imshow(numpy.imag(data[i]))
        colorbar(realplot)
        colorbar(imagplot)

        axes[i, 0].set_title(f"Re({data_labels[i]})")
        axes[i, 1].set_title(f"Im({data_labels[i]}) ")

        axes[i, 0].set_ylabel("Baseline Index")
        if i == 2:
            axes[i, 0].set_xlabel("Baseline Index")
            axes[i, 1].set_xlabel("Baseline Index")
    figure.subplots_adjust(top=0.9)
    figure.savefig(path + "Plots/" +f"{simulation_type}_Covariance_Plot.pdf")
    return



if __name__ == "__main__":
    beam_covariance_simulation()
