import sys
import numpy
from scipy.constants import c
from scipy import sparse
import matplotlib
from matplotlib import pyplot
from src.util import hexagonal_array
from src.util import redundant_baseline_finder
from src.radiotelescope import beam_width
from src.radiotelescope import AntennaPositions
from src.radiotelescope import BaselineTable
from src.radiotelescope import RadioTelescope
from src.covariance import sky_covariance
from src.covariance import beam_covariance
from src.covariance import position_covariance
from src.covariance import thermal_variance
from src.skymodel import sky_moment_returner


def cramer_rao_bound_comparison(maximum_factor=8, nu=150e6, verbose=True, compute_data=False, load_data=True,
                                save_output=True, make_plot=True, show_plot=False):
    position_precision = 1e-2
    broken_tile_fraction = 0.3
    output_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/general_hex/"

    if compute_data:
        redundant_data, sky_data = cramer_rao_bound_calculator(maximum_factor, position_precision, broken_tile_fraction,
                                                               nu=nu, verbose=verbose)
        if save_output:
            numpy.savetxt(output_path + "redundant_crlb.txt", redundant_data)
            numpy.savetxt(output_path + "skymodel_crlb.txt", sky_data)
    if load_data:
        redundant_data = numpy.loadtxt(output_path + "redundant_crlb.txt")
        sky_data = numpy.loadtxt(output_path + "skymodel_crlb.txt")
    if make_plot:
        plot_cramer_bound(redundant_data, sky_data, plot_path=output_path)
        if show_plot:
            pyplot.show()
    return


def cramer_rao_bound_calculator(maximum_factor=3, position_precision=1e-2, broken_tile_fraction=0.3, nu=150e6,
                                verbose=True):
    size_factor = numpy.arange(2, maximum_factor + 1, 1)

    # Initialise empty Arrays for relative calibration results
    redundancy_metric = numpy.zeros(len(size_factor))
    relative_gain_variance = numpy.zeros_like(redundancy_metric)
    relative_gain_spread = numpy.zeros_like(redundancy_metric)

    # Initialise empty arrays for absolute calibration results
    absolute_gain_variance = numpy.zeros_like(redundancy_metric)

    # Initialise empty array for thermal noise results
    thermal_redundant_variance = numpy.zeros_like(redundancy_metric)

    # Initialise empty array for sky calibrated results
    sky_gain_variance = numpy.zeros_like(redundancy_metric)
    thermal_sky_variance = numpy.zeros_like(redundancy_metric)

    for i in range(len(size_factor)):
        antenna_positions = hexagonal_array(size_factor[i])
        antenna_table = AntennaPositions(load=False)
        antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
        antenna_table.x_coordinates = antenna_positions[:, 0]
        antenna_table.y_coordinates = antenna_positions[:, 1]
        antenna_table.z_coordinates = antenna_positions[:, 2]
        baseline_table = BaselineTable(position_table=antenna_table)

        if verbose:
            print("")
            print(f"Hexagonal array with size {size_factor[i]}")
            print("Finding redundant baselines")

        redundant_baselines = redundant_baseline_finder(baseline_table)
        skymodel_baselines = redundant_baseline_finder(baseline_table, group_minimum=1)

        if verbose:
            print("Populating matrices")

        redundant_crlb = relative_calibration_crlb(redundant_baselines, nu=nu, position_precision=position_precision,
                                                   broken_tile_fraction=broken_tile_fraction)
        absolute_crlb = absolute_calibration_crlb(skymodel_baselines, nu=150e6, position_precision=position_precision)

        redundancy_metric[i] = antenna_table.number_antennas()
        relative_gain_variance[i] = numpy.median(numpy.diag(redundant_crlb))
        relative_gain_spread[i] = numpy.std(numpy.diag(redundant_crlb))
        absolute_gain_variance[i] = absolute_crlb

        sky_gain_variance[i] = numpy.median(numpy.diag(sky_calibration_crlb(skymodel_baselines)))

        thermal_redundant_variance[i] = numpy.median(numpy.diag(thermal_redundant_crlb(redundant_baselines)))
        thermal_sky_variance[i] = numpy.median(numpy.diag(thermal_sky_crlb(skymodel_baselines)))

    redundant_data = numpy.stack((redundancy_metric, relative_gain_variance, absolute_gain_variance,
                                  thermal_redundant_variance))
    sky_data = numpy.stack((redundancy_metric, relative_gain_variance, thermal_sky_variance))

    return redundant_data, sky_data


def thermal_redundant_crlb(redundant_baselines, nu=150e6, SEFD=20e3, B=40e3, t=120):
    redundant_sky = numpy.sqrt(sky_moment_returner(n_order=2))
    jacobian_gain_matrix, red_tiles, red_groups = redundant_matrix_populator(redundant_baselines)

    jacobian_gain_matrix[:, :len(red_tiles)] *= redundant_sky
    thermal_noise = thermal_variance()
    redundant_fisher_information = numpy.dot(jacobian_gain_matrix.T, jacobian_gain_matrix) / thermal_noise

    redundant_crlb = 2 * numpy.real(numpy.linalg.pinv(redundant_fisher_information))
    return redundant_crlb[:len(red_tiles), :len(red_tiles)]


def relative_calibration_crlb(redundant_baselines, nu=150e6, position_precision=1e-2, broken_tile_fraction=1.0):
    redundant_sky = numpy.sqrt(sky_moment_returner(n_order=2))
    jacobian_gain_matrix, red_tiles, red_groups = redundant_matrix_populator(redundant_baselines)

    jacobian_gain_matrix[:, :len(red_tiles)] *= redundant_sky
    uv = numpy.array([0, position_precision / c * nu])
    non_redundancy_variance = beam_covariance(nu, uv, uv, broken_tile_fraction=broken_tile_fraction, mode='baseline') + \
                              position_covariance(nu, uv, uv, position_precision=position_precision, mode='baseline')

    redundant_fisher_information = numpy.dot(jacobian_gain_matrix.T, jacobian_gain_matrix) / non_redundancy_variance[
        0, 0]

    redundant_crlb = 2 * numpy.real(numpy.linalg.pinv(redundant_fisher_information))

    return redundant_crlb[:len(red_tiles), :len(red_tiles)]


def thermal_sky_crlb(redundant_baselines, nu=150e6, SEFD=20e3, B=40e3, t=120):
    sky_based_model = numpy.sqrt(sky_moment_returner(n_order=2, S_low=1, S_high=10))
    antenna_baseline_matrix, red_tiles, red_groups = redundant_matrix_populator(redundant_baselines)

    jacobian_gain_matrix = antenna_baseline_matrix[:, :len(red_tiles)] * sky_based_model
    thermal_noise = thermal_variance()
    redundant_fisher_information = numpy.dot(jacobian_gain_matrix.T, jacobian_gain_matrix) / thermal_noise

    redundant_crlb = 2 * numpy.real(numpy.linalg.pinv(redundant_fisher_information))
    return redundant_crlb[:len(red_tiles), :len(red_tiles)]


def sky_calibration_crlb(redundant_baselines, nu=150e6, position_precision=1e-2, broken_tile_fraction=1):
    absolute_fim = 0
    sky_based_model = numpy.sqrt(sky_moment_returner(n_order=2, S_low=1, S_high=10))
    antenna_baseline_matrix, red_tiles = sky_model_matrix_populator(redundant_baselines)

    uv_scales = numpy.array([0, position_precision / c * nu])
    non_redundant_covariance = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, mode='baseline')

    jacobian_gain_matrix = antenna_baseline_matrix[:, :len(red_tiles)] * sky_based_model

    groups = numpy.unique(redundant_baselines.group_indices)
    for group_index in range(len(groups)):
        group_visibilities_indices = numpy.where(redundant_baselines.group_indices == groups[group_index])[0]
        number_of_redundant_baselines = len(group_visibilities_indices)

        if number_of_redundant_baselines == 1:
            absolute_fim += numpy.dot(jacobian_gain_matrix[group_visibilities_indices, :].T,
                                      jacobian_gain_matrix[group_visibilities_indices, :]) / non_redundant_covariance[
                                0, 0]
        elif number_of_redundant_baselines > 1:
            group_start_index = numpy.min(group_visibilities_indices)
            group_end_index = numpy.max(group_visibilities_indices)

            # print("")
            # print(f"Group {group_index} with {number_of_redundant_baselines} baselines ")

            redundant_block = numpy.zeros((number_of_redundant_baselines, number_of_redundant_baselines)) + \
                              non_redundant_covariance[0, 0]
            # print("Created block of ones")
            redundant_block = restructure_covariance_matrix(redundant_block, diagonal=non_redundant_covariance[0, 0],
                                                            off_diagonal=non_redundant_covariance[0, 1])
            # print("Restructured")
            # print("Inverted Matrices")

            absolute_fim += numpy.dot(numpy.dot(jacobian_gain_matrix[group_start_index:group_end_index + 1, :].T,
                                                numpy.linalg.pinv(redundant_block)),
                                      jacobian_gain_matrix[group_start_index:group_end_index + 1, :])

    sky_crlb = 2 * numpy.real(numpy.linalg.pinv(absolute_fim))

    return sky_crlb


def absolute_calibration_crlb(redundant_baselines, position_precision=1e-2, nu=150e6):
    if redundant_baselines.number_of_baselines < 5000:
        absolute_crlb = small_covariance_matrix(redundant_baselines, nu, position_precision)
    elif redundant_baselines.number_of_baselines > 5000:
        print("Going Block Mode")
        absolute_crlb = large_covariance_matrix(redundant_baselines, nu, position_precision)
    return absolute_crlb


def small_covariance_matrix(redundant_baselines, nu=150e6, position_precision=1e-2):
    model_sky = numpy.sqrt(sky_moment_returner(n_order=2, S_low=1, S_high=10))
    uv_scales = numpy.array([0, position_precision / c * nu])
    non_redundant_covariance = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, mode='baseline')

    ideal_unmodeled_covariance = sky_covariance(nu=nu, u=redundant_baselines.u_coordinates,
                                                v=redundant_baselines.v_coordinates, mode='baseline')
    ideal_unmodeled_covariance = restructure_covariance_matrix(ideal_unmodeled_covariance,
                                                               diagonal=non_redundant_covariance[0, 0],
                                                               off_diagonal=non_redundant_covariance[0, 1])

    jacobian_vector = numpy.zeros(redundant_baselines.number_of_baselines) + model_sky

    absolute_fim = numpy.dot(numpy.dot(jacobian_vector.T, numpy.linalg.pinv(ideal_unmodeled_covariance)),
                             jacobian_vector)

    absolute_crlb = 2 * numpy.real(1 / absolute_fim)

    return absolute_crlb


def large_covariance_matrix(redundant_baselines, nu=150e6, position_precision=1e-2):
    absolute_fim = 0
    model_sky = numpy.sqrt(sky_moment_returner(n_order=2, S_low=1, S_high=10))
    uv_scales = numpy.array([0, position_precision / c * nu])
    non_redundant_covariance = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, mode='baseline')
    groups = numpy.unique(redundant_baselines.group_indices)
    for group_index in range(len(groups)):
        number_of_redundant_baselines = len(redundant_baselines.group_indices[redundant_baselines.group_indices ==
                                                                              groups[group_index]])
        # print("")
        # print(f"Group {group_index} with {number_of_redundant_baselines} baselines ")

        redundant_block = numpy.zeros((number_of_redundant_baselines, number_of_redundant_baselines)) + \
                          non_redundant_covariance[0, 0]

        # print("Created block of ones")
        redundant_block = restructure_covariance_matrix(redundant_block, diagonal=non_redundant_covariance[0, 0],
                                                        off_diagonal=non_redundant_covariance[0, 1])
        # print("Restructured")
        # print("Inverted Matrices")
        jacobian_vector = numpy.zeros(number_of_redundant_baselines) + model_sky

        absolute_fim += numpy.dot(numpy.dot(jacobian_vector.T, numpy.linalg.pinv(redundant_block)), jacobian_vector)

    absolute_crlb = 2 * numpy.real(1 / absolute_fim)
    return absolute_crlb


def restructure_covariance_matrix(matrix, diagonal, off_diagonal):
    matrix /= diagonal
    matrix -= numpy.diag(numpy.zeros(matrix.shape[0]) + 1)
    matrix *= off_diagonal
    matrix += numpy.diag(numpy.zeros(matrix.shape[0]) + diagonal)

    return matrix


def redundant_matrix_populator(uv_positions):
    # so first we sort out the unique antennas
    # and the unique redudant groups, this will allows us to populate the matrix adequately
    antenna_indices = numpy.stack((uv_positions.antenna_id1, uv_positions.antenna_id2))
    red_tiles = numpy.unique(antenna_indices)
    # it's not really finding unique antennas, it just finds unique values
    red_groups = numpy.unique(uv_positions.group_indices)
    # print "There are", len(red_tiles), "redundant tiles"
    # print ""
    # print "Creating the equation matrix"
    # create am empty matrix (#measurements)x(#tiles + #redundant groups)
    amp_matrix = numpy.zeros((uv_positions.number_of_baselines, len(red_tiles) + len(red_groups)))
    for i in range(uv_positions.number_of_baselines):
        index1 = numpy.where(red_tiles == uv_positions.antenna_id1[i])
        index2 = numpy.where(red_tiles == uv_positions.antenna_id2[i])
        index_group = numpy.where(red_groups == uv_positions.group_indices[i])
        amp_matrix[i, index1[0]] = 1
        amp_matrix[i, index2[0]] = 1
        amp_matrix[i, len(red_tiles) + index_group[0]] = 1

    return amp_matrix, red_tiles, red_groups


def sky_model_matrix_populator(uv_positions):
    # so first we sort out the unique antennas
    # and the unique redudant groups, this will allows us to populate the matrix adequately
    antenna_indices = numpy.stack((uv_positions.antenna_id1, uv_positions.antenna_id2))
    red_tiles = numpy.unique(antenna_indices)
    # it's not really finding unique antennas, it just finds unique values
    # create am empty matrix (#measurements)x(#tiles + #redundant groups)
    amp_matrix = numpy.zeros((uv_positions.number_of_baselines, len(red_tiles)))
    for i in range(uv_positions.number_of_baselines):
        index1 = numpy.where(red_tiles == uv_positions.antenna_id1[i])
        index2 = numpy.where(red_tiles == uv_positions.antenna_id2[i])
        amp_matrix[i, index1[0]] = 1
        amp_matrix[i, index2[0]] = 1

    return amp_matrix, red_tiles


def telescope_bounds(position_path, bound_type="redundant", nu=150e6, position_precision=1e-2,
                     broken_tile_fraction=0.3):
    telescope = RadioTelescope(load=True, path=position_path)
    number_antennas = telescope.antenna_positions.number_antennas()
    if bound_type == "redundant":
        redundant_table = redundant_baseline_finder(telescope.baseline_table)
        sky_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)

        redundant_crlb = relative_calibration_crlb(redundant_table, nu=nu, position_precision=position_precision,
                                                   broken_tile_fraction=broken_tile_fraction)
        absolute_crlb = absolute_calibration_crlb(sky_table, nu=150e6, position_precision=position_precision)
        bound = [numpy.median(numpy.diag(redundant_crlb)), absolute_crlb]
    elif bound_type == "sky":
        redundant_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)
        bound = numpy.median(numpy.diag(sky_calibration_crlb(redundant_table)))

    return number_antennas, bound


def plot_cramer_bound(redundant_data, sky_data, plot_path):
    # print("Redundant Calibration Errors")
    # print("HERA 350")
    # hera_350_antennas, hera_350_redundant = telescope_bounds("data/HERA_350.txt", bound_type="redundant")
    # print("HERA 128")
    # hera_128_antennas, hera_128_redundant = telescope_bounds("data/HERA_128.txt", bound_type="redundant")
    # print("MWA Hexes")
    # mwa_hexes_antennas, mwa_hexes_redundant = telescope_bounds("data/MWA_Hexes_Coordinates.txt", bound_type="redundant")
    #
    # print("")
    # print("Sky Model")
    # print("MWA Hexes")
    # mwa_hexes_antennas, mwa_hexes_sky = telescope_bounds("data/MWA_Hexes_Coordinates.txt", bound_type="sky")
    # print("MWA Compact")
    # mwa_compact_antennas, mwa_compact_sky = telescope_bounds("data/MWA_Compact_Coordinates.txt", bound_type="sky")
    # print("MWA Compact")
    # hera_350_antennas, hera_350_sky = telescope_bounds("data/HERA_350.txt", bound_type="sky")
    # print('SKA')
    # ska_low_antennas, ska_low_sky = telescope_bounds("data/SKA_Low_v5_ENU_fullcore.txt", bound_type="sky")

    fig, axes = pyplot.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(redundant_data[0, :], redundant_data[1, :] + redundant_data[2, :], label="Total")

    axes[0].plot(redundant_data[0, :], redundant_data[1, :], label="Relative")
    axes[0].plot(redundant_data[0, :], redundant_data[2, :], label="Absolute")
    axes[0].plot(redundant_data[0, :], redundant_data[3, :], "k--", label="Thermal")

    # axes[0].plot(hera_350_antennas, hera_350_redundant[0] + hera_350_redundant[1], marker ='H', label="HERA 350")
    # axes[0].plot(hera_128_antennas, hera_128_redundant[0] + hera_128_redundant[1], marker ="o", label=" HERA 128")
    # axes[0].plot(mwa_hexes_antennas, mwa_hexes_redundant[0] + mwa_hexes_redundant[1], marker="x", label="MWA Hexes")
    axes[0].set_ylabel("Gain Variance")
    axes[0].set_yscale('log')

    axes[1].semilogy(sky_data[0, :], sky_data[1, :], label="Sky Calibration")
    axes[1].semilogy(sky_data[0, :], sky_data[2, :], "k--", label="Thermal")

    # axes[1].plot(hera_350_antennas, hera_350_sky, marker ='H', label="HERA 350")
    # axes[1].plot(mwa_hexes_antennas, mwa_hexes_sky, marker ='x', label="MWA Hexes")
    # axes[1].plot(mwa_compact_antennas, mwa_compact_sky, marker ='+', label="MWA Compact")
    # axes[1].plot(ska_low_antennas, ska_low_sky, marker ='*', label="SKA_LOW1")

    axes[0].set_xlabel("Number of Antennas")
    axes[1].set_xlabel("Number of Antennas")

    axes[0].set_ylim([1e-7, 1])
    axes[1].set_ylim([1e-7, 1])

    axes[0].legend()
    axes[1].legend()
    fig.savefig(plot_path + "TEST.pdf")
    return


def test_plot():
    print("Redundant Calibration Errors")
    print("MWA Hexes")
    mwa_hexes_antennas, mwa_hexes_redundant = telescope_bounds("data/MWA_Hexes_Coordinates.txt", bound_type="redundant")

    print("Sky Model")
    print("MWA Hexes")
    mwa_hexes_antennas, mwa_hexes_sky = telescope_bounds("data/MWA_Hexes_Coordinates.txt", bound_type="sky")
    pyplot.semilogy(mwa_hexes_antennas, mwa_hexes_redundant[0], marker="+", label="Relative")
    pyplot.semilogy(mwa_hexes_antennas, mwa_hexes_redundant[1], marker="x", label="Absolute")
    pyplot.semilogy(mwa_hexes_antennas, mwa_hexes_sky, marker="*", label="Sky")
    pyplot.legend()
    pyplot.show()

    return


if __name__ == "__main__":
    test_plot()
    # cramer_rao_bound_comparison()
