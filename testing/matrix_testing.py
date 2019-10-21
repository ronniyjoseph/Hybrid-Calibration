import sys
import numpy
from scipy.constants import c
from scipy import sparse
from matplotlib import pyplot

sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
sys.path.append("../../redundant_calibration/code/")

from SCAR.GeneralTools import unique_value_finder
from analytic_covariance import moment_returner
from radiotelescope import beam_width
from radiotelescope import AntennaPositions
from radiotelescope import BaselineTable
from util import hexagonal_array
from cramer_rao_bound import sky_covariance
from cramer_rao_bound import redundant_baseline_finder

def main():
    position_precision = 1e-2
    broken_tile_fraction = 0.3
    maximum_factor = 2

    size_factor = numpy.arange(maximum_factor, maximum_factor + 1, 1)
    small_variance = numpy.zeros(len(size_factor))
    large_variance = numpy.zeros_like(small_variance)
    for i in range(len(size_factor)):
        print(f"At size factor {size_factor[i]}")
        antenna_positions = hexagonal_array(size_factor[i])
        antenna_table = AntennaPositions(load=False)
        antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
        antenna_table.x_coordinates = antenna_positions[:, 0]
        antenna_table.y_coordinates = antenna_positions[:, 1]
        antenna_table.z_coordinates = antenna_positions[:, 2]
        baseline_table = BaselineTable(position_table=antenna_table)

        redundant_baselines = redundant_baseline_finder(baseline_table.antenna_id1, baseline_table.antenna_id2,
                                                        baseline_table.u_coordinates, baseline_table.v_coordinates,
                                                        baseline_table.w_coordinates, group_minimum=2, verbose=False)

        small_mat = small_covariance_matrix(redundant_baselines)
        large_mat = large_covariance_matrix(redundant_baselines)

        small_variance[i] = small_mat
        large_variance[i] = large_mat
    pyplot.show()

    # pyplot.semilogy(size_factor, small_variance, label = "full")
    # pyplot.semilogy(size_factor, large_variance, label = "approx")
    # pyplot.legend()
    # pyplot.show()
    return


def small_covariance_matrix(redundant_baselines, nu=150e6, position_precision=1e-2):
    model_sky = numpy.sqrt(moment_returner(n_order=2, S_low=1))
    non_redundant_covariance = sky_covariance(numpy.array([0, position_precision / c * nu]),
                                              numpy.array([0, position_precision / c * nu]), nu)

    ideal_unmodeled_covariance = sky_covariance(redundant_baselines[:, 2], redundant_baselines[:, 3], nu)
    ideal_unmodeled_covariance = restructure_covariance_matrix(ideal_unmodeled_covariance,
                                                               diagonal=non_redundant_covariance[0, 0],
                                                               off_diagonal=non_redundant_covariance[0, 1])

    jacobian_vector = numpy.zeros(len(redundant_baselines[:, 0])) + model_sky

    fig, axes = pyplot.subplots(1,2, figsize = (10, 5))
    axes[0].imshow(ideal_unmodeled_covariance)
    axes[1].imshow(numpy.linalg.pinv(ideal_unmodeled_covariance))

    absolute_fim = numpy.dot(numpy.dot(jacobian_vector.T, numpy.linalg.pinv(ideal_unmodeled_covariance)),jacobian_vector)
    absolute_crlb = 2 * numpy.real(1 / (absolute_fim))

    return absolute_crlb


def large_covariance_matrix(redundant_baselines, nu=150e6, position_precision = 1e-2):
    absolute_fim = 0
    model_sky = numpy.sqrt(moment_returner(n_order=2, S_low=1))
    non_redundant_covariance = sky_covariance(numpy.array([0, position_precision / c * nu]),
                                              numpy.array([0, position_precision / c * nu]), nu)

    groups = numpy.unique(redundant_baselines[:, 5])
    print(len(groups))
    fig, axes = pyplot.subplots(2, len(groups), figsize=(5*len(groups), 10))
    for group_index in range(len(groups)):
        number_of_redundant_baselines = len(redundant_baselines[redundant_baselines[:, 5] == groups[group_index], 5])
        # print("")
        # print(f"Group {group_index} with {number_of_redundant_baselines} baselines ")

        redundant_block = numpy.zeros((number_of_redundant_baselines, number_of_redundant_baselines)) + non_redundant_covariance[0, 0]
        # print("Created block of ones")
        redundant_block = restructure_covariance_matrix(redundant_block, diagonal = non_redundant_covariance[0, 0],
                                                        off_diagonal=non_redundant_covariance[0, 1])

        # print("Restructured")
        # print("Inverted Matrices")
        jacobian_vector = numpy.zeros(number_of_redundant_baselines) + model_sky

        axes[0, group_index].imshow(redundant_block)
        axes[1, group_index].imshow(numpy.linalg.pinv(redundant_block))

        absolute_fim += numpy.dot(numpy.dot(jacobian_vector.T,  numpy.linalg.pinv(redundant_block)), jacobian_vector)
    absolute_crlb = 2 * numpy.real(1 / (absolute_fim))
    return absolute_crlb


def restructure_covariance_matrix(matrix, diagonal, off_diagonal):
    matrix /= diagonal
    matrix -= numpy.diag(numpy.zeros(matrix.shape[0]) + 1)
    matrix *= off_diagonal
    matrix += numpy.diag(numpy.zeros(matrix.shape[0]) + diagonal)

    return matrix


if __name__ == "__main__":
    main()