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


def cramer_rao_bound_calculator(maximum_factor = 6, nu = 150e6, verbose = True):

    size_factor = numpy.arange(2 , maximum_factor + 1, 1)

    #Initialise empty Arrays for relative calibration results
    redundancy_metric = numpy.zeros(len(size_factor))
    relative_gain_variance = numpy.zeros_like(redundancy_metric)
    relative_gain_spread = numpy.zeros_like(redundancy_metric)

    #Initialise empty arrays for absolute calibration results
    absolute_gain_variance = numpy.zeros_like(redundancy_metric)

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

        redundant_baselines = redundant_baseline_finder(baseline_table.antenna_id1, baseline_table.antenna_id2,
                                                        baseline_table.u_coordinates, baseline_table.v_coordinates,
                                                        baseline_table.w_coordinates, verbose=False)
        if verbose:
            print("Populating matrices")

        redundant_crlb = relative_calibration_crlb(redundant_baselines, nu = nu)
        absolute_crlb = absolute_calibration_crlb(redundant_baselines, nu=150e6)

        redundancy_metric[i] = len(antenna_positions[:, 0])
        relative_gain_variance[i] = numpy.median(numpy.diag(redundant_crlb))
        relative_gain_spread[i] = numpy.std(numpy.diag(redundant_crlb))

        absolute_gain_variance[i] = absolute_crlb


    full_gain_variance = absolute_gain_variance + relative_gain_variance
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(redundancy_metric, relative_gain_variance, label = "Relative Cal")
    axes[0].set_ylabel("Gain Variance")
    axes[0].set_yscale('log')

    axes[1].plot(redundancy_metric, relative_gain_spread)
    axes[1].set_ylabel("Variance of relative variance")
    axes[1].set_yscale('log')

    axes[0].plot(redundancy_metric, absolute_gain_variance, label = "Absolute Cal")

    axes[0].plot(redundancy_metric, full_gain_variance, label = "Full Cal")

    axes[0].set_xlabel("Number of Antennas")
    axes[1].set_xlabel("Number of Antennas")
    axes[0].legend()

    pyplot.show()
    return


def relative_calibration_crlb(redundant_baselines, nu = 150e6):
    redundant_sky = numpy.sqrt(moment_returner(n_order=2))
    jacobian_gain_matrix, red_tiles, red_groups = LogcalMatrixPopulator(redundant_baselines)

    jacobian_gain_matrix[:, :len(red_tiles)] *= redundant_sky
    non_redundancy_variance = beam_variance(nu) + position_variance(nu)

    redundant_fisher_information = numpy.dot(jacobian_gain_matrix.T, jacobian_gain_matrix) / non_redundancy_variance
    redundant_crlb = 2*numpy.real(numpy.linalg.pinv(redundant_fisher_information))

    return redundant_crlb[:len(red_tiles), :len(red_tiles)]


def absolute_calibration_crlb(redundant_baselines, nu=150e6):
    if len(redundant_baselines) < 5000:
        absolute_crlb =  small_covariance_matrix(redundant_baselines, nu)
    elif len(redundant_baselines) > 5000:
        print("Going Block Mode")
        absolute_crlb = large_covariance_matrix(redundant_baselines, nu)

    return absolute_crlb

def small_covariance_matrix(redundant_baselines, nu = 150e6):
    model_sky = numpy.sqrt(moment_returner(n_order=2, S_low=1))
    unmodeled_covariance = sky_covariance(redundant_baselines[:, 2], redundant_baselines[:, 3], nu)
    jacobian_vector = numpy.zeros(len(redundant_baselines[:, 0])) + model_sky

    absolute_fim = numpy.dot(numpy.dot(jacobian_vector.T, numpy.linalg.pinv(unmodeled_covariance)), jacobian_vector)
    absolute_crlb = 2 * numpy.real(1 / absolute_fim)
    return absolute_crlb

def large_covariance_matrix(redundant_baselines, nu = 150e6):

    absolute_fim = 0
    model_sky = numpy.sqrt(moment_returner(n_order=2, S_low=1))
    groups = numpy.unique(redundant_baselines[:, 5])
    for group_index in range(len(groups)):
        print(group_index)
        number_of_redundant_baselines = len(redundant_baselines[:, 5] == groups[group_index])
        redundant_block = numpy.zeros((number_of_redundant_baselines, number_of_redundant_baselines))
        redundant_block += sky_covariance(numpy.array([0]), numpy.array([0]), nu)

        inverse_block = numpy.linalg.pinv(redundant_block)
        absolute_fim += numpy.sum(inverse_block*model_sky**2)

    absolute_crlb = 2 * numpy.real(1 / absolute_fim)
    return absolute_crlb


def sky_covariance(u, v, nu, S_low=0.1, S_mid=1, S_high=1):
    uu1, uu2 = numpy.meshgrid(u, u)
    vv1, vv2 = numpy.meshgrid(v, v)

    width_tile = beam_width(nu)

    mu_2_r = moment_returner(2, S_low=S_low, S_mid=S_mid, S_high=S_high)

    sky_covariance = 2 * numpy.pi * mu_2_r * width_tile ** 2 * numpy.exp(
        -numpy.pi ** 2 * width_tile ** 2 * ((uu1 - uu2) ** 2 + (vv1 - vv2) ** 2))

    return sky_covariance


def beam_variance(nu, N=16, dx=1, gamma=0.8):
    mu_2 = moment_returner(n_order=2)
    tile_beam_width = beam_width(nu)
    dipole_beam_width = beam_width(nu, diameter=1)

    sigma = 0.5 * (tile_beam_width ** 2 * dipole_beam_width ** 2) / (tile_beam_width ** 2 + dipole_beam_width ** 2)

    variance = 2 * numpy.pi * mu_2 * sigma ** 2 / (2 * N ** 2)
    return variance*0.3**2


def position_variance(nu, position_precision=10e-3):
    mu_2 = moment_returner(n_order=2)
    tile_beam_width = beam_width(nu)

    variance = (2 * numpy.pi) ** 5 * mu_2 * (position_precision * c / nu) ** 2 * tile_beam_width ** 2
    return variance


def redundant_baseline_finder(antenna_id1, antenna_id2, u, v, w, baseline_direction=None, group_minimum=3,
                              threshold=1 / 6, verbose=False):
    """

    antenna_id1:
    antenna_id2:
    u:
    v:
    w:
    baseline_direction:
    group_minimum:
    threshold:
    verbose:
    :return:
    """
    n_baselines = u.shape[0]
    # create empty table
    baseline_selection = numpy.zeros((n_baselines, 6))
    # arbitrary counters
    # Let's find all the redundant baselines within our threshold
    group_counter = 0
    k = 0
    # Go through all antennas, take each antenna out and all antennas
    # which are part of the not redundant enough group
    while u.shape[0] > 0:
        # calculate uv separation at the calibration wavelength
        separation = numpy.sqrt((u - u[0]) ** 2. + (v - v[0]) ** 2.)
        # find all baselines within the lambda fraction
        select_indices = numpy.where(separation <= threshold)

        # is this number larger than the minimum number
        if len(select_indices[0]) >= group_minimum:
            # go through the selected baselines

            for i in range(len(select_indices[0])):
                # add antenna number
                baseline_selection[k, 0] = antenna_id1[select_indices[0][i]]
                baseline_selection[k, 1] = antenna_id2[select_indices[0][i]]
                # add coordinates uvw
                baseline_selection[k, 2] = u[select_indices[0][i]]
                baseline_selection[k, 3] = v[select_indices[0][i]]
                baseline_selection[k, 4] = w[select_indices[0][i]]
                # add baseline group identifier
                baseline_selection[k, 5] = 50000000 + 52 * (group_counter + 1)
                k += 1
            group_counter += 1
        # update the list, take out the used antennas
        all_indices = numpy.arange(len(u))
        unselected_indices = numpy.setdiff1d(all_indices, select_indices[0])

        antenna_id1 = antenna_id1[unselected_indices]
        antenna_id2 = antenna_id2[unselected_indices]
        u = u[unselected_indices]
        v = v[unselected_indices]
        w = w[unselected_indices]

    if verbose:
        print("There are", k, "redundant baselines in this array.")
        print("There are", group_counter, "redundant groups in this array")

    # find the filled entries
    non_zero_indices = numpy.where(baseline_selection[:, 0] != 0)
    # remove the empty entries
    baseline_selection = baseline_selection[non_zero_indices[0], :]
    # Sort on length
    baseline_lengths = numpy.sqrt(baseline_selection[:, 2] ** 2 + baseline_selection[:, 3] ** 2)

    # sorted_baselines = baseline_selection[numpy.argsort(baseline_lengths), :]

    # sorted_baselines = baseline_selection[numpy.argsort(sorted_baselines[:, 5]), :]
    # sorted_baselines = sorted_baselines[numpy.argsort(sorted_baselines[:,1,middle_index]),:,:]
    # if we want only the EW select all the  uv positions around v = 0
    return baseline_selection


def LogcalMatrixPopulator(uv_positions):
    # so first we sort out the unique antennas
    # and the unique redudant groups, this will allows us to populate the matrix adequately
    red_tiles = unique_value_finder(uv_positions[:, 0:2], 'values')
    # it's not really finding unique antennas, it just finds unique values
    red_groups = unique_value_finder(uv_positions[:, 5], 'values')
    # print "There are", len(red_tiles), "redundant tiles"
    # print ""
    # print "Creating the equation matrix"
    # create am empty matrix (#measurements)x(#tiles + #redundant groups)
    amp_matrix = numpy.zeros((len(uv_positions), len(red_tiles) + len(red_groups)))
    for i in range(len(uv_positions)):
        index1 = numpy.where(red_tiles == uv_positions[i, 0])
        index2 = numpy.where(red_tiles == uv_positions[i, 1])
        index_group = numpy.where(red_groups == uv_positions[i, 5])

        amp_matrix[i, index1[0]] = 1
        amp_matrix[i, index2[0]] = 1
        amp_matrix[i, len(red_tiles) + index_group[0]] = 1

    return amp_matrix, red_tiles, red_groups


if __name__ == "__main__":
    cramer_rao_bound_calculator()
