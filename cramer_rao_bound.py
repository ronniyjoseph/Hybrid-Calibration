import sys
import numpy
from scipy.constants import c
from scipy import sparse
from matplotlib import pyplot

sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
sys.path.append("../../redundant_calibration/code/")

from SCAR.RadioTelescope import redundant_baseline_finder
from SCAR.GeneralTools import unique_value_finder
from analytic_covariance import moment_returner
from radiotelescope import beam_width
from radiotelescope import AntennaPositions
from radiotelescope import BaselineTable
from util import hexagonal_array

def cramer_rao_bound_calculator():
    nu = 150e6
    maximum_factor = 8
    degree_of_redundancy = numpy.zeros(maximum_factor)
    red_gain_variance = numpy.zeros(maximum_factor)
    red_variance_spread = numpy.zeros(maximum_factor)

    mod_gain_variance = numpy.zeros(maximum_factor)
    mod_variance_spread = numpy.zeros(maximum_factor)
    redundant_sky = numpy.sqrt(moment_returner(n_order = 2))

    model_sky = numpy.sqrt(moment_returner(n_order =2, S_low = 1))

    for i in range(1, maximum_factor + 1):
        print(i)
        factor = i
        spacing = 4
        #telescope = RadioTelescope(load=False, shape=['square', factor*spacing, 1 + 2*factor, 0, 0])
        #baseline_table = telescope.baseline_table

        antenna_positions = hexagonal_array((i))
        antenna_table = AntennaPositions(load = False)
        antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
        antenna_table.x_coordinates = antenna_positions[:,0]
        antenna_table.y_coordinates = antenna_positions[:, 1]
        antenna_table.z_coordinates = antenna_positions[:, 2]
        baseline_table = BaselineTable(position_table = antenna_table)

        print("finding redundant baselines")
        redundant_baselines = redundant_baseline_finder(baseline_table.antenna_id1, baseline_table.antenna_id2,
                                                        baseline_table.u_coordinates, baseline_table.w_coordinates,
                                                        baseline_table.w_coordinates, verbose=False)
        print("populating matrixes")

        derivative_matrix, red_tiles, red_groups = LogcalMatrixPopulator(redundant_baselines)

        print("Doing calculations")
        redundant_matrix = derivative_matrix.copy()
        redundant_matrix[:, :len(red_tiles)] *= redundant_sky
        redundant_variance = beam_variance(nu) + position_variance(nu)

        redundant_FIM = numpy.dot(redundant_matrix.T, redundant_matrix)/redundant_variance
        redundant_CRLB= numpy.linalg.pinv(redundant_FIM)
        print(redundant_FIM.shape)

        model_covariance = sky_covariance(redundant_baselines[:, 0], redundant_baselines[:, 1],nu )
        model_matrix = derivative_matrix[:,:len(red_tiles)]*model_sky

        if model_covariance.shape[0] < 1e3:
            model_FIM = numpy.dot(numpy.dot(model_matrix.T, numpy.linalg.pinv(model_covariance)), model_matrix)
            model_CRLB= numpy.linalg.pinv(model_FIM)

        elif model_covariance.shape[0] > 1e3:
            print("sparse mode")
            model_FIM = sparse.csc_matrix(model_matrix.T)*sparse.linalg.inv(sparse.csc_matrix(model_covariance))*\
                        sparse.csc_matrix(model_matrix)

            model_CRLB= numpy.linalg.pinv(model_FIM.todense())


        # fig, axes = pyplot.subplots(2,3, figsize =(15, 10))
        # axes[0, 0].axis('off')
        # axes[0, 1].imshow(numpy.log10(redundant_FIM))
        # axes[0, 2].imshow(numpy.log10(model_CRLB))
        #
        # axes[1, 0].imshow(model_covariance)
        # axes[1, 1].imshow(model_FIM)
        # axes[1, 2].imshow(model_CRLB)
        # pyplot.show()
        degree_of_redundancy[i - 1] = factor#1- len(red_groups)/len(redundant_baselines[:,0])
        red_gain_variance[i - 1] = numpy.median(numpy.diag(redundant_CRLB)[:len(red_tiles)])
        red_variance_spread[i - 1] = numpy.std(numpy.diag(redundant_CRLB)[:len(red_tiles)])

        mod_gain_variance[i - 1] = numpy.median(numpy.diag(model_CRLB))
        mod_variance_spread[i - 1] = numpy.std(numpy.diag(model_CRLB))


    fig, axes = pyplot.subplots(2,2, figsize = (10,5))
    axes[0,0].plot(degree_of_redundancy, red_gain_variance)
    axes[0,1].plot(degree_of_redundancy, red_variance_spread)

    axes[1,0].plot(degree_of_redundancy, mod_gain_variance)
    axes[1,1].plot(degree_of_redundancy, mod_variance_spread)

    axes[0,0].set_ylabel("Redundant Gain Variance")
    axes[0,1].set_ylabel("Variance of the Variance")
    axes[1,0].set_ylabel("Sky Gain Variance")
    axes[1,1].set_ylabel("Variance of the Variance")


    axes[1,0].set_xlabel("Degree of Redundancy")
    axes[1,1].set_xlabel("Degree of Redundancy")
    axes[0,0].set_yscale('log')
    axes[0,1].set_yscale('log')
    axes[1,0].set_yscale('log')
    axes[1,1].set_yscale('log')

    pyplot.show()
    return

def sky_covariance(u, v, nu, S_low=0.1, S_mid=1, S_high=1):

    uu1, uu2 = numpy.meshgrid(u, u)
    vv1, vv2 = numpy.meshgrid(v, v)

    width_tile = beam_width(nu)

    mu_2_r = moment_returner(2, S_low=S_low, S_mid=S_mid, S_high=S_high)

    sky_covariance = 2 * numpy.pi * mu_2_r * width_tile**2/2 * numpy.exp(-numpy.pi ** 2 *width_tile**2*((uu1-uu2) ** 2 + (vv1 - vv2) ** 2))

    return sky_covariance

def beam_variance(nu, N=16, dx = 1, gamma = 0.8):

    mu_2 = moment_returner(n_order = 2)
    tile_beam_width = beam_width(nu)
    dipole_beam_width= beam_width(nu, diameter=1)

    sigma = 0.5*(tile_beam_width**2*dipole_beam_width**2)/(tile_beam_width**2 + dipole_beam_width**2)

    variance = 2*numpy.pi*mu_2*sigma**2/(2*N**2)
    return variance


def position_variance(nu, position_precision = 10e-2):
    mu_2 = moment_returner(n_order = 2)
    tile_beam_width = beam_width(nu)

    variance = (2*numpy.pi)**5*mu_2*(position_precision*c/nu)**2*tile_beam_width**2
    return variance


def redundant_baseline_finder(antenna_id1, antenna_id2, u, v, w,  baseline_direction = None,verbose=False):
    """
	"""
    ################################################################
    minimum_baselines = 3.
    wave_fraction = 1. / 6
    ################################################################

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
        separation = numpy.sqrt((u - u[0]) ** 2. +(v - v[0]) ** 2.)
        # find all baselines within the lambda fraction
        select_indices = numpy.where(separation <= wave_fraction)

        # is this number larger than the minimum number
        if len(select_indices[0]) >= minimum_baselines:
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
    baseline_lengths = numpy.sqrt(baseline_selection[:, 2] ** 2 \
                                  + baseline_selection[:, 3] ** 2)

    sorted_baselines = baseline_selection[numpy.argsort(baseline_lengths), :]

    sorted_baselines = baseline_selection[numpy.argsort(sorted_baselines[:, 5]), :]
    # sorted_baselines = sorted_baselines[numpy.argsort(sorted_baselines[:,1,middle_index]),:,:]
    # if we want only the EW select all the  uv positions around v = 0
    return sorted_baselines


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
