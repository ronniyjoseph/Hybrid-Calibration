import sys
import numpy
import os
from scipy.constants import c
from scipy import sparse
import matplotlib
# matplotlib.use("Agg")
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


def cramer_rao_bound_comparison(maximum_factor=10, nu=150e6, verbose=True, compute_data=True, load_data=True,
                                save_output=True, make_plot=True, show_plot=False):
    """

    Parameters
    ----------
    maximum_factor  :   int
                        Size factor of the hexagonal array - how many hexagonal layers
    nu              :   float
                        Frequency for calculation in MHz
    verbose         :   bool
                        True of False if you want stderr output
    compute_data    :   bool
                        Flag to compute data for this settings
    load_data       :   bool
                        Flag to load previously computed data
    save_output     :   bool
                        Flag to save computed data
    make_plot       :   bool
                        Flag to plot and save plot of computed or loaded data
    show_plot       :   bool
                        Flag to show plot interactively

    Returns
    -------

    """

    position_precision = 1e-2
    broken_tile_fraction = 0.3
    sky_model_limit = 1e-3
    output_path = "/data/rjoseph/Hybrid_Calibration/TEST/"
    if not os.path.exists(output_path + "/"):
        print("Creating Project folder at output destination!")
        os.makedirs(output_path)

    if compute_data:
        redundant_data, sky_data = cramer_rao_bound_calculator(maximum_factor, position_precision, broken_tile_fraction,
                                                               sky_model_depth = sky_model_limit, nu=nu,
                                                               verbose=verbose)
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


def cramer_rao_bound_calculator(maximum_factor=3, position_precision=1e-2, broken_tile_fraction=0.3,
                                sky_model_depth = 1.0, nu=150e6, verbose=True):
    """

    Parameters
    ----------
    maximum_factor          :   int
                            Maximum size factor of the hexagonal array - how many hexagonal layers do you want to go up
                            to
    position_precision      :   float
                            Set the position precision of the antennas in meters
    broken_tile_fraction    : float
                            Set the fraction of broken tiles to down weight the covariances
    nu                      :   float
                            Set the frequency of the observations
    verbose                 :   bool
                            Flag to set if outputs to STDERR are desired

    Returns

    -------

    """

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
        absolute_crlb = absolute_calibration_crlb(skymodel_baselines, nu=150e6, position_precision=position_precision,
                                                  sky_model_depth=sky_model_depth)
        sky_crlb = sky_calibration_crlb(skymodel_baselines, position_precision=position_precision,
                                        sky_model_depth=sky_model_depth)

        # Save data into arrays
        redundancy_metric[i] = antenna_table.number_antennas()
        relative_gain_variance[i] = numpy.median(numpy.diag(redundant_crlb))
        relative_gain_spread[i] = numpy.std(numpy.diag(redundant_crlb))
        absolute_gain_variance[i] = absolute_crlb
        thermal_redundant_variance[i] = numpy.median(numpy.diag(thermal_redundant_crlb(redundant_baselines)))

        sky_gain_variance[i] = numpy.median(numpy.diag(sky_crlb))
        thermal_sky_variance[i] = numpy.median(numpy.diag(thermal_sky_crlb(skymodel_baselines)))

    redundant_data = numpy.stack((redundancy_metric, relative_gain_variance, absolute_gain_variance,
                                  thermal_redundant_variance))
    sky_data = numpy.stack((redundancy_metric, sky_gain_variance, thermal_sky_variance))

    return redundant_data, sky_data


def thermal_redundant_crlb(redundant_baselines, nu=150e6, SEFD=20e3, B=40e3, t=120):
    """

    Parameters
    ----------
    redundant_baselines : object
                        a radiotelescope object containing the baseline table for the redundant baselines
    nu                  : float
                        The frequency at which you want to compute the gains
    SEFD                : float
                        System Equivalent Flux Density of the array in Jy
    B                   : float
                        Frequency Bandwidth in MHZ
    t                   : float
                        Integration time for the calibration observation

    Returns
    -------

    """
    redundant_sky = numpy.sqrt(sky_moment_returner(n_order=2))
    jacobian_gain_matrix, red_tiles, red_groups = redundant_matrix_populator(redundant_baselines)

    jacobian_gain_matrix[:, :len(red_tiles)] *= redundant_sky
    thermal_noise = thermal_variance()
    redundant_fisher_information = numpy.dot(jacobian_gain_matrix.T, jacobian_gain_matrix) / thermal_noise

    redundant_crlb = 2 * numpy.real(numpy.linalg.pinv(redundant_fisher_information))
    return redundant_crlb[:len(red_tiles), :len(red_tiles)]


def absolute_calibration_crlb(redundant_baselines, position_precision=1e-2, sky_model_depth=1.0, nu=150e6,
                              verbose=True):
    """

    Parameters
    ----------
    sky_model_depth
    redundant_baselines :   object
                         a radiotelescope object containing the baseline table for the redundant baselines
    position_precision  :
    nu

    Returns
    -------

    """

    if verbose:
        print("Computing Absolute Calibration CRLB")

    sky_based_model = numpy.sqrt(sky_moment_returner(n_order=2, s_low=sky_model_depth, s_high=10))
    jacobian_vector = numpy.zeros(redundant_baselines.number_of_baselines) + sky_based_model
    uv_scales = numpy.array([0, position_precision / c * nu])
    non_redundant_block = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, S_high=sky_model_depth,
                                         mode='baseline')
    non_redundant_block += numpy.diag(numpy.zeros(len(uv_scales)) + thermal_variance())
    if redundant_baselines.number_of_baselines < 5000:
        ideal_covariance = sky_covariance(nu=nu, u=redundant_baselines.u_coordinates,
                                          v=redundant_baselines.v_coordinates, S_high=sky_model_depth,
                                          mode='baseline')
        ideal_covariance += numpy.diag(numpy.zeros(redundant_baselines.number_of_baselines) + thermal_variance())

        absolute_crlb = small_matrix(jacobian_vector, non_redundant_block, ideal_covariance)
    elif redundant_baselines.number_of_baselines > 5000:
        absolute_crlb = large_matrix(redundant_baselines, jacobian_vector, non_redundant_block)
    return absolute_crlb


def relative_calibration_crlb(redundant_baselines, nu=150e6, position_precision=1e-2, broken_tile_fraction=1.0,
                              verbose =True):
    """

    Parameters
    ----------
    redundant_baselines
    nu
    position_precision
    broken_tile_fraction

    Returns
    -------

    """
    if verbose:
        print("Computing Relative Calibration CRLB")

    # Compute the signal
    redundant_sky = numpy.sqrt(sky_moment_returner(n_order=2))
    # Compute the Jacobian matrix that determines which measurements contribute to the Fisher Information
    jacobian_gain_matrix, red_tiles, red_groups = redundant_matrix_populator(redundant_baselines)
    jacobian_gain_matrix[:, :len(red_tiles)] *= redundant_sky

    # Compute a 2x2 covariance block
    uv_scales = numpy.array([0, position_precision/c*nu])
    non_redundant_block = beam_covariance(nu, u=uv_scales, v=uv_scales, broken_tile_fraction=broken_tile_fraction,
                                               mode='baseline') + \
                               position_covariance(nu, u=uv_scales, v=uv_scales, position_precision=position_precision,
                                                   mode='baseline')
    non_redundant_block += numpy.diag(numpy.zeros(len(uv_scales)) + thermal_variance())
    if redundant_baselines.number_of_baselines < 5000:
        beam_error = beam_covariance(nu, u=redundant_baselines.u(nu), v=redundant_baselines.v(nu),
                                     broken_tile_fraction=broken_tile_fraction, mode='baseline')
        position_error = position_covariance(nu, u=redundant_baselines.u(nu), v=redundant_baselines.v(nu),
                                             position_precision=position_precision, mode='baseline')
        thermal_noise = numpy.diag(numpy.zeros(redundant_baselines.number_of_baselines) + thermal_variance())
        ideal_covariance = position_error + beam_error + thermal_noise
        redundant_crlb = small_matrix(jacobian_gain_matrix, non_redundant_block, ideal_covariance)
    elif redundant_baselines.number_of_baselines > 5000:
        redundant_crlb = large_matrix(redundant_baselines, jacobian_gain_matrix, non_redundant_block)

    return redundant_crlb[:len(red_tiles), :len(red_tiles)]


def thermal_sky_crlb(redundant_baselines, nu=150e6, SEFD=20e3, B=40e3, t=120):

    """
    Parameters
    ----------
    redundant_baselines : array_like
                         a radiotelescope object containing the baseline table for the redundant baselines
    nu                  : float
                        The frequency at which you want to compute the gain variance in MHz
    SEFD                : float
                        The system equivalent flux density of the array in Jy
    B                   : float
                        Calibration bandwidth in MHz
    t                   : float
                        Observation integration time

    Returns
    -------

    """
    sky_based_model = numpy.sqrt(sky_moment_returner(n_order=2, s_low=1, s_high=10))
    antenna_baseline_matrix, red_tiles, red_groups = redundant_matrix_populator(redundant_baselines)

    jacobian_gain_matrix = antenna_baseline_matrix[:, :len(red_tiles)] * sky_based_model
    thermal_noise = thermal_variance()
    redundant_fisher_information = numpy.dot(jacobian_gain_matrix.T, jacobian_gain_matrix) / thermal_noise

    redundant_crlb = 2 * numpy.real(numpy.linalg.pinv(redundant_fisher_information))
    return redundant_crlb[:len(red_tiles), :len(red_tiles)]


def sky_calibration_crlb(redundant_baselines, nu=150e6, position_precision=1e-2, broken_tile_fraction=1,
                         sky_model_depth=1, verbose=True):

    """

    Parameters
    ----------
    redundant_baselines     : object
                            a radiotelescope object containing the baseline table for the redundant baselines
    nu                      : float
                            Frequency of observations in MHz
    position_precision      : float
                            Array position precision in metres
    broken_tile_fraction    : float
                            Fraction of tiles that have broken dipole

    Returns
    -------

    """

    if verbose:
        print("Computing Sky Calibration CRLB")

    sky_based_model = numpy.sqrt(sky_moment_returner(n_order=2, s_low=1, s_high=10))
    antenna_baseline_matrix, red_tiles = sky_model_matrix_populator(redundant_baselines)

    uv_scales = numpy.array([0, position_precision / c * nu])
    non_redundant_covariance = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, S_high=sky_model_depth, mode='baseline')
    non_redundant_covariance += numpy.diag(numpy.zeros(len(uv_scales)) + thermal_variance())
    jacobian_matrix = antenna_baseline_matrix[:, :len(red_tiles)] * sky_based_model
    if redundant_baselines.number_of_baselines < 5000:
        ideal_covariance = sky_covariance(nu, u = redundant_baselines.u(nu), v = redundant_baselines.v(nu),
                                          S_high=sky_model_depth, mode = 'baseline')
        ideal_covariance += numpy.diag(numpy.zeros(redundant_baselines.number_of_baselines) + thermal_variance())
        sky_crlb = small_matrix(jacobian_matrix, non_redundant_covariance, ideal_covariance)
    elif redundant_baselines.number_of_baselines > 5000:
        sky_crlb = large_matrix(redundant_baselines, jacobian_matrix, non_redundant_covariance)
    return sky_crlb


def small_matrix(jacobian, non_redundant_covariance, ideal_covariance):
    """

    Parameters
    ----------
    jacobian
    non_redundant_covariance
    ideal_covariance

    Returns
    -------

    """
    covariance_matrix = restructure_covariance_matrix(ideal_covariance, diagonal= non_redundant_covariance[0, 0],
                                                      off_diagonal=non_redundant_covariance[0, 1])
    fisher_information = compute_fisher_information(covariance_matrix, jacobian)
    cramer_rao_lower_bound = compute_cramer_rao_lower_bound(fisher_information)

    return cramer_rao_lower_bound


def large_matrix(baseline_table, jacobian_matrix, perturbed_covariance):
    """

    Parameters
    ----------
    baseline_table
    jacobian_matrix
    perturbed_covariance

    Returns
    -------

    """

    groups = numpy.unique(baseline_table.group_indices)
    fisher_information = 0
    for group_index in range(len(groups)):
        # Determine which baselines are part of the group
        group_visibilities_indices = numpy.where(baseline_table.group_indices == groups[group_index])[0]
        # Determine the size of the group
        number_of_group_baselines = len(group_visibilities_indices)

        if number_of_group_baselines == 1:
            # Compute FIM for a single baseline
            fisher_information += numpy.dot(jacobian_matrix[group_visibilities_indices, ...].T,
                                            jacobian_matrix[group_visibilities_indices, ...]) / perturbed_covariance[0, 0]
        elif number_of_group_baselines > 1:
            group_start_index = numpy.min(group_visibilities_indices)
            group_end_index = numpy.max(group_visibilities_indices)

            # Create a perfectly redundant block
            covariance_block = numpy.zeros((number_of_group_baselines, number_of_group_baselines)) + \
                             perturbed_covariance[0, 0]
            # Perturb the redundancy
            covariance_block = restructure_covariance_matrix(covariance_block, diagonal=perturbed_covariance[0, 0],
                                                           off_diagonal=perturbed_covariance[0, 1])
            jacobian_block = jacobian_matrix[group_start_index:group_end_index + 1, ...]
            # Compute FIM for a group of baselines
            fisher_information += compute_fisher_information(covariance_block, jacobian_block)
    cramer_rao_lower_bound = compute_cramer_rao_lower_bound(fisher_information)

    return cramer_rao_lower_bound


def compute_fisher_information(covariance_matrix, jacobian, verbose =True):
    if verbose:
        print(f"\tCovariance matrix condition number {numpy.linalg.cond(covariance_matrix)}")
    fisher_information = numpy.dot(numpy.dot(jacobian.T, numpy.linalg.pinv(covariance_matrix)), jacobian)

    return fisher_information


def compute_cramer_rao_lower_bound(fisher_information, verbose =True):
    if type(fisher_information) == numpy.ndarray:
        cramer_rao_lower_bound = 2*numpy.real(numpy.linalg.pinv(fisher_information))
        if verbose:
            print(f"\tFIM condition number {numpy.linalg.cond(fisher_information)}")
    else:
        cramer_rao_lower_bound = 2 * numpy.real(1 / fisher_information)

    return cramer_rao_lower_bound


def restructure_covariance_matrix(matrix, diagonal, off_diagonal):
    """

    Parameters
    ----------
    matrix
    diagonal
    off_diagonal

    Returns
    -------

    """

    matrix /= diagonal
    matrix -= numpy.diag(numpy.zeros(matrix.shape[0]) + 1)
    matrix *= off_diagonal
    matrix += numpy.diag(numpy.zeros(matrix.shape[0]) + diagonal)

    return matrix


def redundant_matrix_populator(uv_positions):
    """

    Parameters
    ----------
    uv_positions

    Returns
    -------

    """

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
    """

    Parameters
    ----------
    uv_positions

    Returns
    -------

    """
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


def telescope_bounds(position_path, bound_type="redundant", nu=150e6, position_precision=1e-2, broken_tile_fraction=0.3,
                     sky_model_depth = 1):
    """

    Parameters
    ----------
    position_path
    bound_type
    nu
    position_precision
    broken_tile_fraction

    Returns
    -------

    """

    telescope = RadioTelescope(load=True, path=position_path)
    number_antennas = telescope.antenna_positions.number_antennas()
    if bound_type == "redundant":
        redundant_table = redundant_baseline_finder(telescope.baseline_table)
        sky_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)
        print("Relative Bounds")
        redundant_crlb = relative_calibration_crlb(redundant_table, nu=nu, position_precision=position_precision,
                                                   broken_tile_fraction=broken_tile_fraction)
        print("Absolute Bounds")
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
    axes[0].plot(redundant_data[0, :], numpy.sqrt(redundant_data[1, :] + redundant_data[2, :]), label="Total")

    axes[0].plot(redundant_data[0, :], numpy.sqrt(redundant_data[1, :]), label="Relative")
    axes[0].plot(redundant_data[0, :], numpy.sqrt(redundant_data[2, :]), label="Absolute")
    axes[0].plot(redundant_data[0, :], numpy.sqrt(redundant_data[3, :]), "k--", label="Thermal")

    # axes[0].plot(mwa_hexes_antennas, numpy.sqrt(mwa_hexes_redundant[0] + mwa_hexes_redundant[1]), marker="x", label="MWA Hexes")
    # axes[0].plot(hera_128_antennas, numpy.sqrt(hera_128_redundant[0] + hera_128_redundant[1]), marker="o", label="HERA 128")
    # axes[0].plot(hera_350_antennas, numpy.sqrt(hera_350_redundant[0] + hera_350_redundant[1]), marker='H', label="HERA 350")
    axes[0].set_ylabel("Gain Variance")
    axes[0].set_yscale('log')

    axes[1].semilogy(sky_data[0, :], numpy.sqrt(sky_data[1, :]), label="Sky Calibration")
    axes[1].semilogy(sky_data[0, :], numpy.sqrt(sky_data[2, :]), "k--", label="Thermal")

    # axes[1].plot(mwa_hexes_antennas, numpy.sqrt(mwa_hexes_sky), marker='x', label="MWA Hexes")
    # axes[1].plot(hera_350_antennas, numpy.sqrt(hera_350_sky), marker='H', label="HERA 350")
    # axes[1].plot(mwa_compact_antennas, numpy.sqrt(mwa_compact_sky), marker='+', label="MWA Compact")
    # axes[1].plot(ska_low_antennas, numpy.sqrt(ska_low_sky), marker='*', label="SKA_LOW1")

    axes[0].set_xlabel("Number of Antennas")
    axes[1].set_xlabel("Number of Antennas")

    axes[0].set_ylim([1e-4, 1])
    axes[1].set_ylim([1e-4, 1])

    axes[0].legend()
    axes[1].legend()
    fig.savefig(plot_path + "TEST.pdf")
    return


if __name__ == "__main__":
    # test_plot()
    cramer_rao_bound_comparison()
