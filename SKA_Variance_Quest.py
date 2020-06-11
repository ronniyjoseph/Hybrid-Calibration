import sys
import numpy
import os
import argparse

from scipy.constants import c
from src.util import hexagonal_array
from src.util import redundant_baseline_finder
from src.radiotelescope import AntennaPositions
from src.radiotelescope import BaselineTable
from src.radiotelescope import RadioTelescope
from src.covariance import sky_covariance
from src.covariance import beam_covariance
from src.covariance import position_covariance
from src.covariance import thermal_variance
from src.skymodel import sky_moment_returner

from cramer_rao_bound import small_matrix
from cramer_rao_bound import large_matrix
from cramer_rao_bound import sky_model_matrix_populator
from cramer_rao_bound import compute_cramer_rao_lower_bound
from cramer_rao_bound import restructure_covariance_matrix

def main(    position_precision = 1e-2, broken_tile_fraction = 0.3, sky_model_depth = 1e-1):
    # telescope = RadioTelescope(load=True, path="data/SKA_Low_v5_ENU_mini.txt")
    # baselines = telescope.baseline_table
    # lengths = numpy.sqrt(baselines.u_coordinates**2 + baselines.v_coordinates**2)
    # telescope_bounds("data/SKA_Low_v5_ENU_mini.txt", bound_type="sky")
    # mwa_hexes_sky = telescope_bounds("data/MWA_Hexes_Coordinates.txt", bound_type="sky")
    # print("MWA Compact")
    # mwa_compact_sky = telescope_bounds("data/MWA_Compact_Coordinates.txt", bound_type="sky")

    # telescope = RadioTelescope(load=True, path=position_table_path)
    # print("Grouping baselines")
    # redundant_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)
    #
    # print(f"Ratio Number of groups/number of baselines:"
    #       f"{len(numpy.unique(redundant_table.group_indices))/len(redundant_table.antenna_id1)}")
    # pyplot.scatter(redundant_table.u_coordinates, redundant_table.v_coordinates, c = redundant_table.group_indices,
    #                cmap="Set3")
    # pyplot.show()

    # telescope = RadioTelescope(load=True, path="data/MWA_Hexes_Coordinates.txt")
    # telescope = RadioTelescope(load=True, path="data/SKA_Low_v5_ENU_fullcore.txt")
    telescope = RadioTelescope(load=True, path="data/SKA_Low_v5_ENU_mini.txt")

    redundant_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1, threshold=50)
    small_FIM, small_crlb, large_FIM, large_crlb = TEST_sky_calibration_crlb(redundant_table, sky_model_depth=sky_model_depth,
                         broken_tile_fraction=broken_tile_fraction,
                         position_precision=position_precision)

    # print(numpy.sqrt(numpy.median(numpy.diag(pure_crlb))))
    print(numpy.sqrt(numpy.median(numpy.diag(small_crlb))))
    print(numpy.sqrt(numpy.median(numpy.diag(large_crlb))))
    figure, axes = pyplot.subplots(2,4, figsize = (20, 10))

    # axes[0, 0].imshow(pure_FIM)
    axes[0, 1].imshow(small_FIM)
    axes[0, 2].imshow(large_FIM)
    axes[0, 3].imshow(small_FIM - large_FIM)

    # axes[1, 0].imshow(pure_crlb)
    axes[1, 1].imshow(small_crlb)
    axes[1, 2].imshow(large_crlb)
    axes[1, 3].imshow(small_crlb - large_crlb)
    pyplot.show()
    return


def TEST_sky_calibration_crlb(redundant_baselines, nu=150e6, position_precision=1e-2, broken_tile_fraction=1,
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

    model_covariance = sky_covariance(u=numpy.array([0,0]), v=numpy.array([0,0]), nu=nu, mode='baseline',
                                     S_low=sky_model_depth)
    sky_based_model = numpy.sqrt(model_covariance[0,0])

    antenna_baseline_matrix, red_tiles = sky_model_matrix_populator(redundant_baselines)

    uv_scales = numpy.array([0, position_precision / c * nu])
    sky_block_covariance = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, S_high=sky_model_depth, mode='baseline')
    beam_block_covariance = beam_covariance(nu=nu, u=uv_scales, v=uv_scales, broken_tile_fraction=broken_tile_fraction,
                                            mode='baseline', calibration_type='sky', model_limit=sky_model_depth)
    non_redundant_covariance = sky_block_covariance +  numpy.diag(numpy.zeros(len(uv_scales)) + beam_block_covariance[0, 0])
                                                                   # + thermal_variance()
    jacobian_matrix = antenna_baseline_matrix[:, :len(red_tiles)] * sky_based_model



    ideal_covariance = sky_covariance(nu=nu, u = redundant_baselines.u(nu), v = redundant_baselines.v(nu),
                                          S_high=sky_model_depth, mode = 'baseline')

    # pure_FIM, pure_crlb = compute_standard_crlb(jacobian_matrix, non_redundant_covariance, ideal_covariance)
    small_FIM, small_sky_crlb = small_matrix(jacobian_matrix, non_redundant_covariance, ideal_covariance)
    large_FIM, large_sky_crlb = large_matrix(redundant_baselines, jacobian_matrix, non_redundant_covariance)

    print(large_FIM[0, 1])
    print(large_FIM[4, 10])
    return small_FIM, small_sky_crlb, large_FIM, large_sky_crlb#, pure_FIM, pure_crlb


def compute_standard_crlb(jacobian, non_redundant_covariance, ideal_covariance):

    covariance_matrix = restructure_covariance_matrix(ideal_covariance, diagonal= non_redundant_covariance[0, 0],
                                                      off_diagonal=non_redundant_covariance[0, 1])
    fisher_information = numpy.zeros((jacobian.shape[1],jacobian.shape[1]))
    for i in range(jacobian.shape[1]):
        fisher_information[i,i] =  numpy.dot(jacobian[...,i].T, numpy.linalg.solve(covariance_matrix,
                                                                                       jacobian[...,i]))

    print(numpy.dot(jacobian[...,0].T, numpy.linalg.solve(covariance_matrix,jacobian[...,1])))
    print(numpy.dot(jacobian[...,4].T, numpy.linalg.solve(covariance_matrix,jacobian[...,10])))

    cramer_rao_lower_bound = compute_cramer_rao_lower_bound(fisher_information)

    return fisher_information, cramer_rao_lower_bound


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot
    main()

