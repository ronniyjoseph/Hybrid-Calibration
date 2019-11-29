import sys
import numpy
import os
from scipy.constants import c
from scipy import sparse
import matplotlib
from matplotlib import colors
#matplotlib.use("Agg")
from matplotlib import pyplot

from src.plottools import colorbar
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

from cramer_rao_bound import redundant_matrix_populator
from cramer_rao_bound import restructure_covariance_matrix
from cramer_rao_bound import compute_fisher_information

def main():
    nu=150e6
    position_precision = 0
    broken_tile_fraction = 0

    antenna_positions = hexagonal_array(4)
    antenna_table = AntennaPositions(load=False)
    antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
    antenna_table.x_coordinates = antenna_positions[:, 0]
    antenna_table.y_coordinates = antenna_positions[:, 1]
    antenna_table.z_coordinates = antenna_positions[:, 2]
    baseline_table = BaselineTable(position_table=antenna_table)
    redundant_baselines = redundant_baseline_finder(baseline_table)

    redundant_sky = numpy.sqrt(sky_moment_returner(n_order=2))
    jacobian_gain_matrix, red_tiles, red_groups = redundant_matrix_populator(redundant_baselines)
    jacobian_gain_matrix[:, :len(red_tiles)] *= redundant_sky

    constraints_matrix = numpy.zeros((len(red_tiles) + len(red_groups), 4))
    constraints_matrix[1:, 0] = -1
    constraints_matrix[len(red_tiles) +1:, 1] = -1
    constraints_matrix[:len(red_tiles) + 1, 2] = antenna_table.x_coordinates
    constraints_matrix[:len(red_tiles) + 1, 3] = antenna_table.y_coordinates
    pyplot.imshow(constraints_matrix.T)
    pyplot.show()
    sky_noise = sky_covariance(nu, u=redundant_baselines.u(nu), v=redundant_baselines.v(nu), mode='baseline')
    beam_error = beam_covariance(nu, u=redundant_baselines.u(nu), v=redundant_baselines.v(nu),
                                     broken_tile_fraction=broken_tile_fraction, mode='baseline')
    position_error = position_covariance(nu, u=redundant_baselines.u(nu), v=redundant_baselines.v(nu),
                                             position_precision=position_precision, mode='baseline')

    thermal_noise = numpy.diag(numpy.zeros(redundant_baselines.number_of_baselines) + thermal_variance())


    idealistic_covariance = sky_noise + numpy.diag(numpy.zeros(redundant_baselines.number_of_baselines) + position_error[0, 0]) + \
                       numpy.diag(numpy.zeros(redundant_baselines.number_of_baselines) + beam_error[0,0]) + thermal_noise



    FIM_old = compute_fisher_information(idealistic_covariance, jacobian_gain_matrix)
    print(f"number of redundant_groups = {len(red_groups)}")
    print(sky_noise[0,0])
    print(numpy.sqrt(sky_noise[0,0]))
    print(numpy.sqrt(sky_noise[0,0])/sky_noise[0,0])
    FIM_new = compute_fisher_information(idealistic_covariance, jacobian_gain_matrix, covariance_jacobian=2*numpy.sqrt(sky_noise),
                                         antennas_indices=red_tiles, redundant_groups=red_groups,
                                         redundant_baselines=redundant_baselines)

    pyplot.imshow(numpy.linalg.pinv(FIM_new))
    pyplot.show()
    print(FIM_new.shape)
    print(constraints_matrix.shape)
    # fisher_information = constraints_matrix @ numpy.linalg.pinv(constraints_matrix.T @ FIM_new @ constraints_matrix) @ constraints_matrix.T
    print(f"Old FIM condition number = {numpy.linalg.cond(FIM_new)}")
    print(f"New FIM condition number = {numpy.linalg.cond(fisher_information)}")

    fig, axes = pyplot.subplots(1, 3, figsize = (15, 5))
    old = axes[0].imshow(FIM_new)
    new = axes[1].imshow(fisher_information)
    diff= axes[2].imshow(FIM_new - fisher_information)
    colorbar(old)
    colorbar(new)
    colorbar(diff)
    pyplot.show()

    return

def ska_test():
    file = "data/SKA_Low_v5_ENU_fullcore.txt"
    telescope = RadioTelescope(load=True, path=file)
    print("Finding Redundant Baselines")
    redundant_table = redundant_baseline_finder(telescope.baseline_table)
    print(redundant_table.number_of_baselines)
    return

if __name__ == "__main__":
    # /main()
    ska_test()