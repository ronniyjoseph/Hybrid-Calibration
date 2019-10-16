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


def cramer_rao_bound_calculator():
    nu = 150e6
    maximum_factor = 9
    size_factor = numpy.arange(2, maximum_factor + 1, 1)
    memory_uptake = numpy.zeros_like(size_factor, dtype = float)
    fig, axes = pyplot.subplots(3, 3, figsize=(15, 15))

    for i in range(len(size_factor)):
        print(f"size of array {size_factor[i]} and index {i}")
        antenna_positions = hexagonal_array(size_factor[i])
        antenna_table = AntennaPositions(load=False)
        antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
        antenna_table.x_coordinates = antenna_positions[:, 0]
        antenna_table.y_coordinates = antenna_positions[:, 1]
        antenna_table.z_coordinates = antenna_positions[:, 2]
        baseline_table = BaselineTable(position_table=antenna_table)

        redundant_baselines = redundant_baseline_finder(baseline_table.antenna_id1, baseline_table.antenna_id2,
                                                        baseline_table.u_coordinates, baseline_table.v_coordinates,
                                                        baseline_table.w_coordinates, group_minimum=1, verbose=False)
        model_covariance = sky_covariance(redundant_baselines[:, 2], redundant_baselines[:, 3], nu)
        memory_uptake[i] = model_covariance.nbytes / 1e9

        axes[int(i / 3), int(i % 3)].imshow(model_covariance, origin='lower')

    axes[-1, -1].semilogy(size_factor, memory_uptake)
    axes[-1, -1].set_xlabel("Array Size Factor")
    axes[-1, -1].set_ylabel("Memory in GB")
    #axes[-1, -1].set_aspect('equal')
    pyplot.show()
    return


def single_array_test():
    antenna_positions = hexagonal_array(4)
    antenna_table = AntennaPositions(load=False)
    antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
    antenna_table.x_coordinates = antenna_positions[:, 0]
    antenna_table.y_coordinates = antenna_positions[:, 1]
    antenna_table.z_coordinates = antenna_positions[:, 2]
    baseline_table = BaselineTable(position_table=antenna_table)

    redundant_baselines = redundant_baseline_finder(baseline_table.antenna_id1, baseline_table.antenna_id2,
                                                    baseline_table.u_coordinates, baseline_table.v_coordinates,
                                                    baseline_table.w_coordinates, group_minimum=3, verbose=False)

    model_covariance = sky_covariance(redundant_baselines[:, 2], redundant_baselines[:, 3], 150e6)

    # pyplot.plot(redundant_baselines[:, 3])
    pyplot.imshow(model_covariance)
    pyplot.show()
    return


if __name__ == "__main__":
    single_array_test()
    # cramer_rao_bound_calculator()
