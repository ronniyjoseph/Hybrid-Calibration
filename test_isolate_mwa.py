import numpy
from matplotlib import pyplot
from matplotlib import colors
from scipy.constants import c

from src.radiotelescope import RadioTelescope
from src.radiotelescope import AntennaPositions
from src.radiotelescope import BaselineTable
from src.util import hexagonal_array
from src.util import redundant_baseline_finder

def plot_mwa_weirdos():
    path  = "data/MWA_Compact_Coordinates.txt"
    telescope = RadioTelescope(load=True, path=path)
    antenna_indices = numpy.array([40, 46, 47, 48, 49, 50, 53, 54, 55])

    pyplot.scatter(telescope.antenna_positions.x_coordinates, telescope.antenna_positions.y_coordinates, s=10)
    pyplot.scatter(telescope.antenna_positions.x_coordinates[antenna_indices],
                   telescope.antenna_positions.y_coordinates[antenna_indices], s=20)
    pyplot.show()
    return


def plot_baseline_numbers(maximum_factor=3):
    size_factor = numpy.arange(7, maximum_factor + 1, 1)

    x = numpy.zeros(len(size_factor))
    y = numpy.zeros(len(size_factor))
    z = numpy.zeros(len(size_factor))

    for i in range(len(size_factor)):
        antenna_positions = hexagonal_array(size_factor[i])
        antenna_table = AntennaPositions(load=False)
        antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
        antenna_table.x_coordinates = antenna_positions[:, 0]
        antenna_table.y_coordinates = antenna_positions[:, 1]
        antenna_table.z_coordinates = antenna_positions[:, 2]
        baseline_table = BaselineTable(position_table=antenna_table)

        redundant_baselines = redundant_baseline_finder(baseline_table)
        skymodel_baselines = redundant_baseline_finder(baseline_table, group_minimum=1)

        x[i] = size_factor[i]
        y[i] = redundant_baselines.number_of_baselines
        z[i] = skymodel_baselines.number_of_baselines

    pyplot.semilogy(x, y)
    pyplot.semilogy(x, z, "--")
    pyplot.show()
    return


if __name__ == "__main__":
    # plot_mwa_weirdos()
    plot_baseline_numbers(16)