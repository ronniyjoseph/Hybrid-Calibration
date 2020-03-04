import argparse
import numpy
import sys
from src.radiotelescope import AntennaPositions
from src.radiotelescope import BaselineTable
from src.plottools import colorbar
sys.path.append("../../CorrCal_UKZN_Development/corrcal")
from corrcal import grid_data


def main():
    data_type = "noise_visibilities"
    path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/Square_Large_Array_100Jy_Noise_No_Models/"
    input_parameters = numpy.loadtxt(path + "input_parameters.txt")
    n_realisations = 1000 #input_parameters[-1]
    ideal_visibilities = load_data(path, 'ideal_visibilities', n_realisations)
    noise_visibilities = load_data(path, 'noise_visibilities', n_realisations)
    measured_visibilities = load_data(path, "measured_visibilities", n_realisations)

    telescope_positions = numpy.load(path+"realisation_0/telescope_positions.npy")
    telescope_gains = numpy.load(path+"realisation_0/telescope_gains.npy")
    antenna_table = AntennaPositions(load=False)

    antenna_table.antenna_ids = telescope_positions[0, :]
    antenna_table.x_coordinates = telescope_positions[1, :]
    antenna_table.y_coordinates = telescope_positions[2, :]
    antenna_table.z_coordinates = telescope_positions[3, :]
    antenna_table.antenna_gains = telescope_gains

    baseline_table = BaselineTable(position_table=antenna_table, frequency_channels=numpy.array([150e6]))
    data_sorted, u_sorted, v_sorted, noise_sorted, ant1_sorted, ant2_sorted, edges_sorted, sorting_indices, \
    conjugation_flag = grid_data(ideal_visibilities[:, 0],
                                 baseline_table.u_coordinates,
                                 baseline_table.v_coordinates,
                                 noise_visibilities,
                                 baseline_table.antenna_id1.astype(int),
                                 baseline_table.antenna_id2.astype(int))

    sky_covariance = numpy.cov(ideal_visibilities[sorting_indices, :])
    noise_covariance = numpy.cov(noise_visibilities[sorting_indices, :])
    data_covariance = numpy.cov(measured_visibilities[sorting_indices, :])

    figure, axes = pyplot.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(numpy.real(sky_covariance), interpolation=None)
    axes[0, 1].imshow(numpy.real(noise_covariance), interpolation=None)
    axes[0, 2].imshow(numpy.real(data_covariance), interpolation=None)

    axes[1, 0].imshow(numpy.imag(sky_covariance), interpolation=None)
    axes[1, 1].imshow(numpy.imag(noise_covariance), interpolation=None)
    axes[1, 2].imshow(numpy.imag(data_covariance), interpolation=None)

    #colorbar(axes[0, 0])
    # colorbar(axes[0, 1])
    # colorbar(axes[0, 2])
    # colorbar(axes[1, 0])
    # colorbar(axes[1, 1])
    # colorbar(axes[1, 2])

    figure.savefig(path + "visibilities.pdf")
    pyplot.show()
    return


def load_data(path, data_type, n_realisations):
    for i in range(n_realisations):
        folder = f"realisation_{i}/"
        file = data_type + ".npy"
        if i == 0:
            data = numpy.load(path + folder + file)
        elif i == -1:
            new_realisation = numpy.load(path + folder + file)
            data = numpy.append(data, new_realisation, axis=-1)
        else:
            new_realisation = numpy.load(path + folder + file)
            data = numpy.append(data, new_realisation, axis=1)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()