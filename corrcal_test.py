from __future__ import print_function
import sys
import numpy
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot
from scipy.optimize import fmin_cg


# Import local codes
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
sys.path.append("../corrcal2")

import skymodel
import radiotelescope
from gain_variance_simulation import get_observations_numba
import corrcal2
from corrcal2 import sparse_2level
from corrcal2 import grid_data
from corrcal2 import sparse_2level

from analytic_covariance import sky_covariance


################################################
# ant1.dat contains the antenna id/index for the first antenna in each baseline
# ant2.dat contains the antenna id/index for the second antenna in each baseline
# gtmp.dat contains the antenna gains split up in real and imaginary components
# vis.dat contains the visibility data split up in real and imaginary components

# signal_sparse2_test.dat contains a lot of information that apparently needs to be shaped in the right way
#
################################################

# Antenna indices have to integers and index to location in position table



def main(path, tol=0.1):
    calibrator_flux = 100
    calibrator_l = 0
    calibrator_m = 0
    frequency_range = numpy.array([150]) * 1e6
    noise_level = 1e-5


    path = "../../beam_perturbations/code/tile_beam_perturbations/Data/" + path
    # radio_telescope = radiotelescope.RadioTelescope(load = False  )
    # radio_telescope = radiotelescope.RadioTelescope(load=False, shape=['doublehex', 7, 0, 0, 20, 20])
    radio_telescope = radiotelescope.RadioTelescope(load=False, shape=['square', 100, 12, 0, 0])
    radio_telescope.antenna_positions.antenna_ids = numpy.arange(0, len(radio_telescope.antenna_positions.antenna_ids), 1)
    radio_telescope.baseline_table = radiotelescope.BaselineTable(radio_telescope.antenna_positions)

    sky_realisation = skymodel.SkyRealisation(sky_type="random", flux_high=1)

    sky_realisation.fluxes = numpy.append(sky_realisation.fluxes, calibrator_flux)
    sky_realisation.l_coordinates = numpy.append(sky_realisation.l_coordinates, calibrator_l)
    sky_realisation.m_coordinates = numpy.append(sky_realisation.m_coordinates, calibrator_m)
    visibility_data = get_observations_numba(sky_realisation, radio_telescope.baseline_table, frequency_range)
    thermal_noise = numpy.random.normal(scale = noise_level, size = visibility_data.shape)
    # Reorders all the data into redundant groupings
    data, u, v, noise, ant1, ant2, edges, ii, isonj = grid_data(visibility_data + thermal_noise,
                                                                radio_telescope.baseline_table.u_coordinates,
                                                                radio_telescope.baseline_table.v_coordinates,
                                                                thermal_noise,
                                                                radio_telescope.baseline_table.antenna_id1.astype(int),
                                                                radio_telescope.baseline_table.antenna_id2.astype(int),
                                                                tol=tol)

    # We need to split up all data into real an imaginary parts and then rearrange them alternating Re1, Im1, Re2, Im2
    data_split = split_visibility(data)

    # We need to create a model visibility vector and do the same thing
    # Create a sky model object, NOTE: for multiple sources we need to split this up per source.
    sky_model = skymodel.SkyRealisation(sky_type="point", fluxes=calibrator_flux, l_coordinates=calibrator_l,
                                        m_coordinates=calibrator_m)
    visibility_model = get_observations_numba(sky_model, radio_telescope.baseline_table, frequency_range)
    model_vectors = split_visibility((visibility_model[ii])).reshape([1, len(data_split) ])
    #
    # print(data.shape)
    # print(visibility_model[ii].shape)
    # pyplot.plot(data_split[::2]**2+data_split[1::2]**2)
    # pyplot.show()

    # We need to create a data covariance vector that describes the correlation between data in redundant blocks.
    # Currently the code seems to assume that data in redundant blocks are perfectly redundant, i.e. the covariance has
    # the same value throughout the block. We know however, that due to position errors this covariance changes
    # within a "Redundant/Quasi Redundant" block. So at some point we need to understand how to include the covariance
    # for quasi redundant baselines
    # This opens the rabbit hole of computing eigenvectors etc for the construction of the magic covariance matrix.

    # Okay there is some horrible reshaping going on, need to ensure that all reshaping is done properly
    covariance_vectors = numpy.zeros((2, data_split.shape[0]))
    covariance_vectors[0::2] = 1
    covariance_vectors[1::2] = 1
    # set the level of variance
    covariance_vectors *= sky_covariance(0, 0, frequency_range)
    # Create a noise variance vector, that describes the diagonal
    # This noise can't be to small because other it becomes massive in the inverse diagonal
    noise_variance = numpy.zeros(data_split.shape[0]) + noise_level
    sparse_matrix_object = sparse_2level(noise_variance, covariance_vectors, model_vectors, edges)

    fac = 1000.0
    n_antennas = len(radio_telescope.antenna_positions.antenna_ids)
    gain_guess = numpy.zeros(2*n_antennas)
    gain_guess[::2] = 1


    corrcal2.get_chisq(gain_guess*fac, data_split, sparse_matrix_object, ant1, ant2, scale_fac = fac)
    corrcal2.get_gradient(gain_guess*fac,  data_split, sparse_matrix_object, ant1, ant2, fac)
    gain_solutions = fmin_cg(corrcal2.get_chisq, gain_guess * fac, corrcal2.get_gradient, (data_split, sparse_matrix_object, ant1, ant2, fac))
    pyplot.plot(numpy.sqrt(gain_solutions[1::2] ** 2 + gain_solutions[::2] ** 2))
    pyplot.show()
    print("HURAAAAAAH I DID NOT CRASH")
    return


def split_visibility(data):
    data_real = numpy.real(data)
    data_imag = numpy.imag(data)

    data_split = numpy.hstack((data_real, data_imag)).reshape((1, 2 * len(data_real)), order="C")
    return data_split[0,:]


def plot_weights():
    mwa_edges = main("MWA_Compact_Coordinates.txt", tol=1)
    ska_edges = main("SKA_Low_v5_ENU_fullcore.txt", tol=5)

    mwa_group_fraction = numpy.linspace(0, 100, len(mwa_edges))
    ska_group_fraction = numpy.linspace(0, 100, len(ska_edges))
    linear_group = numpy.linspace(0, 100, 100)

    pyplot.plot(linear_group, linear_group, "--")
    pyplot.plot(mwa_group_fraction, mwa_edges / numpy.max(mwa_edges) * 100, label="MWA")
    pyplot.plot(ska_group_fraction, ska_edges / numpy.max(ska_edges) * 100, label="SKA")
    pyplot.xlabel("Percentage of Redundant Groups")
    pyplot.ylabel("Percentage of Baselines")
    pyplot.legend()

    pyplot.savefig("Redundancy_Comparison.png")

    return


if "__main__" == __name__:
    main("MWA_Hexes_Coordinates.txt")
