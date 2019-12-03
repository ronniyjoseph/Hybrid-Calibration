import sys
import numpy
import os
import multiprocessing
from scipy.constants import c
from scipy import sparse
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot
from matplotlib import colors

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
from cramer_rao_bound import restructure_covariance_matrix

def main(nu =150e6, position_precision = 1e-2, broken_tile_fraction=0.3, sky_model_depth = 1e-2, verbose = False):
    var_dense = []
    var_sparse = []

    for i in range(2, 16):
        print("")
        print(f"Array size {i}")
        antenna_positions = hexagonal_array(i)
        antenna_table = AntennaPositions(load=False)
        antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
        antenna_table.x_coordinates = antenna_positions[:, 0]
        antenna_table.y_coordinates = antenna_positions[:, 1]
        antenna_table.z_coordinates = antenna_positions[:, 2]
        baseline_table = BaselineTable(position_table=antenna_table)

        print(antenna_table.antenna_ids)
        if verbose:
            print("")
            print("Finding redundant baselines")



        skymodel_baselines = redundant_baseline_finder(baseline_table, group_minimum=1)
        sky_signal = numpy.sqrt(sky_moment_returner(n_order=2))
        modelled_signal = numpy.sqrt(sky_moment_returner(n_order=2, s_low=sky_model_depth))

        thermal_noise = thermal_variance()

        uv_scales = numpy.array([0, position_precision/c*nu*150])
        sky_block_covariance = sky_covariance(nu, u=numpy.array([0,0]), v=numpy.array([0,0]), mode='baseline')
        beam_block_covariance = beam_covariance(nu, u=uv_scales, v=uv_scales, broken_tile_fraction=broken_tile_fraction,
                                                   mode='baseline')
        position_block_covariance = position_covariance(nu, u=uv_scales, v=uv_scales, position_precision=position_precision,
                                                       mode='baseline')
        non_redundant_block =  sky_block_covariance +  numpy.diag(numpy.zeros(len(uv_scales)) + beam_block_covariance[0,0]) + \
                               numpy.diag(numpy.zeros(len(uv_scales)) + position_block_covariance[0, 0])

        # sky_noise = sky_covariance(nu, u=skymodel_baselines.u(nu), v=skymodel_baselines.v(nu), mode='baseline')
        # beam_error = beam_covariance(nu, u=skymodel_baselines.u(nu), v=skymodel_baselines.v(nu),
        #                              broken_tile_fraction=broken_tile_fraction, mode='baseline')
        # position_error = position_covariance(nu, u=skymodel_baselines.u(nu), v=skymodel_baselines.v(nu),
        #                                      position_precision=position_precision, mode='baseline')
        # ideal_covariance = sky_noise + position_error + beam_error


        # sky_model_covariance = sky_noise/sky_noise[0,0]*modelled_signal
        # covariance_matrix = restructure_covariance_matrix(ideal_covariance, diagonal= non_redundant_block[0, 0],
        #                                                   off_diagonal=non_redundant_block[0, 1])
        # data = numpy.zeros(skymodel_baselines.number_of_baselines) + sky_signal
        # noise_covariance = numpy.diag(numpy.zeros(skymodel_baselines.number_of_baselines) + thermal_variance())


        # FIM_dense = compute_fim_dense(antenna_table, skymodel_baselines, covariance_matrix, sky_model_covariance, noise_covariance, data)
        FIM_sparse = compute_fim_sparse(antenna_table, skymodel_baselines, non_redundant_block, modelled_signal, thermal_noise,
                                sky_signal)

        # norm = colors.Normalize()
        # fig,axes = pyplot.subplots(1, 3, figsize = (15, 5))
        # axes[1].imshow(FIM_sparse, norm = norm)
        # axes[0].imshow(FIM_dense, norm = norm)
        # plot_diff = axes[2].imshow(FIM_dense - FIM_sparse, norm = norm)
        #
        # colorbar(plot_diff)
        # pyplot.show()

        # var_dense.append(numpy.median(numpy.diag(numpy.linalg.pinv(FIM_dense))))
        var_sparse.append(numpy.median(numpy.diag(numpy.linalg.pinv(FIM_sparse))))
        print(var_sparse)
        numpy.savetxt("test_corrcal_variance.txt",numpy.array(var_sparse))
    pyplot.semilogy(var_dense)
    pyplot.savefig("test.pdf")
    return


def compute_fim_dense(antenna_table, baseline_table, R, S, N, D, verbose = False):
    C = R + S
    H = numpy.diag(numpy.zeros(baseline_table.number_of_baselines) + 1)
    HCH = H.T @ C @ H
    N_HCH =  N + HCH
    inv_N_HCH = numpy.linalg.pinv(N_HCH)

    n_antennas = len(antenna_table.antenna_ids)
    FIM = numpy.zeros((n_antennas, n_antennas))


    for i in range(n_antennas):
        di_H = compute_gain_derivative(baseline_table, antenna_index=antenna_table.antenna_ids[i])
        di_HCH = di_H.T @ C @ H
        for j in range(i, n_antennas):

            dj_H = compute_gain_derivative(baseline_table, antenna_index=antenna_table.antenna_ids[j])
            dj_HCH = dj_H.T @ C @ H

            di_dj_H = compute_gain_double_derivative(baseline_table, antenna_table.antenna_ids[i],
                                                     antenna_table.antenna_ids[j])

            FIM_element = 4*D.T @ inv_N_HCH @ dj_HCH @ inv_N_HCH @ di_HCH @ inv_N_HCH @ D +\
                       D.T @ inv_N_HCH @ (di_dj_H.T @ C @ H + di_dj_H.T @ C @ dj_H) @ inv_N_HCH @ D  + \
                           4*D.T @ inv_N_HCH @ di_HCH @ inv_N_HCH @ dj_HCH @ inv_N_HCH @ D

            FIM[i, j] =FIM_element
            FIM[j, i] = FIM_element
            # cols = 4
            # fig, axes = pyplot.subplots(1, cols, figsize = (int(cols*4), 5))
            # axes[0].imshow(dj_H.T)
            # axes[1].imshow(C)
            # axes[2].imshow(dj_H.T @ C)
            # axes[3].imshow(dj_HCH)
            # pyplot.show()

            # FIM[i, j] = 4 * numpy.linalg.solve(N_HCH.T, D ).T @ numpy.linalg.solve(dj_HCH @ inv_N_HCH) @ di_HCH @ inv_N_HCH @ D + \
            #     D.T @ inv_N_HCH @ (di_dj_H.T @ C @ H + di_dj_H.T @ C @ dj_H) @ inv_N_HCH @ D + \
            #     4 * D.T @ inv_N_HCH @ di_HCH @ inv_N_HCH @ dj_HCH @ inv_N_HCH @ D
    return FIM


def compute_fim_sparse(antenna_table, baseline_table, covariance_block, sky_model, thermal_noise, data):

    n_antennas = len(antenna_table.antenna_ids)
    group_indices = numpy.unique(baseline_table.group_indices)
    n_groups = len(group_indices)
    print(f"Number of groups {n_groups}")
    FIM = numpy.zeros((n_antennas, n_antennas))

    for k in range(n_groups):
        verbose = False
        baseline_indices = numpy.where(baseline_table.group_indices == group_indices[k])[0]

        block_size = len(baseline_indices)

        R = numpy.zeros((block_size, block_size))
        R += covariance_block[0, 1]
        R -= numpy.diag(numpy.zeros(block_size) + covariance_block[0, 1])
        R += numpy.diag(numpy.zeros(block_size) + covariance_block[0, 0])

        S = numpy.zeros((block_size, block_size)) + sky_model

        N = numpy.diag(numpy.zeros(block_size) + thermal_noise)
        D = numpy.zeros(block_size) + data

        group_table = baseline_table.sub_table(baseline_indices)

        FIM += compute_fim_dense(antenna_table, group_table, R, S, N, D, verbose)

    return FIM


def compute_gain_derivative(baseline_table, antenna_index):
    gains = numpy.zeros(baseline_table.number_of_baselines)
    indices = numpy.where(((baseline_table.antenna_id1 == antenna_index) |
                          (baseline_table.antenna_id2 == antenna_index)))[0]
    gains[indices] = 1
    gain_1st_derivative = numpy.diag(gains)

    return gain_1st_derivative


def compute_gain_double_derivative(baseline_table, antenna_index1, antenna_index2):
    gains = numpy.zeros(baseline_table.number_of_baselines)
    indices_antenna1 = numpy.where((baseline_table.antenna_id1 == antenna_index1) |
                          (baseline_table.antenna_id2 == antenna_index1))[0]
    indices_antenna2 = numpy.where((baseline_table.antenna_id1 == antenna_index2) |
                                   (baseline_table.antenna_id2 == antenna_index2))[0]

    overlapping_indices = numpy.intersect1d(indices_antenna1, indices_antenna2)
    gains[overlapping_indices] = 1
    gain_2nd_derivative = numpy.diag(gains)

    return gain_2nd_derivative


if __name__ == "__main__":
    main()