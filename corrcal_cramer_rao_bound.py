import sys
import numpy
import os
import time
import multiprocessing
from scipy.constants import c
from scipy import sparse
from functools import partial
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
    parallisation_flag = False
    vectorisation_flag = True
    for i in range(2,3):
        print("")
        print(f"Array size {i}")
        antenna_positions = hexagonal_array(i)
        antenna_table = AntennaPositions(load=False)
        antenna_table.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
        antenna_table.x_coordinates = antenna_positions[:, 0]
        antenna_table.y_coordinates = antenna_positions[:, 1]
        antenna_table.z_coordinates = antenna_positions[:, 2]
        baseline_table = BaselineTable(position_table=antenna_table)

        if verbose:
            print("")
            print("Finding redundant baselines")



        skymodel_baselines = redundant_baseline_finder(baseline_table, group_minimum=3)
        sky_signal = numpy.sqrt(sky_moment_returner(n_order=2))
        modelled_signal = numpy.sqrt(sky_moment_returner(n_order=2, s_low=sky_model_depth))

        thermal_noise = thermal_variance()

        uv_scales = numpy.array([0, 0])
        sky_block_covariance = sky_covariance(nu, u=uv_scales, v=uv_scales, mode='baseline', S_high=sky_model_depth)
        beam_block_covariance = beam_covariance(nu, u=uv_scales, v=uv_scales, broken_tile_fraction=broken_tile_fraction,
                                                   mode='baseline')
        position_block_covariance = position_covariance(nu, u=uv_scales, v=uv_scales, position_precision=position_precision,
                                                       mode='baseline')
        non_redundant_block =  sky_block_covariance #+ beam_block_covariance

        # sky_noise = sky_covariance(nu, u=skymodel_baselines.u(nu), v=skymodel_baselines.v(nu), mode='baseline', S_high=sky_model_depth)
        # beam_error = beam_covariance(nu, u=skymodel_baselines.u(nu), v=skymodel_baselines.v(nu),
        #                              broken_tile_fraction=broken_tile_fraction, mode='baseline')
        # position_error = position_covariance(nu, u=skymodel_baselines.u(nu), v=skymodel_baselines.v(nu),
        #                                      position_precision=position_precision, mode='baseline')
        # ideal_covariance = sky_noise #+ position_error + beam_error


        # sky_model_covariance = sky_noise/sky_noise[0,0]*modelled_signal
        # covariance_matrix = restructure_covariance_matrix(ideal_covariance, diagonal= non_redundant_block[0, 0],
        #                                                   off_diagonal=non_redundant_block[0, 1])
        # data = numpy.zeros(skymodel_baselines.number_of_baselines) + sky_signal
        # noise_covariance = numpy.diag(numpy.zeros(skymodel_baselines.number_of_baselines) + thermal_variance())


        # FIM_loop = compute_fim_sparse(antenna_table, skymodel_baselines, non_redundant_block, sky_signal, modelled_signal, thermal_noise,
        #                         parallelised=parallisation_flag, vectorised =  False)
        FIM_vec = compute_fim_sparse(antenna_table, skymodel_baselines, non_redundant_block, sky_signal,
                                     modelled_signal, thermal_noise, parallelised=parallisation_flag,
                                     vectorised =  vectorisation_flag)


        # fig, axes = pyplot.subplots(1, 3, figsize = (15, 5))
        # axes[0].imshow(FIM_loop)
        # axes[1].imshow(FIM_vec)
        # axes[2].imshow(FIM_loop - FIM_vec)
        # pyplot.show()
        # norm = colors.Normalize()
        # fig,axes = pyplot.subplots(1, 3, figsize = (15, 5))
        # axes[1].imshow(FIM_sparse, norm = norm)
        # axes[0].imshow(FIM_dense, norm = norm)
        # plot_diff = axes[2].imshow(FIM_dense - FIM_sparse, norm = norm)
        #
        # colorbar(plot_diff)
        # pyplot.show()

        # var_dense.append(numpy.median(numpy.diag(numpy.linalg.pinv(FIM_loop))))
        var_sparse.append(numpy.median(numpy.diag(numpy.linalg.pinv(FIM_vec))))

        numpy.savetxt("corrcal_CLRB_vectorised.txt",numpy.array(var_sparse))
    pyplot.semilogy(var_dense)
    pyplot.savefig("test.pdf")
    return


def compute_fim_sparse(antenna_table, baseline_table, covariance_block, total_signal, sky_model, thermal_noise,
                       parallelised = False, vectorised = False):

    n_antennas = len(antenna_table.antenna_ids)
    group_indices = numpy.unique(baseline_table.group_indices)
    n_groups = len(group_indices)
    FIM = numpy.zeros((n_antennas, n_antennas))

    if parallelised:
        k = numpy.arange(0, n_groups)
        pool = multiprocessing.Pool(processes = 4)
        FIM_blocks = pool.map(partial(single_group_fim, covariance_block[0,0], covariance_block[0, 1], total_signal, sky_model,
                                      thermal_noise, antenna_table, baseline_table, group_indices, vectorised), k)

        FIM = numpy.sum(numpy.array(FIM_blocks), axis=0)
    else:
        for k in range(n_groups):
            print(f"group {k}")
            FIM +=single_group_fim(covariance_block[0,0], covariance_block[0, 1], total_signal, sky_model,
                                   thermal_noise, antenna_table, baseline_table, group_indices, vectorised, k)

    return FIM


def compute_fim_dense(antenna_table, baseline_table, R, S, N, D, verbose = False):
    C = R + S
    H = numpy.diag(numpy.zeros(baseline_table.number_of_baselines) + 1)
    HCH = H.T @ C @ H
    N_HCH =  N + HCH
    inv_N_HCH = numpy.linalg.pinv(N_HCH)

    n_antennas = len(antenna_table.antenna_ids)
    FIM = numpy.zeros((n_antennas, n_antennas))

    for i in range(n_antennas):
        di_H = numpy.diag(compute_gain_derivative(baseline_table, antenna_index=antenna_table.antenna_ids[i]))
        di_HCH = di_H.T @ C @ H
        for j in range(i, n_antennas):

            dj_H = numpy.diag(compute_gain_derivative(baseline_table, antenna_index=antenna_table.antenna_ids[j]))
            dj_HCH = dj_H.T @ C @ H

            di_dj_H = numpy.diag(compute_gain_double_derivative(baseline_table, antenna_table.antenna_ids[i],
                                                     antenna_table.antenna_ids[j]))

            FIM_element = 4*D.T @ inv_N_HCH @ dj_HCH @ inv_N_HCH @ di_HCH @ inv_N_HCH @ D +\
                       D.T @ inv_N_HCH @ (di_dj_H.T @ C @ H + di_dj_H.T @ C @ dj_H) @ inv_N_HCH @ D  + \
                           4*D.T @ inv_N_HCH @ di_HCH @ inv_N_HCH @ dj_HCH @ inv_N_HCH @ D

            FIM[i, j] = FIM_element
            FIM[j, i] = FIM_element

    return FIM



def single_group_fim(diagonal, offdiagonal, total_signal, sky_model, thermal_noise, antenna_table, baseline_table,
                     group_indices, vectorised = False, index=0):
    baseline_indices = numpy.where(baseline_table.group_indices == group_indices[index])[0]

    block_size = len(baseline_indices)
    print(f"\t with size {block_size}")
    R = numpy.zeros((block_size, block_size))
    R += offdiagonal
    R -= numpy.diag(numpy.zeros(block_size) + offdiagonal)
    R += numpy.diag(numpy.zeros(block_size) + diagonal)

    S = numpy.zeros((block_size, block_size)) + sky_model

    N = numpy.diag(numpy.zeros(block_size) + thermal_noise)
    D = numpy.zeros(block_size) + total_signal

    group_table = baseline_table.sub_table(baseline_indices)
    if vectorised:
        FIM_block = compute_fim_vectorised(antenna_table, group_table, R, S, N, D)
    else:
        FIM_block = compute_fim_dense(antenna_table, group_table, R, S, N, D)

    return FIM_block


def compute_fim_vectorised(antenna_table, baseline_table, covariance_block, model_block, noise_block, total_signal,
                           verbose = False):
    n_antennas = len(antenna_table.antenna_ids)
    # stacked_di_H = numpy.zeros((baseline_table.number_of_baselines, n_antennas*baseline_table.number_of_baselines))
    # stacked_C = stacked_di_H.copy()
    # stacked_N = stacked_di_H.copy()

    D = numpy.zeros((n_antennas*baseline_table.number_of_baselines, n_antennas))
    di_H = numpy.zeros((n_antennas*baseline_table.number_of_baselines, baseline_table.number_of_baselines))
    C = covariance_block
    N = noise_block
    H_block = numpy.diag(numpy.zeros(baseline_table.number_of_baselines) + 1)
    H = di_H.copy()

    for i in range(n_antennas):

        start = i*baseline_table.number_of_baselines
        end = (i + 1)*baseline_table.number_of_baselines
        di_H[start:end, :] = numpy.diag(compute_gain_derivative(baseline_table,
                                                                        antenna_index=antenna_table.antenna_ids[i]))
        H[start:end, :] = H_block
        D[start:end, i] = total_signal
        # C[start:end, start:end] = covariance_block + model_block
        # N[start:end, start:end] = noise_block
        # inv_N_HCH[start:end, start:end] = invert_block

    inv_N_HCH = H @ numpy.linalg.pinv(noise_block + covariance_block + model_block) @ H.T

    # di_H = numpy.tile(stacked_di_H, (n_antennas, 1))
    # N = numpy.tile(stacked_N, (n_antennas, 1))
    # C = numpy.tile(stacked_C, (n_antennas, 1))
    # HCH = C
    # N_HCH = N + HCH


    # print("Hier")
    # inv_N_HCH = numpy.linalg.pinv(N_HCH)
    # print("Daar")

    dj_H = di_H.T

    di_dj_H = di_H @ dj_H
    t0 = time.clock()

    di_HCH = numpy.dot(di_H, C)
    # if noise_block.shape[0] >= 80:
    #     pyplot.imshow(C)
    #     pyplot.savefig("blaaaaaaaaaaaah.pdf")
    t1 = time.clock()

    dj_HCH = dj_H.T @ C
    t2 = time.clock()

    print(f"\t time {t1-t0}")
    print(f"\t time {t2-t1}")

    print(D.T.shape)
    print(inv_N_HCH.shape)
    print(dj_HCH.shape)
    print(inv_N_HCH.shape)
    print(di_HCH.shape)
    print(inv_N_HCH.shape)
    print(D.shape)

    print(di_dj_H.shape)
    FIM = 4*D.T @ inv_N_HCH @ dj_HCH @ inv_N_HCH @ di_HCH @ inv_N_HCH @ D +\
                       D.T @ inv_N_HCH @ (di_dj_H.T @ C @ H + di_dj_H.T @ C @ dj_H) @ inv_N_HCH @ D  + \
                           4*D.T @ inv_N_HCH @ di_HCH @ inv_N_HCH @ dj_HCH @ inv_N_HCH @ D
    return FIM


def compute_gain_derivative(baseline_table, antenna_index):
    gains = numpy.zeros(baseline_table.number_of_baselines)
    indices = numpy.where(((baseline_table.antenna_id1 == antenna_index) |
                          (baseline_table.antenna_id2 == antenna_index)))[0]
    gains[indices] = 1
    return gains


def compute_gain_double_derivative(baseline_table, antenna_index1, antenna_index2):
    gains = numpy.zeros(baseline_table.number_of_baselines)
    indices_antenna1 = numpy.where((baseline_table.antenna_id1 == antenna_index1) |
                          (baseline_table.antenna_id2 == antenna_index1))[0]
    indices_antenna2 = numpy.where((baseline_table.antenna_id1 == antenna_index2) |
                                   (baseline_table.antenna_id2 == antenna_index2))[0]

    overlapping_indices = numpy.intersect1d(indices_antenna1, indices_antenna2)
    gains[overlapping_indices] = 1

    return gains


if __name__ == "__main__":
    main()