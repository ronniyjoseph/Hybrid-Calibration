import numpy
from matplotlib import pyplot
from matplotlib import colors
from scipy.constants import c

from cramer_rao_bound import telescope_bounds
from cramer_rao_bound import small_matrix
from cramer_rao_bound import large_matrix
from cramer_rao_bound import sky_model_matrix_populator
from cramer_rao_bound import absolute_calibration_crlb

from src.util import redundant_baseline_finder
from src.plottools import colorbar
from src.radiotelescope import RadioTelescope
from src.skymodel import sky_moment_returner
from src.covariance import sky_covariance


def test_plot():
    path  = "data/MWA_Compact_Coordinates.txt"
    print("")
    print("Redundant Calibration Errors")
    mwa_hexes_antennas, mwa_hexes_redundant = telescope_bounds(path, bound_type="redundant",
                                                               position_precision= 1e-1)
    print("")
    print("Sky Model")
    mwa_hexes_antennas, mwa_hexes_sky = telescope_bounds(path, bound_type="sky")
    pyplot.show()
    # pyplot.semilogy(mwa_hexes_antennas, mwa_hexes_redundant[0], marker="+", label="Relative")
    # pyplot.semilogy(mwa_hexes_antennas, mwa_hexes_redundant[1], marker="x", label="Absolute")
    # pyplot.semilogy(mwa_hexes_antennas, mwa_hexes_sky, marker="*", label="Sky")
    # pyplot.legend()
    # pyplot.show()

    return


def test_fim_approximation(nu = 150e6, position_precision = 1e-2):
    path  = "data/MWA_Compact_Coordinates.txt"
    telescope = RadioTelescope(load=True, path=path)
    sky_based_model = numpy.sqrt(sky_moment_returner(n_order=2, S_low=1, S_high=10))

    redundant_baselines = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)
    # pyplot.plot(redundant_baselines.antenna_id1, redundant_baselines.group_indices, "ro")
    # pyplot.plot(redundant_baselines.antenna_id2, redundant_baselines.group_indices, "ro")
    # pyplot.show()
    antenna_baseline_matrix, red_tiles = sky_model_matrix_populator(redundant_baselines)
    jacobian_matrix = antenna_baseline_matrix[:, :len(red_tiles)] * sky_based_model
    uv_scales = numpy.array([0, position_precision / c * nu])

    non_redundant_covariance = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, mode='baseline')
    ideal_covariance = sky_covariance(nu, u=redundant_baselines.u(nu), v=redundant_baselines.v(nu),
                                      mode='baseline')
    dense_crlb, dense_fim = small_matrix(jacobian_matrix, non_redundant_covariance, ideal_covariance)
    sparse_crlb, sparse_fim = large_matrix(redundant_baselines,jacobian_matrix, non_redundant_covariance)

    # numpy.save("dense", dense_crlb)
    numpy.save("sparse", sparse_crlb)
    fig, axes = pyplot.subplots(2, 3, figsize = (15, 10))

    norm = colors.Normalize()
    denseplot = axes[0, 0].imshow(dense_crlb, norm = norm, interpolation = 'none')
    sparseplot = axes[0, 1].imshow(sparse_crlb, norm = norm, interpolation = 'none')
    norm = colors.Normalize()

    diffplot = axes[0, 2].imshow(sparse_crlb - dense_crlb, norm = norm, interpolation = 'none')

    colorbar(denseplot)
    colorbar(sparseplot)
    colorbar(diffplot)

    norm = colors.Normalize()

    dense_fimplot = axes[1, 0].imshow(dense_fim, norm = norm, interpolation = 'none')
    sparse_fimplot = axes[1, 1].imshow(sparse_fim, norm = norm, interpolation = 'none')
    norm = colors.Normalize()

    diff_fimplot = axes[1, 2].imshow(sparse_fim - dense_fim, norm = norm, interpolation = 'none')

    colorbar(dense_fimplot)
    colorbar(sparse_fimplot)
    colorbar(diff_fimplot)

    axes[0, 0].set_title("Dense CRLB")
    axes[0, 1].set_title("Sparse CRLB")
    axes[0, 2].set_title("Sparse - Dense")

    axes[1, 0].set_title("Dense FIM")
    axes[1, 1].set_title("Sparse FIM")
    axes[1, 2].set_title("Sparse - Dense")

    axes[0, 0].set_ylabel("Antenna Index")
    axes[1, 0].set_ylabel("Antenna Index")

    axes[1, 0].set_xlabel("Antenna Index")
    axes[1, 1].set_xlabel("Antenna Index")
    axes[1, 2] .set_xlabel("Antenna Index")
    pyplot.show()
    return


def test_absolute_calibration():
    path = "data/MWA_Hexes_Coordinates.txt"
    telescope = RadioTelescope(load=True, path=path)
    baseline_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)
    absolute_calibration_crlb(baseline_table)

    position_precision = 1e-2
    nu = 150e6

    sky_based_model = numpy.sqrt(sky_moment_returner(n_order=2, S_low=1, S_high=10))
    jacobian_vector = numpy.zeros(baseline_table.number_of_baselines) + sky_based_model
    uv_scales = numpy.array([0, position_precision / c * nu])
    non_redundant_block = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, mode='baseline')
    ideal_covariance = sky_covariance(nu=nu, u=baseline_table.u_coordinates,
                                      v=baseline_table.v_coordinates, mode='baseline')

    small_crlb = small_matrix(jacobian_vector, non_redundant_block, ideal_covariance)
    large_crlb = large_matrix(baseline_table, jacobian_vector, non_redundant_block)
    print(f"small crlb {small_crlb}")
    print(f"large crlb {large_crlb}")
    return


def test_compare_old_new_results():
    new_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/general_hex/"
    # new_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/new_general_hex_including_covariances/"

    old_red_crlb = numpy.loadtxt("TESTING_redundant_crlb.txt")
    old_sky_crlb = numpy.loadtxt("TESTING_skymodel_crlb.txt")

    new_red_crlb = numpy.loadtxt(new_path + "redundant_crlb.txt")
    new_sky_crlb = numpy.loadtxt(new_path + "skymodel_crlb.txt")

    fig, axes = pyplot.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(old_red_crlb[0, :], numpy.sqrt(old_red_crlb[1, :] + old_red_crlb[2, :]), color = "C0", label="Total")
    axes[0].plot(old_red_crlb[0, :], numpy.sqrt(old_red_crlb[1, :]), color = "C1", label="Relative")
    axes[0].plot(old_red_crlb[0, :], numpy.sqrt(old_red_crlb[2, :]), color = "C2", label="Absolute")
    axes[0].plot(old_red_crlb[0, :], numpy.sqrt(old_red_crlb[3, :]), color ="k", label="Thermal")

    axes[0].plot(new_red_crlb[0, :], numpy.sqrt(new_red_crlb[1, :] + old_red_crlb[2, :]), color = "C0", linestyle= "--")
    axes[0].plot(new_red_crlb[0, :], numpy.sqrt(new_red_crlb[1, :]), color = "C1", linestyle = "--",label="Relative")
    axes[0].plot(new_red_crlb[0, :], numpy.sqrt(new_red_crlb[2, :]), color = "C2",linestyle = "--", label="Absolute")
    axes[0].plot(new_red_crlb[0, :], numpy.sqrt(new_red_crlb[3, :]), color ="k",linestyle = "--", label="Thermal")

    axes[0].set_ylabel("Gain Variance")
    axes[0].set_yscale('log')

    axes[1].semilogy(old_sky_crlb[0, :], numpy.sqrt(old_sky_crlb[1, :]), color = "C0", label="Sky Calibration")
    axes[1].semilogy(old_sky_crlb[0, :], numpy.sqrt(old_sky_crlb[2, :]), color ="k", label="Thermal")


    axes[1].semilogy(new_sky_crlb[0, :], numpy.sqrt(new_sky_crlb[1, :]), color = "C0", linestyle= "--")
    axes[1].semilogy(new_sky_crlb[0, :], numpy.sqrt(new_sky_crlb[2, :]), color ="k", linestyle= "--")

    axes[0].set_xlabel("Number of Antennas")
    axes[1].set_xlabel("Number of Antennas")

    axes[0].set_ylim([1e-4, 1])
    axes[1].set_ylim([1e-4, 1])

    axes[0].legend()
    axes[1].legend()
    pyplot.show()
    return

if __name__ == "__main__":
    test_plot()
    # test_fim_approximation()
    # test_absolute_calibration()