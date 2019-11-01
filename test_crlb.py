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
    # mwa_hexes_antennas, mwa_hexes_redundant = telescope_bounds(path, bound_type="redundant",
    #                                                            position_precision= 1e-1)
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
    path  = "data/MWA_Hexes_Coordinates.txt"
    telescope = RadioTelescope(load=True, path=path)
    sky_based_model = numpy.sqrt(sky_moment_returner(n_order=2, S_low=1, S_high=10))

    redundant_baselines = redundant_baseline_finder(telescope.baseline_table, group_minimum= 3)
    antenna_baseline_matrix, red_tiles = sky_model_matrix_populator(redundant_baselines)
    jacobian_matrix = antenna_baseline_matrix[:, :len(red_tiles)] * sky_based_model
    uv_scales = numpy.array([0, position_precision / c * nu])

    non_redundant_covariance = sky_covariance(nu=nu, u=uv_scales, v=uv_scales, mode='baseline')
    ideal_covariance = sky_covariance(nu, u=redundant_baselines.u(nu), v=redundant_baselines.v(nu),
                                      mode='baseline')
    dense_crlb, dense_fim = small_matrix(jacobian_matrix, non_redundant_covariance, ideal_covariance)
    sparse_crlb, sparse_fim = large_matrix(redundant_baselines,jacobian_matrix, non_redundant_covariance)

    numpy.save("dense", dense_crlb)
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
    return

if __name__ == "__main__":
    # test_plot()
    test_fim_approximation()