import numpy
import sys
from src.radiotelescope import RadioTelescope
from src.radiotelescope import AntennaPositions
from src.radiotelescope import BaselineTable
from src.util import redundant_baseline_finder
from src.util import hexagonal_array
from src.covariance import position_covariance
from src.covariance import beam_covariance
from src.covariance import sky_covariance
from src.covariance import blackman_harris_taper
from src.covariance import dft_matrix
from src.covariance import compute_ps_variance
from matplotlib import pyplot
from matplotlib import colors

from src.plottools import plot_power_spectrum


def calculate_beam_power_spectrum(u, nu, save=False, plot_name="beam_2D_ps.pdf"):
    diameter = 4
    window_function = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)
    dftmatrix, eta = dft_matrix(nu)

    variance = numpy.zeros((len(u), len(nu)))

    print(f"Calculating covariances for all baselines")
    for i in range(len(u)):
        nu_cov = position_covariance(u=u[i], v=0, nu=nu, position_precision=1e-2, tile_diameter=diameter)
        variance[i, :] = compute_ps_variance(taper1, taper2, nu_cov, dftmatrix)

    figure, axes = pyplot.subplots(1, 1, figsize =(5, 5))
    ps_norm = colors.LogNorm()

    plot_power_spectrum(u, eta[:int(len(eta) / 2)], nu, variance[:, :int(len(eta) / 2)], axes=axes, norm = ps_norm,
                        colorbar_show =True, xlabel_show = True)
    pyplot.show()
    return


def compare_power_spectrum():
    diameter = 4
    u = numpy.logspace(-3, numpy.log10(5000), 100)
    nu = numpy.linspace(140, 160, 300) * 1e6

    window_function = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)
    dftmatrix, eta = dft_matrix(nu)

    position_variance = numpy.zeros((len(u), len(nu)))
    sky_variance = numpy.zeros((len(u), len(nu)))
    print(f"Calculating covariances for all baselines")
    for i in range(len(u)):
        position_cov = position_covariance(u=u[i], v=0, nu=nu, position_precision=1e-2, tile_diameter=diameter)
        sky_cov = sky_covariance(u=u[i], v=0, nu=nu)
        position_variance[i, :] = compute_ps_variance(taper1, taper2, position_cov, dftmatrix)
        sky_variance[i, :] = compute_ps_variance(taper1, taper2, sky_cov, dftmatrix)

    x_range = [1e-3, 1e0]
    y_range = [1e-3, 1e0]

    figure, axes = pyplot.subplots(1, 3, figsize =(15, 5))
    ps_norm = colors.LogNorm()
    plot_power_spectrum(u, eta[:int(len(eta) / 2)], nu, position_variance[:, :int(len(eta) / 2)], axes=axes[0],
                        norm = ps_norm, colorbar_show =True, xlabel_show = True, zlabel_show= False, ylabel_show=True,
                        x_range=x_range, y_range=y_range)
    plot_power_spectrum(u, eta[:int(len(eta) / 2)], nu, sky_variance[:, :int(len(eta) / 2)], axes=axes[1],
                        norm = ps_norm, colorbar_show =True, xlabel_show = True, zlabel_show= False, x_range=x_range,
                        y_range=y_range)

    diff_norm = colors.SymLogNorm(linthresh= 1e2, linscale = 1.5, vmin = -1e6, vmax = 1e6)
    plot_power_spectrum(u, eta[:int(len(eta) / 2)], nu, sky_variance[:, :int(len(eta) / 2)] -
                        position_variance[:, :int(len(eta) / 2)] , axes=axes[2], norm=diff_norm, colorbar_show=True,
                        xlabel_show=True, diff = True, zlabel_show=True, x_range=x_range, y_range=y_range)

    pyplot.show()
    return


def test_baseline_covariance(nu = 150e6):
    # telescope = RadioTelescope(load = True, path="data/MWA_Hexes_Coordinates.txt")
    telescope = RadioTelescope(load=False)
    antenna_positions = hexagonal_array(4)
    telescope.antenna_positions = AntennaPositions(load=False)
    telescope.antenna_positions.antenna_ids = numpy.arange(0, antenna_positions.shape[0], 1)
    telescope.antenna_positions.x_coordinates = antenna_positions[:, 0]
    telescope.antenna_positions.y_coordinates = antenna_positions[:, 1]
    telescope.antenna_positions.z_coordinates = antenna_positions[:, 2]
    telescope.baseline_table = BaselineTable(position_table=telescope.antenna_positions)



    # telescope.antenna_positions.x_coordinates += numpy.random.normal(0, 1e-1, telescope.antenna_positions.number_antennas())
    # telescope.antenna_positions.y_coordinates += numpy.random.normal(0, 1e-1, telescope.antenna_positions.number_antennas())
    telescope.baseline_table = BaselineTable(position_table=telescope.antenna_positions)

    redundant_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=3)
    skymodel_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)

    red_skycov = sky_covariance(nu, u=redundant_table.u(nu), v=redundant_table.v(nu), mode='baseline')
    full_skycov = sky_covariance(nu, u=skymodel_table.u(nu), v=skymodel_table.v(nu), mode='baseline')

    fig, axes = pyplot.subplots(1, 2, figsize = (10, 5))
    norm = colors.LogNorm()
    axes[0].imshow(red_skycov, norm = norm)
    axes[1].imshow(full_skycov, norm = norm)

    pyplot.show()
    return


if __name__ == "__main__":
    # compare_power_spectrum()
    test_baseline_covariance()





    # nu = 150e6
    # u = redundant_table.u(nu)
    # v = redundant_table.v(nu)
    # mode = 'baseline'
    # calculate_beam_power_spectrum(u, nu)
    #
    #
    # pyplot.imshow(cov)
    # pyplot.show()