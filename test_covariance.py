import numpy
import sys
from src.radiotelescope import RadioTelescope
from src.util import redundant_baseline_finder
from src.covariance import position_covariance
from src.covariance import beam_covariance
from src.covariance import sky_covariance
from src.covariance import blackman_harris_taper
from src.covariance import dft_matrix
from src.covariance import compute_ps_variance
from matplotlib import pyplot
from matplotlib import colors
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")

from plottools import plot_power_spectrum


def calculate_beam_power_spectrum(u, nu, save=False, plot_name="beam_2D_ps.pdf"):

    window_function = blackman_harris_taper(nu)
    taper1, taper2 = numpy.meshgrid(window_function, window_function)
    dftmatrix, eta = dft_matrix(nu)

    variance = numpy.zeros((len(u), len(nu)))

    print(f"Calculating covariances for all baselines")
    for i in range(len(u)):
        nu_cov = position_covariance(u=u[i], v=0, nu=nu, delta_u=1e-2)
        variance[i, :] = compute_ps_variance(taper1, taper2, nu_cov, dftmatrix)

    figure, axes = pyplot.subplots(1,1, figsize =(5,5))
    ps_norm = colors.LogNorm()

    plot_power_spectrum(u, eta[:int(len(eta) / 2)], nu, variance[:, :int(len(eta) / 2)], axes=axes, norm = ps_norm, colorbar_show =True, xlabel_show = True)
    pyplot.show()
    return

if __name__ == "__main__":
    # u = numpy.logspace(-2, numpy.log10(500), 100)
    # v = 0
    # u = 100
    # nu = numpy.linspace(140, 160, 200) * 1e6

    telescope = RadioTelescope(load = True, path="data/MWA_Hexes_Coordinates.txt")
    redundant_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)
    nu = 150e6
    u = redundant_table.u(nu)
    v = redundant_table.v(nu)
    mode = 'baseline'
    # calculate_beam_power_spectrum(u, nu)
    # cov = position_covariance(nu, u=u, v=v, position_precision=1e-1, mode = mode)
    # cov = beam_covariance(nu, u=u, v=v, broken_tile_fraction=1., mode = mode)
    cov = sky_covariance(nu, u=u, v=v, mode = mode)

    pyplot.imshow(cov)
    pyplot.show()