import sys
from scipy.constants import c
import numpy
import matplotlib
from matplotlib import pyplot

# Import local codes
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")

from radiotelescope import AntennaPositions
from radiotelescope import BaselineTable
from radiotelescope import ideal_gaussian_beam
from skymodel import SkyRealisation
from skymodel import apparent_fluxes_numba
from util import split_visibility
from util import find_sky_model_sources
from util import generate_sky_model_vectors


def main():
    # 12 500 sources within 2.5 primary beam width
    # about 10 calibrator sources with primary beam weighted flux > 3x RMS
    # beam width 13 wavelength dishes
    # 8x8 array with 20 wavelength spacing
    # position noise of 0.04 wavelengths
    # per visibility noise 0.1*brightes source

    frequency_range = numpy.array([150]) * 1e6
    n_antenna_grid_points = 8
    separation_scale = 20
    antenna_scale = 13 #wavelengths

    wavelength = c/frequency_range[0]
    array_size = (n_antenna_grid_points - 1)*wavelength*separation_scale
    antenna_size = antenna_scale*wavelength

    antenna_table = AntennaPositions(load=False, shape=['square', array_size/2, n_antenna_grid_points, 0, 0])
    baseline_table = BaselineTable(position_table = antenna_table, frequency_channels = frequency_range)

    # We go down to 40 mili-Jansky to get about 10 calibration sources
    sky_realisation = SkyRealisation(sky_type="random", flux_low=40e-3, flux_high=10)
    sky_model = find_sky_model_sources(sky_realisation, frequency_range, antenna_size = antenna_size)

    # now compute the visibilities
    data_complex = sky_realisation.create_visibility_model(baseline_table, frequency_range, antenna_size = antenna_size)
    data_split = split_visibility(data_complex)

    source_vectors = generate_sky_model_vectors(sky_model, baseline_table, frequency_range, antenna_size)
    print(source_vectors.shape)

    return


def plot_sky_counts(fluxes):
    bins = numpy.logspace(numpy.log10(fluxes.min()), numpy.log10(fluxes.max()), 100)
    hist, edges = numpy.histogram(fluxes, bins=bins)

    bin_centers = (edges[:len(edges)-1]+edges[1:])/2.
    pyplot.plot(bin_centers, hist)
    pyplot.xscale('log')
    pyplot.yscale('log')

    return

if "__main__" == __name__:
    main()