import sys
import numpy
from matplotlib import pyplot

#Import local codes
sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
sys.path.append("../corrcal2")

import skymodel
import radiotelescope
from gain_variance_simulation import get_observations_numba
from corrcal2 import sparse_2level
from corrcal2 import grid_data

################################################
# ant1.dat contains the antenna id/index for the first antenna in each baseline
# ant2.dat contains the antenna id/index for the second antenna in each baseline
# gtmp.dat contains the antenna gains split up in real and imaginary components
# vis.dat contains the visibility data split up in real and imaginary components

#signal_sparse2_test.dat contains a lot of information that apparently needs to be shaped in the right way
#
################################################


def main():
    calibrator_flux = 100
    calibrator_l = 0
    calibrator_m = 0
    frequency_range = numpy.array([150])*1e6

    radio_telescope = radiotelescope.RadioTelescope(load = False, shape = ['hex', 7, 0, 0 ])
    sky_realisation = skymodel.SkyRealisation(sky_type = "random", flux_high = 1)

    sky_realisation.fluxes = numpy.append(sky_realisation.fluxes, calibrator_flux)
    sky_realisation.l_coordinates = numpy.append(sky_realisation.l_coordinates, calibrator_l)
    sky_realisation.m_coordinates = numpy.append(sky_realisation.m_coordinates, calibrator_m)

    visibility_data = get_observations_numba(sky_realisation, radio_telescope.baseline_table, frequency_range)
    data, u, v, noise, ant1, ant2, edges, ii, isonj = grid_data(visibility_data,
                                                                radio_telescope.baseline_table.u_coordinates,
                                                                radio_telescope.baseline_table.v_coordinates,
                                                                numpy.zeros_like(visibility_data),
                                                                radio_telescope.baseline_table.antenna_id1,
                                                                radio_telescope.baseline_table.antenna_id2)

    return




if "__main__" == __name__:
    main()