import sys
import numpy
from matplotlib import pyplot

sys.path.append("../../beam_perturbations/code/tile_beam_perturbations/")
from analytic_covariance import moment_returner
from radiotelescope import beam_width

def cramer_rao_bound_calculator():

    return

def beam_variance(nu):
    mu_2 = moment_returner()
    beamwidth = beam_width(nu)

    2*numpy.pi*mu_2

    return

def position_variance():
    return