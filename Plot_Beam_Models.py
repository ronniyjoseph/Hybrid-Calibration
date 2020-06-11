import numpy
import argparse
from matplotlib import colors

from src.radiotelescope import mwa_tile_beam
from src.radiotelescope import ideal_gaussian_beam
from src.radiotelescope import airy_beam
from src.radiotelescope import simple_mwa_tile

def main():
    theta = numpy.linspace(0, numpy.pi/2, 300)
    phi = 0

    mwa_model = mwa_tile_beam(theta, phi)
    gaussian_model = ideal_gaussian_beam(numpy.sin(theta), 0, nu=150e6, epsilon=0.42)
    airy_model = airy_beam(theta, diameter=4)
    mwa_simple = simple_mwa_tile(theta, 0)


    figure, axes = pyplot.subplots(1,1, figsize = (5,5))
    axes.plot(numpy.degrees(theta), numpy.abs(mwa_model), linewidth = 5, alpha=0.4, color='k')
    # axes.plot(numpy.degrees(theta), numpy.abs(gaussian_model))
    # axes.plot(numpy.degrees(theta), numpy.abs(airy_model))
    axes.plot(numpy.degrees(theta), numpy.abs(mwa_simple))
    axes.set_yscale('log')
    axes.set_ylim(1e-3,2)
    pyplot.show()


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()