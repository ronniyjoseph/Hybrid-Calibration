import numpy
import argparse
from matplotlib import colors


def main(labelfontsize = 16, ticksize= 11):
    k_perp_range = numpy.array([1e-4, 1.1e-1])
    u_range = numpy.logspace(0, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6

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
