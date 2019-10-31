import numpy
from matplotlib import pyplot
from cramer_rao_bound import telescope_bounds


def test_plot():
    path  = "data/HE.txt"
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

if __name__ == "__main__":
    test_plot()