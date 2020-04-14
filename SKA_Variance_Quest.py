import numpy
import argparse

from src.radiotelescope import RadioTelescope
from src.util import redundant_baseline_finder
from cramer_rao_bound import telescope_bounds
from src.skymodel import sky_moment_returner


def main(model_limit = 1e-2):
    telescope = RadioTelescope(load=True, path="data/SKA_Low_v5_ENU_mini.txt")
    baselines = telescope.baseline_table
    lengths = numpy.sqrt(baselines.u_coordinates**2 + baselines.v_coordinates**2)
    print(numpy.min(lengths)*2)
    # telescope_bounds("data/SKA_Low_v5_ENU_mini.txt", bound_type="sky")
    # mwa_hexes_sky = telescope_bounds("data/MWA_Hexes_Coordinates.txt", bound_type="sky")
    # print("MWA Compact")
    # mwa_compact_sky = telescope_bounds("data/MWA_Compact_Coordinates.txt", bound_type="sky")

    # telescope = RadioTelescope(load=True, path=position_table_path)
    # print("Grouping baselines")
    # redundant_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)
    #
    # print(f"Ratio Number of groups/number of baselines:"
    #       f"{len(numpy.unique(redundant_table.group_indices))/len(redundant_table.antenna_id1)}")
    # pyplot.scatter(redundant_table.u_coordinates, redundant_table.v_coordinates, c = redundant_table.group_indices,
    #                cmap="Set3")
    # pyplot.show()


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

