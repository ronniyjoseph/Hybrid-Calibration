import numpy
import argparse

from src.radiotelescope import RadioTelescope
from src.util import redundant_baseline_finder


def main():
    position_table_path = "data/SKA_Low_v5_ENU_fullcore.txt"
    telescope = RadioTelescope(load=True, path=position_table_path)
    print("Grouping baselines")
    redundant_table = redundant_baseline_finder(telescope.baseline_table, group_minimum=1)

    print(f"Ratio Number of groups/number of baselines:"
          f"{len(numpy.unique(redundant_table.group_indices))/len(redundant_table.antenna_id1)}")
    pyplot.scatter(redundant_table.u_coordinates, redundant_table.v_coordinates, c = redundant_table.group_indices,
                   cmap="Set3")
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
