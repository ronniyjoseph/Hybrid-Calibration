import argparse
import numpy


def main():
    output_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/sky_limit_100mJy/"
    # Set all relevant data paths
    redundant_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/sky_limit_100mJy/"
    skymodel_current_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/sky_limit_100mJy/"
    skymodel_deep_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/sky_limit_10mJy/"

    # Import all plotting data
    data_redundant = numpy.loadtxt(redundant_path + "redundant_crlb.txt")
    data_sky_model = numpy.loadtxt(skymodel_current_path + "skymodel_crlb.txt")
    data_sky_model_deep = numpy.loadtxt(skymodel_deep_path + "skymodel_crlb.txt")

    mwa_hexes_redundant= numpy.loadtxt(redundant_path + "mwa_hexes_redundant.txt")
    hera_128_redundant = numpy.loadtxt(redundant_path + "hera_128_redundant.txt")
    hera_350_redundant = numpy.loadtxt(redundant_path + "hera_350_redundant.txt")

    mwa_hexes_skymodel = numpy.loadtxt(skymodel_current_path + "mwa_hexes_skymodel.txt")
    mwa_compact_skymodel = numpy.loadtxt(skymodel_current_path + "mwa_compact_skymodel.txt")
    hera_350_skymodel = numpy.loadtxt(skymodel_current_path + "hera_350_skymodel.txt")
    ska_low_skymodel = numpy.loadtxt(skymodel_current_path + "ska_low_skymodel.txt")

    fig, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    # Plot Redundant Calibration Data
    axes[0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[1, :] + data_redundant[2, :]), 'C0',
                 label="Total")
    axes[0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[1, :]), 'C1', label="Relative")
    axes[0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[2, :]), 'C2', label="Absolute")
    axes[0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[3, :]), "k--", label="Thermal")

    # Plot Telescope Bounds
    axes[0].plot(mwa_hexes_redundant[0], numpy.sqrt(mwa_hexes_redundant[1] + mwa_hexes_redundant[2]), marker="x",
                 linestyle = 'None', label="MWA Hexes")
    axes[0].plot(hera_128_redundant[0], numpy.sqrt(hera_128_redundant[1] + hera_128_redundant[2]), marker="o",
                 linestyle = 'None', label="HERA 128")
    axes[0].plot(hera_350_redundant[0], numpy.sqrt(hera_350_redundant[1] + hera_350_redundant[2]), marker='H',
                 linestyle='None', label="HERA 350")

    # Plot Sky Calibration Data
    axes[1].semilogy(data_sky_model[0, :], numpy.sqrt(data_sky_model[1, :]),color = "C1", label="Sky Based")
    axes[1].semilogy(data_sky_model[0, :], numpy.sqrt(data_sky_model[2, :]), "k--", label="Thermal")

    # Plot Telescope Bounds for sky based calibration
    axes[1].plot(mwa_hexes_skymodel[0], numpy.sqrt(mwa_hexes_skymodel[1]), marker='x', linestyle = 'None',
                 label="MWA Hexes")
    axes[1].plot(mwa_compact_skymodel[0], numpy.sqrt(mwa_compact_skymodel[1]), marker='+',linestyle = 'None',
                 label="MWA Compact")
    axes[1].plot(hera_350_skymodel[0], numpy.sqrt(hera_350_skymodel[1]), marker='H', linestyle = 'None', label="HERA 350")
    axes[1].plot(ska_low_skymodel[0], numpy.sqrt(ska_low_skymodel[1]), marker='*', linestyle = 'None', label="SKA_LOW1")

    axes[2].plot(data_redundant[0, :], numpy.sqrt(data_redundant[1, :] + data_redundant[2, :]), 'C0',
                 label="Redundancy Based")
    axes[2].plot(data_sky_model[0, :], numpy.sqrt(data_sky_model[1, :]), color = "C2", label="Sky Based 10 mJy")
    axes[2].plot(data_sky_model_deep[0, :], numpy.sqrt(data_sky_model_deep[1, :]),color = "C1", label="Sky Based 100 mJy")

    axes[0].set_ylabel("Gain Error")
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    axes[0].set_xlabel("Number of Antennas")
    axes[1].set_xlabel("Number of Antennas")
    axes[2].set_xlabel("Number of Antennas")

    axes[0].set_ylim([1e-4, 1])
    axes[1].set_ylim([1e-4, 1])
    axes[2].set_ylim([1e-4, 1])

    axes[0].legend()
    axes[1].legend()
    axes[2].legend()

    axes[0].set_title("Redundant Calibration")
    axes[1].set_title("Sky Based Calibration")
    axes[2].set_title("Comparison")
    fig.savefig(output_path + "Calibration_FIM.pdf", transparent = True)
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