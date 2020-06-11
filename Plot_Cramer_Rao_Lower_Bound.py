import argparse
import numpy


def main(plot_telescopes = True, axes_label_font = 20, tickfontsize=15,title_font=20 ):
    output_path = "/home/ronniyjoseph/Sync/PhD/Projects/hybrid_calibration/Plots/"
    # Set all relevant data paths
    redundant_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/sky_limit_100mJy/"
    skymodel_current_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/sky_limit_100mJy/"
    telescope_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/sky_limit_100mJy_Thesis/"
    skymodel_deep_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/sky_limit_10mJy/"


    # Import all plotting data
    data_redundant = numpy.loadtxt(redundant_path + "redundant_crlb.txt")
    data_sky_model = numpy.loadtxt(skymodel_current_path + "skymodel_crlb.txt")
    data_sky_model_deep = numpy.loadtxt(skymodel_deep_path + "skymodel_crlb.txt")
    if plot_telescopes:
        mwa_hexes_redundant= numpy.loadtxt(telescope_path + "mwa_hexes_redundant.txt")
        hera_128_redundant = numpy.loadtxt(telescope_path + "hera_128_redundant.txt")
        hera_350_redundant = numpy.loadtxt(telescope_path + "hera_350_redundant.txt")

        mwa_hexes_skymodel = numpy.loadtxt(telescope_path + "mwa_hexes_skymodel.txt")
        mwa_compact_skymodel = numpy.loadtxt(telescope_path + "mwa_compact_skymodel.txt")
        hera_350_skymodel = numpy.loadtxt(telescope_path + "hera_350_skymodel.txt")
        ska_low_skymodel = numpy.loadtxt(telescope_path + "ska_low_skymodel.txt")

    fig, axes = pyplot.subplots(2, 2, figsize=(10, 10))

    # Plot Redundant Calibration Data
    axes[0, 0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[1, :] + data_redundant[2, :]), 'C0',
                 label="Total")
    axes[0, 0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[1, :]), 'C1', label="Relative")
    axes[0, 0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[2, :]), 'C2', label="Absolute")
    axes[0, 0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[3, :]), "k--", label="Thermal")

    # Plot Sky Calibration Data
    axes[0, 1].semilogy(data_sky_model[0, :], numpy.sqrt(data_sky_model[1, :]), color="C1", label="Sky Based")
    axes[0, 1].semilogy(data_sky_model[0, :], numpy.sqrt(data_sky_model[2, :]), "k--", label="Thermal")

    # Plot Telescope Bounds
    if plot_telescopes:
        axes[0, 0].plot(mwa_hexes_redundant[0], numpy.sqrt(mwa_hexes_redundant[1] + mwa_hexes_redundant[2]), marker="x",
                     linestyle = 'None', label="MWA Hexes")
        axes[0, 0].plot(hera_128_redundant[0], numpy.sqrt(hera_128_redundant[1] + hera_128_redundant[2]), marker="o",
                     linestyle = 'None', label="HERA 128")
        axes[0, 0].plot(hera_350_redundant[0], numpy.sqrt(hera_350_redundant[1] + hera_350_redundant[2]), marker='H',
                     linestyle='None', label="HERA 350")

        # Plot Telescope Bounds for sky based calibration
        axes[0, 1].plot(mwa_hexes_skymodel[0], numpy.sqrt(mwa_hexes_skymodel[1]), marker='x', linestyle = 'None',
                     label="MWA Hexes")
        axes[0, 1].plot(mwa_compact_skymodel[0], numpy.sqrt(mwa_compact_skymodel[1]), marker='+',linestyle = 'None',
                     label="MWA Compact")
        axes[0, 1].plot(hera_350_skymodel[0], numpy.sqrt(hera_350_skymodel[1]), marker='H', linestyle = 'None', label="HERA 350")
        axes[0, 1].plot(ska_low_skymodel[0], numpy.sqrt(ska_low_skymodel[1]), marker='*', linestyle = 'None', label="SKA_LOW1")

    axes[1, 0].plot(data_redundant[0, :], numpy.sqrt(data_redundant[1, :] + data_redundant[2, :]), 'C0',
                 label="Redundancy Based")
    axes[1, 0].plot(data_sky_model[0, :], numpy.sqrt(data_sky_model[1, :]),color = "C1", label="Sky Based 100 mJy")
    axes[1, 0].plot(data_sky_model_deep[0, :], numpy.sqrt(data_sky_model_deep[1, :]), color = "C2", label="Sky Based 10 mJy")

    axes[0, 0].set_ylabel("Gain Error", fontsize=axes_label_font)
    axes[1, 0].set_ylabel("Gain Error", fontsize=axes_label_font)


    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')
    axes[1, 0].set_yscale('log')

    axes[0, 1].set_xlabel("Number of Antennas", fontsize=axes_label_font)
    axes[1, 0].set_xlabel("Number of Antennas", fontsize=axes_label_font)
    # axes[2].set_xlabel("Number of Antennas", fontsize=axes_label_font)

    axes[0, 0].set_ylim([1e-3, 1])
    axes[0, 1].set_ylim([1e-3, 1])
    axes[1, 0].set_ylim([1e-3, 1])

    axes[0, 0].legend(fontsize = tickfontsize)
    axes[0, 1].legend(fontsize = tickfontsize)
    axes[1, 0].legend(fontsize = tickfontsize)

    axes[0, 0].set_title("Redundant Calibration", fontsize=title_font)
    axes[0, 1].set_title("Sky Based Calibration", fontsize=title_font)
    axes[1, 0].set_title("Comparison", fontsize=title_font)

    axes[0, 0].tick_params(axis='both', which='major', labelsize=tickfontsize)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=tickfontsize)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=tickfontsize)
    axes[1, 1].axis("off")
    pyplot.tight_layout()
    fig.savefig(output_path + "Calibration_FIM_Thesis2.0.pdf", transparent = True)
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