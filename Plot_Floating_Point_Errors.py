import numpy
import argparse
from matplotlib import colors

from src.powerspectrum import from_frequency_to_eta
from src.plottools import plot_power_spectrum
from Plot_Residuals_Redundant import calculate_position_power_spectrum
from Plot_Residuals_Sky import calculate_sky_power_spectrum


def main(labelfontsize = 16, ticksize= 11):
    k_perp_range = numpy.array([1e-4, 1.1e-1])
    u_range = numpy.logspace(-1, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6
    eta = from_frequency_to_eta(frequency_range)

    position_error_power = calculate_position_power_spectrum(u=u_range, nu=frequency_range)
    sky_error_power = calculate_sky_power_spectrum(u=u_range, nu=frequency_range)

    figure, axes = pyplot.subplots(1, 2, figsize=(10, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_power_spectrum(u_range, eta, frequency_range, sky_error_power, title="Sky Model Error", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, ylabel_show=True)

    plot_power_spectrum(u_range, eta, frequency_range, position_error_power, title="Position Error", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, zlabel_show=True)

    figure.tight_layout()

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
