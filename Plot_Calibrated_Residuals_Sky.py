import numpy
import argparse
from matplotlib import colors

from src.powerspectrum import from_frequency_to_eta
from src.powerspectrum import fiducial_eor_power_spectrum

from src.plottools import plot_power_spectrum
from src.plottools import plot_power_contours
from src.generaltools import from_jansky_to_milikelvin

from src.covariance import calculate_residual_error


def main(labelfontsize = 16, ticksize= 11):
    output_path = "/data/rjoseph/Hybrid_Calibration/theoretical_calculations/power_spectra/"
    k_perp_range = numpy.array([1e-4, 1.1e-1])
    u_range = numpy.logspace(-1, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6
    contour_levels = numpy.array([1e1, 1e2, 1e3])

    eta = from_frequency_to_eta(frequency_range)

    eor_power_spectrum = fiducial_eor_power_spectrum(u_range, eta)

    sky_raw, sky_calibrated = calculate_residual_error(u=u_range, nu=frequency_range, residuals='sky', calibration_type='sky')
    beam_raw, beam_calibrated = calculate_residual_error(u=u_range, nu=frequency_range, residuals='beam',calibration_type='sky')
    total_raw, total_calibrated = calculate_residual_error(u=u_range, nu=frequency_range, residuals='both',calibration_type='sky')
    #
    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_power_spectrum(u_range, eta, frequency_range, sky_calibrated, title="Sky Error", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=False,
                        xlabel_show=True, norm=ps_norm, ylabel_show=True)

    plot_power_spectrum(u_range, eta, frequency_range, beam_calibrated, title="Beam Variations", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=False,
                        xlabel_show=True, norm=ps_norm, ylabel_show=False)

    plot_power_spectrum(u_range, eta, frequency_range, total_calibrated, title="Total Error", axes=axes[2],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, ylabel_show=False, zlabel_show=True)

    # pyplot.imshow(numpy.log10(from_jansky_to_milikelvin(sky_calibrated, frequency_range).T), origin='lower')
    # pyplot.show()
    plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(sky_calibrated, frequency_range)/eor_power_spectrum,
                        axes=axes[0], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, norm=ps_norm, ylabel_show=True, contour_levels=contour_levels)

    plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(beam_calibrated, frequency_range)/eor_power_spectrum,
                        axes=axes[1], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, norm=ps_norm, ylabel_show=False, contour_levels=contour_levels)

    plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(total_calibrated, frequency_range)/eor_power_spectrum,
                        axes=axes[2], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True, norm=ps_norm, ylabel_show=False, contour_levels=contour_levels)
    #
    pyplot.savefig(output_path + "Calibrated_Residuals_Redundant.pdf")
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
