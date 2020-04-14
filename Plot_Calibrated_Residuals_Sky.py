import numpy
import argparse
from matplotlib import colors

from src.powerspectrum import from_frequency_to_eta
from src.powerspectrum import fiducial_eor_power_spectrum

from src.radiotelescope import RadioTelescope

from src.plottools import plot_2dpower_spectrum
from src.plottools import plot_power_contours
from src.generaltools import from_jansky_to_milikelvin

from src.covariance import calibrated_residual_error
from src.covariance import compute_weights

from src.util import redundant_baseline_finder

def main(labelfontsize = 16, ticksize= 11):
    output_path = "/home/ronniyjoseph/Sync/PhD/Thesis/ThesisTex/images/chapter_7/"


    contour_levels = numpy.array([1e0, 1e1, 1e2])

    # telescope_position_path = "./Data/MWA_Compact_Coordinates.txt"
    # tile_diameter = 4
    # fraction_broken = 0.3
    # model_limit = 1e-1

    telescope_position_path = "./Data/HERA_128.txt"
    tile_diameter = 14
    fraction_broken = 0.3
    model_limit = 1e-1

    k_perp_range = numpy.array([1e-4, 1.1e-1])

    u_range = numpy.logspace(0, numpy.log10(500), 50)

    frequency_range = numpy.linspace(135, 165, 251) * 1e6
    eta = from_frequency_to_eta(frequency_range)
    eor_power_spectrum = fiducial_eor_power_spectrum(u_range, eta)


    telescope = RadioTelescope(load=True, path=telescope_position_path)
    redundant_table = telescope.baseline_table
    # redundant_table = redundant_baseline_finder(telescope.baseline_table)
    weights = compute_weights(u_range, redundant_table.u_coordinates,
                              redundant_table.v_coordinates)


    sky_clocations = None# [(6e-2, 0.21), (4e-2, 0.13), (3e-2, 0.07 )]
    beam_clocations = sky_clocations
    total_clocations = sky_clocations
    # print(numpy.max(numpy.sqrt(redundant_table.u_coordinates**2 + redundant_table.v_coordinates**2)))
    sky_calibrated = calibrated_residual_error(u=u_range, nu=frequency_range, residuals='sky',
                                                 calibration_type='sky', weights = weights, tile_diameter=tile_diameter,
                                                    broken_baselines_weight = fraction_broken, model_limit=model_limit)
    beam_calibrated = calibrated_residual_error(u=u_range, nu=frequency_range, residuals='beam',
                                                         calibration_type='sky', weights = weights,
                                                                 tile_diameter=tile_diameter,
                                                    broken_baselines_weight = fraction_broken, model_limit=model_limit)
    total_calibrated = calibrated_residual_error(u=u_range, nu=frequency_range, residuals='both',
                                                           calibration_type='sky', weights = weights,
                                                                 tile_diameter=tile_diameter,
                                                    broken_baselines_weight = fraction_broken, model_limit=model_limit)


    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_2dpower_spectrum(u_range, eta, frequency_range, sky_calibrated, title="Sky Error", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=False,
                        xlabel_show=True, norm=ps_norm, ylabel_show=True)

    plot_2dpower_spectrum(u_range, eta, frequency_range, beam_calibrated, title="Beam Variations", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=False,
                        xlabel_show=True, norm=ps_norm, ylabel_show=False)

    plot_2dpower_spectrum(u_range, eta, frequency_range, total_calibrated, title="Total Error", axes=axes[2],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, ylabel_show=False, zlabel_show=True)

    plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(sky_calibrated,
                                                                                 frequency_range)/eor_power_spectrum,
                        axes=axes[0], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True,
                        norm=ps_norm, ylabel_show=True, contour_levels=contour_levels, contour_label_locs=sky_clocations)

    plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(beam_calibrated, frequency_range)/eor_power_spectrum,
                        axes=axes[1], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True,
                        norm=ps_norm, ylabel_show=False, contour_levels=contour_levels, contour_label_locs=beam_clocations)

    plot_power_contours(u_range, eta, frequency_range, from_jansky_to_milikelvin(total_calibrated, frequency_range)/eor_power_spectrum,
                        axes=axes[2], ratio=True, axes_label_font=labelfontsize, tickfontsize=ticksize, xlabel_show=True,
                        norm=ps_norm, ylabel_show=False, contour_levels=contour_levels, contour_label_locs=total_clocations)

    pyplot.tight_layout()
    # pyplot.savefig(output_path + "Calibrated_Residuals_Sky_MWA.pdf")
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
