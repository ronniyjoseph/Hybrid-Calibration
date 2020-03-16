import numpy
import argparse
from matplotlib import colors

from src.covariance import sky_covariance
from src.covariance import position_covariance
from src.covariance import beam_covariance
from src.powerspectrum import compute_power
from src.covariance import gain_error_covariance
from src.powerspectrum import from_frequency_to_eta
from src.plottools import plot_power_spectrum


def main(labelfontsize = 16, ticksize= 11):
    k_perp_range = numpy.array([1e-4, 1.1e-1])
    u_range = numpy.logspace(-1, numpy.log10(500), 100)
    frequency_range = numpy.linspace(135, 165, 251) * 1e6
    eta = from_frequency_to_eta(frequency_range)

    position_raw, position_calibrated = calculate_residual_error(u=u_range, nu=frequency_range, residuals='position')
    beam_raw, beam_calibrated = calculate_residual_error(u=u_range, nu=frequency_range, residuals='beam')
    total_raw, total_calibrated = calculate_residual_error(u=u_range, nu=frequency_range, residuals='both')

    figure, axes = pyplot.subplots(1, 3, figsize=(15, 5))

    ps_norm = colors.LogNorm(vmin=1e3, vmax=1e15)

    plot_power_spectrum(u_range, eta, frequency_range, position_calibrated, title="Position Error", axes=axes[0],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, ylabel_show=True)

    plot_power_spectrum(u_range, eta, frequency_range, beam_calibrated, title="Beam Variations", axes=axes[1],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm)

    plot_power_spectrum(u_range, eta, frequency_range, total_calibrated, title="Total Error", axes=axes[2],
                        axes_label_font=labelfontsize, tickfontsize=ticksize, colorbar_show=True,
                        xlabel_show=True, norm=ps_norm, zlabel_show=True)

    figure.tight_layout()

    pyplot.show()

    return


def calculate_residual_error(u, nu, residuals='both', broken_baselines_weight = 1, weights = None, path="./", plot=True):
    cal_variance = numpy.zeros((len(u), int(len(nu) / 2)))
    raw_variance = numpy.zeros((len(u), int(len(nu) / 2)))

    gain_averaged_covariance = gain_error_covariance(u, nu, residuals=residuals, weights= weights,
                                                     broken_baseline_weight = broken_baselines_weight, calibration_type='redundant')

    # Compute the gain corrected residuals at all u scales
    # if residuals == "position":
    #     residual_variance = position_covariance(0, 0, frequency_range)
    # elif residuals == "beam":
    #     residual_variance = broken_baselines_weight **2*beam_covariance(0, v=0, nu=frequency_range)
    # elif residuals == 'both':
    #     residual_variance = sky_covariance(0, 0, frequency_range) + \
    #                         broken_baselines_weight **2*beam_covariance(0, v=0, nu=frequency_range)
    #
    # gain = residual_variance / sky_covariance(0, 0, frequency_range)

    for i in range(len(u)):
        if residuals == "position":
            residual_covariance = position_covariance(u[i], 0, nu)
            blaah = 0
        elif residuals == "beam":
            residual_covariance = beam_covariance(u[i], v=0, nu=nu, broken_tile_fraction=broken_baselines_weight,
                                                  calibration_type='redundant')
            blaah = 0
        elif residuals == 'both':
            residual_covariance = position_covariance(u[i], 0, nu) + \
                                  beam_covariance(u[i], v=0, nu=nu, broken_tile_fraction=broken_baselines_weight,
                                                  calibration_type='redundant')
            blaah = 0

        model_covariance = sky_covariance(u[i], 0, nu, S_low=1, S_high=10)
        sky_residuals = sky_covariance(u[i], 0, nu, S_low=1e-3, S_high=1)
        scale = numpy.diag(numpy.zeros_like(nu)) + 1 + blaah
        if weights is None:
            nu_cov = 2*gain_averaged_covariance*model_covariance + \
                     (scale + 2*gain_averaged_covariance)*sky_residuals
        else:
            nu_cov = 2*gain_averaged_covariance[i, ...]*model_covariance + \
                     (scale + 2*gain_averaged_covariance[i, ...])*sky_residuals

        cal_variance[i, :] = compute_power(nu, nu_cov)
        raw_variance[i, :] = compute_power(nu, residual_covariance)

    return raw_variance, cal_variance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()
