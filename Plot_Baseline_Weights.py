import sys
import numpy
import powerbox
import argparse
import matplotlib.colors as colors

from src.radiotelescope import RadioTelescope
from src.radiotelescope import ideal_gaussian_beam
from src.util import redundant_baseline_finder

from src.skymodel import SkyRealisation
from src.generaltools import from_lm_to_theta_phi
from src.plottools import colorbar

from scipy.signal import convolve2d

sys.path.append("../")

def main(plot_u_dist = False ,plot_array_matrix = False, plot_inverse_matrix = False, plot_weights = False ,
         grid_weights = True, binned_weights = True):
    show_plot = True
    save_plot = False
    path = "./Data/MWA_Compact_Coordinates.txt"
    plot_folder = "../Plots/Analytic_Covariance/"

    telescope = RadioTelescope(load=True, path=path)

    if plot_u_dist:
        make_plot_uv_distribution(telescope, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)


    sky_matrix = sky_matrix_constructor(telescope)
    redundant_matrix = redundant_matrix_constructor(telescope)

    print(sky_matrix.shape)
    print(f"Sky Calibration uses {len(sky_matrix[:, 3])/2} baselines")
    print(f"Redundant Calibration uses {len(redundant_matrix[:, 3])/2} baselines")

    if plot_array_matrix:
        make_plot_array_matrix(sky_matrix, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)
        make_plot_array_matrix(redundant_matrix, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)

    print(numpy.linalg.cond(sky_matrix))
    print(numpy.linalg.cond(redundant_matrix))

    inverse_sky_matrix = numpy.linalg.pinv(sky_matrix)
    inverse_redundant_matrix = numpy.linalg.pinv(redundant_matrix)

    if plot_inverse_matrix:
        make_plot_array_matrix(inverse_sky_matrix.T, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)
        make_plot_array_matrix(inverse_redundant_matrix.T, show_plot=show_plot, save_plot=save_plot, plot_folder=plot_folder)

    sky_weights = numpy.sqrt((numpy.abs(inverse_sky_matrix[::2, ::2])**2 +
                                   numpy.abs(inverse_sky_matrix[1::2, 1::2])**2))
    redundant_weights = numpy.sqrt((numpy.abs(inverse_redundant_matrix[::2, ::2])**2 +
                                   numpy.abs(inverse_redundant_matrix[1::2, 1::2])**2))

    print(f"Every Sky calibrated Tile sees {len(sky_weights[0,:][sky_weights[0, :] > 1e-4])}")
    print(f"Every redundant Tile sees {len(redundant_weights[0,:][redundant_weights[0, :] > 1e-4])}")

    u_bins = numpy.linspace(0, 375, 101)
    redundant_bins, redundant_uu_weights, red_counts = compute_binned_weights(redundant_baseline_finder(telescope.baseline_table),
                                                                  redundant_weights, binned=True, u_bins=u_bins)
    sky_bins, sky_uu_weights, sky_counts = compute_binned_weights(telescope.baseline_table, sky_weights, binned=True, u_bins=u_bins)

    print(redundant_uu_weights.shape)
    figure, axes = pyplot.subplots(2,3, figsize=(10, 15))

    norm = colors.LogNorm()
    redplot = axes[1, 0].pcolor(redundant_bins, redundant_bins, redundant_uu_weights, norm = norm)
    norm = colors.LogNorm()

    skyplot = axes[0, 0].pcolor(sky_bins, sky_bins, sky_uu_weights, norm=norm)

    norm = colors.LogNorm()
    countsredplot = axes[1, 1].pcolor(redundant_bins, redundant_bins, red_counts, norm = norm)
    norm = colors.LogNorm()

    countskyplot = axes[0, 1].pcolor(sky_bins, sky_bins, sky_counts, norm = norm)

    norm = colors.LogNorm()
    normredplot = axes[1, 2].pcolor(redundant_bins, redundant_bins, redundant_uu_weights / red_counts, norm = norm)
    normskyplot = axes[0, 2].pcolor(sky_bins, sky_bins, sky_uu_weights / sky_counts, norm = norm)


    axes[1, 0].set_xlabel(r"$u\,[\lambda]$")
    axes[1, 1].set_xlabel(r"$u\,[\lambda]$")
    axes[1, 1].set_xlabel(r"$u\,[\lambda]$")

    axes[0, 0].set_ylabel(r"$u^{\prime}\,[\lambda]$")
    axes[1, 0].set_ylabel(r"$u^{\prime}\,[\lambda]$")


    axes[0, 0].set_title("Sky Weights")
    axes[1, 0].set_title("Redundant Weights")

    axes[0, 1].set_title("Sky Normalisation")
    axes[1, 1].set_title("Redundant Normalisation")

    axes[0, 2].set_title("Sky Normalised Weights")
    axes[1, 2].set_title("Redundant Normalised Weights")

    colorbar(redplot)
    colorbar(skyplot)
    colorbar(countsredplot)
    colorbar(countskyplot)
    colorbar(normredplot)
    colorbar(normskyplot)

    pyplot.tight_layout()
    pyplot.show()



    return

""""
    baseline_weights = numpy.sqrt((numpy.abs(inverse_array_matrix[::2, ::2])**2 +
                                   numpy.abs(inverse_array_matrix[1::2, 1::2])**2))
    print(f"Every Tile sees {len(baseline_weights[0,:][baseline_weights[0, :] > 1e-4])}")

    # baseline_weights = numpy.sqrt(numpy.abs(inverse_array_matrix[:int(len(telescope.antenna_positions.antenna_ids) - 1), :int(len(baseline_lengths))])**2 + \
    #                    numpy.abs(inverse_array_matrix[int(len(telescope.antenna_positions.antenna_ids) -1 ):, :int(len(baseline_lengths)):])**2)
    if plot_weights:
        figure_weights, axes_weights = pyplot.subplots(1,1)
        normalised_weights = axes_weights.imshow(baseline_weights)
        axes_weights.set_title("Antenna Baseline Weights")
        colorbar(normalised_weights)

        # blaah = numpy.unique(baseline_weights)
        # figblaah, axblaah = pyplot.subplots(1,1)
        # axblaah.hist(baseline_weights.flatten(), bins = 100)
        # axblaah.set_yscale('log')

    uu_weights = numpy.zeros((len(baseline_lengths), len(baseline_lengths)))
    baselines = telescope.baseline_table
    antennas = telescope.antenna_positions.antenna_ids
    for i in range(len(baseline_lengths)):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        if index1 == 0:
            baseline_weights1 = 0
        else:
            baseline_weights1 = baseline_weights[index1 - 1, :]

        if index2 == 0:
            baseline_weights2 = 0
        else:
            baseline_weights2 = baseline_weights[index2 - 1, :]
        uu_weights[i, :] = numpy.sqrt((baseline_weights1**2 + baseline_weights2**2))

    u_bins = numpy.linspace(0, numpy.max(baseline_lengths), 101)
    bin_size = (u_bins.max() - u_bins.min())/len(u_bins)
    sorted_indices = numpy.argsort(baseline_lengths)
    sorted_weights = uu_weights[sorted_indices, :][:, sorted_indices]

    bin_indices = numpy.digitize(baseline_lengths[sorted_indices], u_bins)
    print(f"A uncalibrated baseline sees  {len(uu_weights[:, 190][uu_weights[:, 190] > 1e-4])}")
    print(f"A calibrated baseline sees {len(uu_weights[190, :][uu_weights[190, :] > 1e-4])}")

    print(f"A sorted uncalibrated baseline sees  {len(sorted_weights[:, 2489][sorted_weights[:, 2489] > 1e-4])}")
    print(f"A sorted calibrated baseline sees {len(sorted_weights[190, :][sorted_weights[190, :] > 1e-4])}")


    if grid_weights:
        fig_cal, axes_cal = pyplot.subplots(1,2, figsize = (100, 50))
        cal_plot = axes_cal[0].imshow(uu_weights, origin = 'lower', interpolation = 'none')
        axes_cal[0].set_xlabel("Uncalibrated Baseline Index")
        axes_cal[0].set_ylabel("Calibrated Baseline Index")
        axes_cal[0].set_title("Quadrature Added Real and Imaginary Weights")
        colorbar(cal_plot)

        sorted_plot = axes_cal[1].imshow(sorted_weights, interpolation='none', origin='lower')
        axes_cal[1].set_xlabel("Uncalibrated Baseline Index")
        axes_cal[1].set_title(" Baseline Length Sorted Weights")
        colorbar(sorted_plot)

        for i in range(len(bin_indices)):
            if i == 0:
                pass
            elif bin_indices[i] == bin_indices[i-1]:
                pass
            else:
                axes_cal[1].axvline(i, linestyle = "-", color = 'gray', alpha = 0.4 )
                axes_cal[1].axhline(i, linestyle = "-", color = 'gray', alpha = 0.4)

        # unique_values = numpy.unique(u_u_weights)
        # figs, axs = pyplot.subplots(1,1)
        # axs.hist(unique_values, bins = 1000)

    if binned_weights:
        bin_counter = numpy.zeros_like(uu_weights)
        bin_counter[uu_weights != 0] = 1

        uu1, uu2 = numpy.meshgrid(baseline_lengths, baseline_lengths)
        flattened_uu1 = uu1.flatten()
        flattened_uu2 = uu2.flatten()

        computed_weights = numpy.histogram2d(flattened_uu1, flattened_uu2,  bins=u_bins,
                                             weights=uu_weights.flatten())
        computed_counts = numpy.histogram2d(flattened_uu1, flattened_uu2,  bins=u_bins,
                                            weights = bin_counter.flatten())

        figure_binned, axes_binned = pyplot.subplots(3, 3,  figsize = (12,  15), subplot_kw= dict(aspect = 'equal') )

        summed_norm = colors.LogNorm()
        counts_norm = colors.LogNorm()
        averaged_norm = colors.LogNorm()

        summed = axes_binned[0, 1].pcolor(u_bins, u_bins, computed_weights[0], norm = summed_norm)
        counts = axes_binned[0, 2].pcolor(u_bins, u_bins, computed_counts[0], norm = counts_norm)
        averaged = axes_binned[0, 0].pcolor(u_bins, u_bins, computed_weights[0]/computed_counts[0]/bin_size**2,
                                                      norm = averaged_norm)

        averaged_cbar = colorbar(averaged)
        counts_cbar = colorbar(counts)
        summed_cbar = colorbar(summed)


        axes_binned[0,1].set_title(r"Summed Weights")
        axes_binned[0,2].set_title(r"Baseline Counts")
        axes_binned[0,0].set_title(r"Averaged Weights")



        baseline_pdf = numpy.histogram(baseline_lengths, bins = u_bins, density = True)

        ww1, ww2 = numpy.meshgrid(baseline_pdf[0], baseline_pdf[0])
        summed_anorm = colors.LogNorm( )
        counts_anorm = colors.LogNorm( )
        averaged_anorm = colors.LogNorm( )

        approx_sum = convolve2d(numpy.diag(baseline_pdf[0]), ww1, mode = 'same')

            #convolve2d(numpy.diag(baseline_pdf[0])*bin_size**2*len(baseline_lengths)**2, ww1*ww2*bin_size**2*len(baseline_lengths)**2, mode = 'same')*ww1*ww2*bin_size**2*len(baseline_lengths)**2
            # convolve2d(ww2, convolve2d(numpy.diag(baseline_pdf[0]), ww1, mode = 'same'), mode='same')# (ww1*ww2)*bin_size**2*len(baseline_lengths)
        approx_counts = (ww1*ww2)*bin_size**2*len(baseline_lengths)**2
        approx_averaged = approx_sum/approx_counts

        averaged_aplot = axes_binned[1, 0].pcolor(u_bins, u_bins , approx_averaged, norm = averaged_anorm)
        sum_aplot = axes_binned[1, 1].pcolor(u_bins, u_bins, approx_sum, norm=summed_anorm)
        counts_aplot = axes_binned[1, 2].pcolor(u_bins, u_bins, approx_counts, norm=counts_anorm)

        acbar_sum = colorbar(sum_aplot)
        acbar_averaged = colorbar(averaged_aplot)
        acbar_counts = colorbar(counts_aplot)

        axes_binned[2, 0].set_title(r"Differences")

        blaah = axes_binned[2, 0].pcolor(u_bins, u_bins, computed_weights[0]/computed_counts[0] - approx_sum)

        axes_binned[2, 0].set_ylabel(r"$u^{\prime}\,[\lambda]$")
        colorbar(blaah)

        axes_binned[0, 0].set_ylabel(r"$u^{\prime}\,[\lambda]$")
        axes_binned[1, 0].set_ylabel(r"$u^{\prime}\,[\lambda]$")
        axes_binned[2, 0].set_ylabel(r"$u^{\prime}\,[\lambda]$")


        axes_binned[2, 0].set_xlabel(r"$u\,[\lambda]$")
        axes_binned[1, 1].set_xlabel(r"$u\,[\lambda]$")
        axes_binned[1, 2].set_xlabel(r"$u\,[\lambda]$")

        figure_binned.savefig(plot_folder + "Baseline_Weights_uu.pdf")


    pyplot.show()
"""

def compute_binned_weights(baseline_table, baseline_weights, binned=True, u_bins = None):
    antennas = numpy.unique([baseline_table.antenna_id1, baseline_table.antenna_id2] )
    baseline_lengths = numpy.sqrt(baseline_table.u_coordinates**2 + baseline_table.v_coordinates**2)
    uu_weights = numpy.zeros((len(baseline_lengths), len(baseline_lengths)))

    for i in range(len(baseline_lengths)):
        index1 = numpy.where(antennas == baseline_table.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baseline_table.antenna_id2[i])[0]
        if index1 == 0:
            baseline_weights1 = 0
        else:
            baseline_weights1 = baseline_weights[index1 - 1, :]

        if index2 == 0:
            baseline_weights2 = 0
        else:
            baseline_weights2 = baseline_weights[index2 - 1, :]
        weights = numpy.sqrt((baseline_weights1**2 + baseline_weights2**2))
        weights[weights < 1e-4] = 0
        uu_weights[i, :] = weights

    sorted_indices = numpy.argsort(baseline_lengths)
    sorted_weights = uu_weights[sorted_indices, :][:, sorted_indices]

    if binned:
        if u_bins is None:
            u_bins = numpy.linspace(0, numpy.max(baseline_lengths), 101)
        bin_counter = numpy.zeros_like(uu_weights)
        bin_counter[uu_weights > 0] = 1

        uu1, uu2 = numpy.meshgrid(baseline_lengths, baseline_lengths)


        flattened_uu1 = uu1.flatten()
        flattened_uu2 = uu2.flatten()
        computed_weights = numpy.histogram2d(flattened_uu1, flattened_uu2, bins=u_bins,
                                         weights=uu_weights.flatten())
        computed_counts = numpy.histogram2d(flattened_uu1, flattened_uu2, bins=u_bins, normed=False)
    return u_bins, computed_weights[0], computed_counts[0]


def make_plot_uv_distribution(telescope, show_plot = True, save_plot = False, plot_folder = "./"):
    baseline_lengths = numpy.sqrt(telescope.baseline_table.u_coordinates**2 + telescope.baseline_table.v_coordinates**2)
    figure_u, axes_u = pyplot.subplots(1, 1)
    axes_u.hist(baseline_lengths, density=True, bins=100, label="MWA Phase II Compact")
    axes_u.set_xlabel(r"$u\,[\lambda]$")
    axes_u.set_ylabel("Baseline PDF")
    axes_u.legend()

    if save_plot:
        figure_u.savefig(plot_folder + "MWA_Phase_II_Baseline_PDF.pdf")
    if show_plot:
        pyplot.show()
    return


def make_plot_array_matrix(array_matrix, show_plot = True, save_plot = False, plot_folder = "./"):
    figure_amatrix = pyplot.figure(figsize=(250, 10))

    axes_amatrix = figure_amatrix.add_subplot(111)
    plot_amatrix = axes_amatrix.imshow(array_matrix.T, origin = 'lower')
    colorbar(plot_amatrix)
    axes_amatrix.set_xlabel("Baseline Number", fontsize = 20)
    axes_amatrix.set_ylabel("Antenna Number", fontsize = 20)
    if save_plot:
        figure_amatrix.savefig(plot_folder + "Array_Matrix_Double.pdf")
    if show_plot:
        pyplot.show()
    return


def redundant_matrix_constructor(telescope):
    redundant_table = redundant_baseline_finder(telescope.baseline_table)
    redundant_group_ids = numpy.unique(redundant_table.group_indices)
    antennas = numpy.unique([redundant_table.antenna_id1, redundant_table.antenna_id2] )
    x_positions, y_positions = redundant_positions(telescope, antennas)


    number_antennas = len(antennas)
    number_groups = len(redundant_group_ids)
    array_matrix = numpy.zeros((2 * redundant_table.number_of_baselines, 2 * number_antennas + 2*number_groups))

    for i in range(redundant_table.number_of_baselines):
        index1 = numpy.where(antennas == redundant_table.antenna_id1[i])[0]
        index2 = numpy.where(antennas == redundant_table.antenna_id2[i])[0]
        index3 = numpy.where(redundant_group_ids == redundant_table.group_indices[i])[0]
        # Fill in the real rows
        array_matrix[2 * i, 2 * index1] = 1
        array_matrix[2 * i, 2 * index2] = 1
        array_matrix[2 * i, 2*number_antennas + 2 * index3] = 1


        # Fill in the imaginary rows
        array_matrix[2 * i + 1, 2 * index1 + 1] = 1
        array_matrix[2 * i + 1, 2 * index2 + 1] = -1
        array_matrix[2 * i + 1, 2*number_antennas + 2 * index3 + 1] = 1

    # constraints = numpy.zeros((2,2 * number_antennas + 2*number_groups))
    # constraints[0, 1: 2* number_antennas +1:2] = x_positions
    # constraints[1, 1: 2 * number_antennas + 1:2] = y_positions
    #
    # constrained_matrix = numpy.vstack((array_matrix, constraints))
    constrained_matrix = array_matrix[:, 2:]
    return constrained_matrix


def sky_matrix_constructor(telescope):
    antennas = telescope.antenna_positions.antenna_ids
    baselines = telescope.baseline_table
    array_matrix = numpy.zeros((2 * baselines.number_of_baselines, 2 * len(antennas)))

    for i in range(baselines.number_of_baselines):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        # Fill in the real rows
        array_matrix[2 * i, 2 * index1] = 1
        array_matrix[2 * i, 2 * index2] = 1

        # Fill in the imaginary rows

        array_matrix[2 * i + 1, 2 * index1 + 1] = 1
        array_matrix[2 * i + 1, 2 * index2 + 1] = -1

    constrained_matrix = array_matrix[:, 2:]
    return constrained_matrix


def redundant_positions(telescope, antennas):
    antenna_table = telescope.antenna_positions
    x_positions = numpy.zeros(len(antennas))
    y_positions = x_positions.copy()

    for i in range(len(antennas)):
        index = numpy.where(antenna_table.antenna_ids == antennas[i])[0]
        x_positions[i] = antenna_table.x_coordinates[index]
        y_positions[i] = antenna_table.y_coordinates[index]
    return x_positions, y_positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()