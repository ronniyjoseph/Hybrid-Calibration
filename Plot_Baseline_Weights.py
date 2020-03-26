import sys
import numpy
import powerbox
from matplotlib import pyplot
from radiotelescope import RadioTelescope

from skymodel import SkyRealisation
from radiotelescope import ideal_gaussian_beam
from generaltools import from_lm_to_theta_phi
from generaltools import colorbar
import matplotlib.colors as colors

from scipy.signal import convolve2d

sys.path.append("../")

def main():
    path = "./Data/MWA_Compact_Coordinates.txt"
    plot_folder = "../../Plots/Analytic_Covariance/"
    plot_u_dist = False
    plot_array_matrix = False
    plot_inverse_matrix = False
    plot_weights = False
    grid_weights = True
    binned_weights = True
    telescope = RadioTelescope(load=True, path=path)
    baseline_lengths = numpy.sqrt(telescope.baseline_table.u_coordinates**2 + telescope.baseline_table.v_coordinates**2)

    if plot_u_dist:
        figure_u, axes_u = pyplot.subplots(1,1)
        axes_u.hist(baseline_lengths, density = True, bins = 100, label = "MWA Phase II Compact")
        axes_u.set_xlabel(r"$u\,[\lambda]$")
        axes_u.set_ylabel("Baseline PDF")
        axes_u.legend()
        figure_u.savefig(plot_folder + "MWA_Phase_II_Baseline_PDF.pdf")

    array_matrix = matrix_constructor_alternate(telescope)
    #
    # pyplot.rcParams['xtick.bottom'] = pyplot.rcParams['xtick.labelbottom'] = False
    # pyplot.rcParams['xtick.top'] = pyplot.rcParams['xtick.labeltop'] = True
    if plot_array_matrix:
        figure_amatrix = pyplot.figure(figsize=(250, 10))
        axes_amatrix = figure_amatrix.add_subplot(111)
        plot_amatrix = axes_amatrix.imshow(array_matrix.T, origin = 'lower')
        colorbar(plot_amatrix)
        axes_amatrix.set_xlabel("Baseline Number", fontsize = 20)
        axes_amatrix.set_ylabel("Antenna Number", fontsize = 20)
        figure_amatrix.savefig(plot_folder + "Array_Matrix_Double.pdf")


    inverse_array_matrix = numpy.linalg.pinv(array_matrix)
    if plot_inverse_matrix:
        figure_inverse = pyplot.figure(figsize = (110, 20))
        axes_inverse = figure_inverse.add_subplot(111)
        plot_inverse = axes_inverse.imshow(numpy.abs(inverse_array_matrix))
        colorbar(plot_inverse)

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
    return


def matrix_constructor_alternate(telescope):
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

def matrix_constructor_sep(telescope):
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

    return array_matrix


def matrix_constructor_sep_shift(telescope):
    antennas = telescope.antenna_positions.antenna_ids
    baselines = telescope.baseline_table
    array_matrix = numpy.zeros((2 * baselines.number_of_baselines, 2 * len(antennas)))

    for i in range(baselines.number_of_baselines):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        # Fill in the real rows
        array_matrix[i, index1] = 1
        array_matrix[i, index2] = 1

        # Fill in the imaginary rows

        array_matrix[baselines.number_of_baselines + i, len(antennas) + index1] = 1
        array_matrix[baselines.number_of_baselines + i, len(antennas) + index2] = -1

    return array_matrix


def matrix_constructor_double(telescope):
    antennas = telescope.antenna_positions.antenna_ids
    baselines = telescope.baseline_table
    array_matrix = numpy.zeros((baselines.number_of_baselines, 2 * len(antennas)))

    for i in range(baselines.number_of_baselines):
        index1 = numpy.where(antennas == baselines.antenna_id1[i])[0]
        index2 = numpy.where(antennas == baselines.antenna_id2[i])[0]

        # Fill in the real rows
        array_matrix[i, index1] = 1
        array_matrix[i, len(antennas) + index2] = 1

    return array_matrix


if __name__ == "__main__":
    main()