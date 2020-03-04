import argparse
import numpy


def main():
    path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/Square_Large_Array_100Jy_Noise_No_Models/"
    input_parameters = numpy.loadtxt(path + "input_parameters.txt")
    n_realisations = 630 #input_parameters[-1]
    data = load_data(path, n_realisations)
    antenna_index = 2

    all_gain_amplitudes = numpy.abs(data.flatten())
    all_gain_phases = numpy.angle(data.flatten())

    gain_2_amplitudes = numpy.abs(data[antenna_index, :])
    gain_2_phases = numpy.angle(data[antenna_index, :])

    # all_gain_amplitudes = numpy.std(numpy.abs(data), axis=0)
    # all_gain_phases = numpy.std(numpy.angle(data), axis = 0)

    figure, axes = pyplot.subplots(2, 2, figsize=(10, 10))

    axes[0, 1].hist([all_gain_amplitudes, gain_2_amplitudes], bins=50)
    axes[0, 0].hist([all_gain_amplitudes[all_gain_amplitudes < 2.5], gain_2_amplitudes[gain_2_amplitudes < 2.5]], bins=50)

    axes[1, 0].hist([all_gain_phases, gain_2_phases], bins=50)
    axes[1, 1].axis("off")

    # axes[0, 0].axvline(x=2, color = 'k')

    # axes[0, 1].set_xscale('log')

    axes[0, 0].set_yscale('log')
    axes[0, 1].set_yscale('log')
    axes[1, 0].set_yscale('log')

    axes[0, 0].set_xlabel(r'$|g|$')
    axes[0, 1].set_xlabel(r'$|g|$')
    axes[1, 0].set_xlabel(r'$ \mathrm{arg}\left(g\right)\,  [rad] $')

    axes[0, 0].set_ylabel(r'Number of solutions')
    axes[1, 0].set_ylabel(r'Number of solutions')

    figure.suptitle(r'Corrcal with $g_{2} = 1$')
    figure.savefig(path + "Solutions_hist.pdf")
    pyplot.show()
    return


def load_data(path, n_realisations):
    for i in range(n_realisations):
        folder = f"realisation_{i}/"
        file = "gain_solutions.npy"
        if i == 0:
            data = numpy.load(path + folder + file)
        elif i == 1:
            new_realisation = numpy.load(path + folder + file)
            data = numpy.stack((data, new_realisation), axis=-1)
        else:
            new_realisation = numpy.load(path + folder + file)
            data = numpy.append(data, new_realisation[:, numpy.newaxis], axis=1)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh", action="store_true", dest="ssh_key", default=False)
    params = parser.parse_args()

    import matplotlib

    if params.ssh_key:
        matplotlib.use("Agg")
    from matplotlib import pyplot

    main()