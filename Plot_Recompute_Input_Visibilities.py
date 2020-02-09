import argparse
import numpy


def main():
    data_type = "noise_visibilities"
    path = "/data/rjoseph/Hybrid_Calibration/numerical_simulations/Initial_Testing2_Gain_2_Two/"
    input_parameters = numpy.loadtxt(path + "input_parameters.txt")
    n_realisations = 1000 #input_parameters[-1]
    ideal_visibilities = load_data(path, 'ideal_visibilities', n_realisations)
    noise_visibilities = load_data(path, 'noise_visibilities', n_realisations)
    antenna_index = 2

    amplitudes = numpy.abs(data.flatten())
    phases = numpy.angle(data.flatten())

    real = numpy.real(data.flatten())
    imaginary = numpy.imag(data.flatten())

    figure, axes = pyplot.subplots(2, 2, figsize= (10,10))
    axes[0, 0].hist(amplitudes, bins=100)
    axes[0, 1].hist(phases, bins=100)

    axes[1, 0].hist(real, bins=100)
    axes[1, 1].hist(imaginary, bins=100)

    # axes[0, 1].set_xscale('log')

    # axes[0, 0].set_yscale('log')
    # axes[0, 1].set_yscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 1].set_yscale('log')

    axes[0, 0].set_xlabel(r'$|V|$')
    axes[0, 1].set_xlabel(r'$ \mathrm{arg}\left(V\right)\,  [rad] $')
    axes[1, 0].set_xlabel(r'$ \mathcal{Re}(V)$')
    axes[1, 1].set_xlabel(r'$ \mathcal{Im}(V)$')

    axes[0, 0].set_ylabel(r'Number of solutions')
    axes[1, 0].set_ylabel(r'Number of solutions')

    figure.suptitle(data_type)
    pyplot.show()
    return


def load_data(path, data_type, n_realisations):
    for i in range(n_realisations):
        folder = f"realisation_{i}/"
        file = data_type + ".npy"
        if i == 0:
            data = numpy.load(path + folder + file)
        elif i == -1:
            new_realisation = numpy.load(path + folder + file)
            data = numpy.append(data, new_realisation, axis=-1)
        else:
            new_realisation = numpy.load(path + folder + file)
            data = numpy.append(data, new_realisation, axis=1)

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