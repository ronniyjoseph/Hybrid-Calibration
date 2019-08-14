import numpy


def split_visibility(data):
    data_real = numpy.real(data)
    data_imag = numpy.imag(data)

    data_split = numpy.hstack((data_real, data_imag)).reshape((1, 2 * len(data_real)), order="C")
    return data_split[0, :]
