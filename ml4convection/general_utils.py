"""General helper methods."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def downsample_in_space(data_matrix, x_coordinates, y_coordinates,
                        downsampling_factor):
    """Downsamples data in space (i.e., coarsens spatial resolution).

    M = number of rows in original grid
    N = number of columns in original grid
    m = number of rows in downsampled grid
    n = number of columns in downsampled grid

    :param data_matrix: M-by-N numpy array of data values.
    :param x_coordinates: length-N numpy array of x-coordinates (must be
        monotonically increasing).
    :param y_coordinates: length-M numpy array of y-coordinates (must be
        onotonically increasing).
    :param downsampling_factor: Downsampling factor (integer).
    :return: data_matrix: m-by-n numpy array of data values.
    :return: x_coordinates: length-n numpy array of x-coordinates (monotonically
        increasing).
    :return: y_coordinates: length-m numpy array of y-coordinates (monotonically
        increasing).
    """

    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    error_checking.assert_is_numpy_array(data_matrix, num_dimensions=2)

    num_rows = data_matrix.shape[0]
    num_columns = data_matrix.shape[1]

    error_checking.assert_is_numpy_array(
        x_coordinates,
        exact_dimensions=numpy.array([num_columns], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(numpy.diff(x_coordinates), 0.)

    error_checking.assert_is_numpy_array(
        y_coordinates,
        exact_dimensions=numpy.array([num_rows], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(numpy.diff(y_coordinates), 0.)

    error_checking.assert_is_integer(downsampling_factor)
    error_checking.assert_is_geq(downsampling_factor, 2)

    x_coordinates = x_coordinates[::downsampling_factor]
    y_coordinates = y_coordinates[::downsampling_factor]
    data_matrix = data_matrix[::downsampling_factor, ::downsampling_factor]

    return data_matrix, x_coordinates, y_coordinates
