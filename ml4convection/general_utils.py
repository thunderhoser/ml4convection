"""General helper methods."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def downsample_in_space(
        data_matrix, x_axis_index, y_axis_index, x_coordinates, y_coordinates,
        downsampling_factor):
    """Downsamples data in space (i.e., coarsens spatial resolution).

    M = number of rows in original grid
    N = number of columns in original grid
    m = number of rows in downsampled grid
    n = number of columns in downsampled grid

    :param data_matrix: numpy array of data values.
    :param x_axis_index: Axis index (in `data_matrix`) for x-coordinates.
    :param y_axis_index: Axis index (in `data_matrix`) for y-coordinates.
    :param x_coordinates: length-N numpy array of x-coordinates (must be
        monotonically increasing).
    :param y_coordinates: length-M numpy array of y-coordinates (must be
        onotonically increasing).
    :param downsampling_factor: Downsampling factor (integer).
    :return: data_matrix: numpy array of data values.
    :return: x_coordinates: length-n numpy array of x-coordinates (monotonically
        increasing).
    :return: y_coordinates: length-m numpy array of y-coordinates (monotonically
        increasing).
    """

    error_checking.assert_is_numpy_array_without_nan(data_matrix)
    num_dimensions = len(data_matrix.shape)
    error_checking.assert_is_geq(num_dimensions, 2)

    error_checking.assert_is_integer(x_axis_index)
    error_checking.assert_is_geq(x_axis_index, 0)
    error_checking.assert_is_less_than(x_axis_index, num_dimensions)

    error_checking.assert_is_integer(y_axis_index)
    error_checking.assert_is_geq(y_axis_index, 0)
    error_checking.assert_is_less_than(y_axis_index, num_dimensions)

    num_rows = data_matrix.shape[y_axis_index]
    num_columns = data_matrix.shape[x_axis_index]

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

    row_indices = numpy.linspace(0, num_rows - 1, num=num_rows, dtype=int)
    column_indices = numpy.linspace(
        0, num_columns - 1, num=num_columns, dtype=int
    )

    data_matrix = numpy.take(
        data_matrix, indices=row_indices[::downsampling_factor],
        axis=y_axis_index
    )
    data_matrix = numpy.take(
        data_matrix, indices=column_indices[::downsampling_factor],
        axis=x_axis_index
    )

    return data_matrix, x_coordinates, y_coordinates
