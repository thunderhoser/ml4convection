"""Methods for setting up, training, and applying upconvolution networks."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking

DEFAULT_SMOOTHING_RADIUS_PX = 1.
DEFAULT_HALF_SMOOTHING_ROWS = 2
DEFAULT_HALF_SMOOTHING_COLUMNS = 2


def create_smoothing_filter(
        num_channels, smoothing_radius_px=DEFAULT_SMOOTHING_RADIUS_PX,
        num_half_filter_rows=DEFAULT_HALF_SMOOTHING_ROWS,
        num_half_filter_columns=DEFAULT_HALF_SMOOTHING_COLUMNS):
    """Creates convolution filter for Gaussian smoothing.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels (or "variables" or "features") to smooth.  Each
        channel will be smoothed independently.

    :param num_channels: C in the above discussion.
    :param smoothing_radius_px: e-folding radius (pixels).
    :param num_half_filter_rows: Number of rows in one half of filter.  Total
        number of rows will be 2 * `num_half_filter_rows` + 1.
    :param num_half_filter_columns: Same but for columns.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of convolution weights.
    """

    error_checking.assert_is_integer(num_channels)
    error_checking.assert_is_greater(num_channels, 0)
    error_checking.assert_is_greater(smoothing_radius_px, 0.)
    error_checking.assert_is_integer(num_half_filter_rows)
    error_checking.assert_is_greater(num_half_filter_rows, 0)
    error_checking.assert_is_integer(num_half_filter_columns)
    error_checking.assert_is_greater(num_half_filter_columns, 0)

    num_filter_rows = 2 * num_half_filter_rows + 1
    num_filter_columns = 2 * num_half_filter_columns + 1

    row_offsets_unique = numpy.linspace(
        -num_half_filter_rows, num_half_filter_rows, num=num_filter_rows,
        dtype=float)
    column_offsets_unique = numpy.linspace(
        -num_half_filter_columns, num_half_filter_columns,
        num=num_filter_columns, dtype=float)

    column_offset_matrix, row_offset_matrix = numpy.meshgrid(
        column_offsets_unique, row_offsets_unique)

    pixel_offset_matrix = numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2)

    small_weight_matrix = numpy.exp(
        -pixel_offset_matrix ** 2 / (2 * smoothing_radius_px ** 2)
    )
    small_weight_matrix = small_weight_matrix / numpy.sum(small_weight_matrix)

    weight_matrix = numpy.zeros(
        (num_filter_rows, num_filter_columns, num_channels, num_channels)
    )

    for k in range(num_channels):
        weight_matrix[..., k, k] = small_weight_matrix

    return weight_matrix
