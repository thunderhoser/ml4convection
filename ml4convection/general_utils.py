"""General helper methods."""

import os
import sys
import numpy
from scipy.ndimage import distance_transform_edt

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import time_conversion

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400


def get_previous_date(date_string):
    """Returns previous date.

    :param date_string: Date (format "yyyymmdd").
    :return: prev_date_string: Previous date (format "yyyymmdd").
    """

    unix_time_sec = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)
    return time_conversion.unix_sec_to_string(
        unix_time_sec - DAYS_TO_SECONDS, DATE_FORMAT
    )


def get_next_date(date_string):
    """Returns next date.

    :param date_string: Date (format "yyyymmdd").
    :return: next_date_string: Next date (format "yyyymmdd").
    """

    unix_time_sec = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)
    return time_conversion.unix_sec_to_string(
        unix_time_sec + DAYS_TO_SECONDS, DATE_FORMAT
    )


def create_mean_filter(half_num_rows, half_num_columns, num_channels):
    """Creates convolutional filter that computes mean.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels

    :param half_num_rows: Number of rows on either side of center.  This is
        (M - 1) / 2.
    :param half_num_columns: Number of columns on either side of center.  This
        is (N - 1) / 2.
    :param num_channels: Number of channels.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of filter weights.
    """

    error_checking.assert_is_integer(half_num_rows)
    error_checking.assert_is_geq(half_num_rows, 0)
    error_checking.assert_is_integer(half_num_columns)
    error_checking.assert_is_geq(half_num_columns, 0)
    error_checking.assert_is_integer(num_channels)
    error_checking.assert_is_greater(num_channels, 0)

    num_rows = 2 * half_num_rows + 1
    num_columns = 2 * half_num_columns + 1
    weight = 1. / (num_rows * num_columns)

    return numpy.full(
        (num_rows, num_columns, num_channels, num_channels), weight
    )


def fill_nans(data_matrix):
    """Fills NaN's with nearest neighbours.

    This method is adapted from the method `fill`, which you can find here:
    https://stackoverflow.com/posts/9262129/revisions

    :param data_matrix: numpy array of real-valued data.
    :return: data_matrix: Same but without NaN's.
    """

    error_checking.assert_is_real_numpy_array(data_matrix)

    indices = distance_transform_edt(
        numpy.isnan(data_matrix), return_distances=False, return_indices=True
    )
    return data_matrix[tuple(indices)]
