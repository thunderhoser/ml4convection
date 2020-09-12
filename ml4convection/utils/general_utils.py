"""General helper methods."""

import numpy
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import time_conversion

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
