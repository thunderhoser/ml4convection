"""General helper methods."""

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
