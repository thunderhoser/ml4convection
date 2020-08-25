"""Methods for time conversion."""

import sys
import os.path
import time
import calendar

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def string_to_unix_sec(time_string, time_directive):
    """Converts time from string to Unix format.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param time_string: Time string.
    :param time_directive: Format of time string (examples: "%Y%m%d" if string
        is "yyyymmdd", "%Y-%m-%d-%H%M%S" if string is "yyyy-mm-dd-HHMMSS",
        etc.).
    :return: unix_time_sec: Time in Unix format.
    """

    error_checking.assert_is_string(time_string)
    error_checking.assert_is_string(time_directive)
    return calendar.timegm(time.strptime(time_string, time_directive))


def unix_sec_to_string(unix_time_sec, time_directive):
    """Converts time from Unix format to string.

    Unix format = seconds since 0000 UTC 1 Jan 1970.

    :param unix_time_sec: Time in Unix format.
    :param time_directive: Format of time string (examples: "%Y%m%d" if string
        is "yyyymmdd", "%Y-%m-%d-%H%M%S" if string is "yyyy-mm-dd-HHMMSS",
        etc.).
    :return: time_string: Time string.
    """

    error_checking.assert_is_integer(unix_time_sec)
    error_checking.assert_is_string(time_directive)
    return time.strftime(time_directive, time.gmtime(unix_time_sec))


def first_and_last_times_in_year(year):
    """Returns first and last times in year (discretized in seconds).

    For example, first/last times in 2017 are 2017-01-01-000000 and
    2017-12-31-235959.

    :param year: Integer.
    :return: start_time_unix_sec: First time in year.
    :return: end_time_unix_sec: Last time in year.
    """

    error_checking.assert_is_integer(year)

    time_format = '%Y-%m-%d-%H%M%S'
    start_time_string = '{0:d}-01-01-000000'.format(year)
    end_time_string = '{0:d}-12-31-235959'.format(year)

    return (
        string_to_unix_sec(start_time_string, time_format),
        string_to_unix_sec(end_time_string, time_format)
    )
