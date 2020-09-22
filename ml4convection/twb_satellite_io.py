"""IO methods for satellite files from Taiwanese Weather Bureau (TWB)."""

import os
import sys
import warnings
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import error_checking

TIME_INTERVAL_SEC = 600

GRID_LATITUDES_DEG_N = numpy.linspace(18, 29, num=881, dtype=float)
GRID_LONGITUDES_DEG_E = numpy.linspace(115, 126.5, num=921)

TIME_FORMAT_IN_MESSAGES = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_IN_DIR_NAMES = '%Y-%m'
TIME_FORMAT_IN_FILE_NAMES = '%Y-%m-%d_%H%M'

BAND_NUMBER_TO_FILE_NAME_PART = {
    8: 'GSD',
    9: 'GDS',
    10: 'GDS',
    11: 'GDS',
    13: 'GSD',
    14: 'GDS',
    16: 'GSD'
}

MIN_BRIGHTNESS_COUNT = 0
MAX_BRIGHTNESS_COUNT = 255
MISSING_COUNT = -1


def _find_lookup_table(band_number):
    """Finds lookup table for converting brightness count to temperature.

    :param band_number: Number of spectral band (integer).
    :return: lookup_table_file_name: Path to text file with lookup table.
    :raises: ValueError: if file is not found.
    """

    lookup_table_file_name = '{0:s}/Channel_B{1:02d}_Standary.LUT'.format(
        THIS_DIRECTORY_NAME, band_number
    )

    if os.path.isfile(lookup_table_file_name):
        return lookup_table_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        lookup_table_file_name
    )
    raise ValueError(error_string)


def _read_lookup_table(band_number):
    """Reads lookup table for converting brightness count to temperature.

    :param band_number: Number of spectral band (integer).
    :return: count_to_temperature_dict_kelvins: Dictionary, where keys are
        brightness counts (integers) and values are brightness temperatures
        (Kelvins).
    """

    lookup_table_file_name = _find_lookup_table(band_number)
    data_matrix = numpy.loadtxt(lookup_table_file_name)

    counts = numpy.round(data_matrix[:, 0]).astype(int)
    temperatures_kelvins = data_matrix[:, 1]

    count_to_temperature_dict_kelvins = dict(zip(
        counts, temperatures_kelvins
    ))
    count_to_temperature_dict_kelvins[MISSING_COUNT] = numpy.nan

    return count_to_temperature_dict_kelvins


def _count_to_temperature(brightness_counts, band_number):
    """Converts raw counts to brightness temperatures.

    :param brightness_counts: numpy array of counts.
    :param band_number: Number of spectral band (integer).
    :return: brightness_temps_kelvins: numpy array of brightness temperatures
        (same shape as `brightness_counts`).
    """

    # error_checking.assert_is_integer_numpy_array(brightness_counts)
    error_checking.assert_is_integer(band_number)
    error_checking.assert_is_geq(band_number, 0)

    brightness_counts_1d = numpy.ravel(brightness_counts)
    brightness_counts_1d[numpy.isnan(brightness_counts_1d)] = MISSING_COUNT
    brightness_counts_1d[brightness_counts_1d < MIN_BRIGHTNESS_COUNT] = (
        MISSING_COUNT
    )
    brightness_counts_1d[brightness_counts_1d > MAX_BRIGHTNESS_COUNT] = (
        MISSING_COUNT
    )
    brightness_counts_1d = numpy.round(brightness_counts_1d).astype(int)

    count_to_temperature_dict_kelvins = _read_lookup_table(band_number)
    brightness_temps_kelvins = numpy.array([
        count_to_temperature_dict_kelvins[c] for c in brightness_counts_1d
    ])

    return numpy.reshape(brightness_temps_kelvins, brightness_counts.shape)


def find_file(
        top_directory_name, valid_time_unix_sec, band_number,
        raise_error_if_missing=True):
    """Finds binary file with satellite data.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_time_unix_sec: Valid time.
    :param band_number: Number of spectral band (integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: satellite_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_integer(band_number)
    error_checking.assert_is_greater(band_number, 0)
    error_checking.assert_is_boolean(raise_error_if_missing)

    satellite_file_name = '{0:s}/{1:s}/{2:s}.B{3:02d}.{4:s}.Cnt'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_DIR_NAMES
        ),
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES
        ),
        band_number,
        BAND_NUMBER_TO_FILE_NAME_PART[band_number]
    )

    if os.path.isfile(satellite_file_name) or not raise_error_if_missing:
        return satellite_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        satellite_file_name
    )
    raise ValueError(error_string)


def file_name_to_time(satellite_file_name):
    """Parses valid time from file name.

    :param satellite_file_name: Path to satellite file (see `find_file` for
        naming convention).
    :return: valid_time_unix_sec: Valid time.
    """

    error_checking.assert_is_string(satellite_file_name)
    pathless_file_name = os.path.split(satellite_file_name)[-1]
    valid_time_string = pathless_file_name.split('.')[0]

    return time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT_IN_FILE_NAMES
    )


def file_name_to_band_number(satellite_file_name):
    """Parses band number from file name.

    :param satellite_file_name: Path to satellite file (see `find_file` for
        naming convention).
    :return: band_number: Number of spectral band (integer).
    """

    error_checking.assert_is_string(satellite_file_name)
    pathless_file_name = os.path.split(satellite_file_name)[-1]
    band_string = pathless_file_name.split('.')[1]

    if band_string.startswith('B'):
        band_string = band_string[1:]

    return int(band_string)


def find_many_files(
        top_directory_name, first_time_unix_sec, last_time_unix_sec,
        band_number, raise_error_if_all_missing=True,
        raise_error_if_any_missing=False, test_mode=False):
    """Finds many binary files with satellite data.

    :param top_directory_name: See doc for `find_file`.
    :param first_time_unix_sec: First valid time.
    :param last_time_unix_sec: Last valid time.
    :param band_number: See doc for `find_file`.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: satellite_file_names: 1-D list of paths to satellite files.  This
        list does *not* contain expected paths to non-existent files.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )

    satellite_file_names = []

    for this_time_unix_sec in valid_times_unix_sec:
        this_file_name = find_file(
            top_directory_name=top_directory_name,
            valid_time_unix_sec=this_time_unix_sec, band_number=band_number,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            satellite_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(satellite_file_names) == 0:
        first_time_string = time_conversion.unix_sec_to_string(
            first_time_unix_sec, TIME_FORMAT_IN_MESSAGES
        )
        last_time_string = time_conversion.unix_sec_to_string(
            last_time_unix_sec, TIME_FORMAT_IN_MESSAGES
        )

        error_string = (
            'Cannot find any file in directory "{0:s}" from times {1:s} to '
            '{2:s}.'
        ).format(
            top_directory_name, first_time_string, last_time_string
        )
        raise ValueError(error_string)

    return satellite_file_names


def read_file(binary_file_name):
    """Reads satellite data (brightness temperatures) from binary file.

    M = number of rows in grid
    N = number of columns in grid

    :param binary_file_name: Path to input file.
    :return: brightness_temp_matrix_kelvins: M-by-N matrix of brightness
        temperatures.
    :return: latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :return: longitudes_deg_e: length-N numpy array of longitudes (deg E).
    """

    brightness_counts = numpy.fromfile(binary_file_name, dtype=numpy.uint8)

    brightness_temp_matrix_kelvins = _count_to_temperature(
        brightness_counts=brightness_counts,
        band_number=file_name_to_band_number(binary_file_name)
    )

    num_rows = len(GRID_LATITUDES_DEG_N)
    num_columns = len(GRID_LONGITUDES_DEG_E)

    try:
        brightness_temp_matrix_kelvins = numpy.reshape(
            brightness_temp_matrix_kelvins, (num_rows, num_columns)
        )
    except ValueError:
        warning_string = (
            'File "{0:s}" appears to be corrupt.  Expected {1:d} grid cells, '
            'found {2:d}.'
        ).format(
            binary_file_name,
            num_rows * num_columns,
            brightness_temp_matrix_kelvins.size
        )

        warnings.warn(warning_string)
        return None, None, None

    brightness_temp_matrix_kelvins = numpy.flipud(
        brightness_temp_matrix_kelvins
    )

    return (
        brightness_temp_matrix_kelvins,
        GRID_LATITUDES_DEG_N, GRID_LONGITUDES_DEG_E
    )
