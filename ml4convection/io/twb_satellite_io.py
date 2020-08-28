"""IO methods for satellite files from Taiwanese Weather Bureau (TWB)."""

import os
import tempfile
import numpy
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))

TIME_INTERVAL_SEC = 600

ERROR_STRING = (
    '\nUnix command failed (log messages shown above should explain why).'
)

DEFAULT_GFORTRAN_COMPILER_NAME = 'gfortran'
FORTRAN_SCRIPT_NAME = (
    '{0:s}/read_twb_satellite_file.f90'.format(THIS_DIRECTORY_NAME)
)
FORTRAN_EXE_NAME = (
    '{0:s}/read_twb_satellite_file.exe'.format(THIS_DIRECTORY_NAME)
)

TIME_FORMAT_IN_MESSAGES = '%Y-%m-%d-%H%M%S'
TIME_FORMAT_IN_DIR_NAMES = '%Y-%m'
TIME_FORMAT_IN_FILE_NAMES = '%Y-%m-%d_%H%M'

LATITUDE_COLUMN_INDEX = 0
LONGITUDE_COLUMN_INDEX = 1
BRIGHTNESS_COUNT_INDEX = 2
LATLNG_PRECISION_DEG = 1e-6

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

    satellite_file_name = '{0:s}/{1:s}/{2:s}.B{3:02d}.GSD.Cnt'.format(
        top_directory_name,
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_DIR_NAMES
        ),
        time_conversion.unix_sec_to_string(
            valid_time_unix_sec, TIME_FORMAT_IN_FILE_NAMES
        ),
        band_number
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


def read_file(
        binary_file_name, return_brightness_temps=True,
        gfortran_compiler_name=DEFAULT_GFORTRAN_COMPILER_NAME,
        temporary_dir_name=None):
    """Reads satellite data (brightness temperatures) from binary file.

    M = number of rows in grid
    N = number of columns in grid

    :param binary_file_name: Path to input file.
    :param return_brightness_temps: Boolean flag.  If True (False), will return
        brightness temperatures (raw counts).
    :param gfortran_compiler_name: Path to gfortran compiler.
    :param temporary_dir_name: Name of temporary directory for text file, which
        will be deleted as soon as it is read.  If None, temporary directory
        will be set to default.
    :return: data_matrix: M-by-N matrix of data values.  If
        `return_brightness_temps == True`, these are brightness temperatures in
        Kelvins.  Otherwise, these are raw counts.
    :return: latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :return: longitudes_deg_e: length-N numpy array of longitudes (deg E).
    """

    error_checking.assert_file_exists(binary_file_name)
    error_checking.assert_is_boolean(return_brightness_temps)
    # error_checking.assert_file_exists(gfortran_compiler_name)

    if not os.path.isfile(FORTRAN_EXE_NAME):
        command_string = '"{0:s}" "{1:s}" -o "{2:s}"'.format(
            gfortran_compiler_name, FORTRAN_SCRIPT_NAME, FORTRAN_EXE_NAME
        )

        exit_code = os.system(command_string)
        if exit_code != 0:
            raise ValueError(ERROR_STRING)

    if temporary_dir_name is not None:
        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=temporary_dir_name
        )

    temporary_text_file_name = tempfile.NamedTemporaryFile(
        dir=temporary_dir_name, delete=False
    ).name

    print('Reading data from binary file: "{0:s}"...'.format(binary_file_name))
    fortran_exe_dir_name, fortran_exe_pathless_name = (
        os.path.split(FORTRAN_EXE_NAME)
    )
    command_string = 'cd "{0:s}"; ./{1:s} "{2:s}" > "{3:s}"'.format(
        fortran_exe_dir_name, fortran_exe_pathless_name, binary_file_name,
        temporary_text_file_name
    )

    exit_code = os.system(command_string)
    if exit_code != 0:
        raise ValueError(ERROR_STRING)

    print('Reading data from temporary text file: "{0:s}"...'.format(
        temporary_text_file_name
    ))
    data_matrix = numpy.loadtxt(temporary_text_file_name)
    os.remove(temporary_text_file_name)

    all_latitudes_deg_n = number_rounding.round_to_nearest(
        data_matrix[:, LATITUDE_COLUMN_INDEX], LATLNG_PRECISION_DEG
    )
    all_longitudes_deg_e = number_rounding.round_to_nearest(
        data_matrix[:, LONGITUDE_COLUMN_INDEX], LATLNG_PRECISION_DEG
    )
    num_grid_rows = numpy.sum(all_longitudes_deg_e == all_longitudes_deg_e[0])
    num_grid_columns = numpy.sum(all_latitudes_deg_n == all_latitudes_deg_n[0])

    if return_brightness_temps:
        brightness_counts = (
            numpy.round(data_matrix[:, BRIGHTNESS_COUNT_INDEX]).astype(int)
        )
        brightness_counts[brightness_counts < MIN_BRIGHTNESS_COUNT] = (
            MISSING_COUNT
        )
        brightness_counts[brightness_counts > MAX_BRIGHTNESS_COUNT] = (
            MISSING_COUNT
        )

        count_to_temperature_dict_kelvins = _read_lookup_table(
            file_name_to_band_number(binary_file_name)
        )
        data_values = numpy.array([
            count_to_temperature_dict_kelvins[n] for n in brightness_counts
        ])
    else:
        brightness_counts = data_matrix[:, BRIGHTNESS_COUNT_INDEX]
        brightness_counts[brightness_counts < MIN_BRIGHTNESS_COUNT] = numpy.nan
        brightness_counts[brightness_counts > MAX_BRIGHTNESS_COUNT] = numpy.nan
        data_values = brightness_counts

    data_matrix = numpy.reshape(data_values, (num_grid_rows, num_grid_columns))
    data_matrix = numpy.flipud(data_matrix)

    latitudes_deg_n = numpy.reshape(
        all_latitudes_deg_n, (num_grid_rows, num_grid_columns)
    )[:, 0]
    latitudes_deg_n = latitudes_deg_n[::-1]

    longitudes_deg_e = numpy.reshape(
        all_longitudes_deg_e, (num_grid_rows, num_grid_columns)
    )[0, :]

    return data_matrix, latitudes_deg_n, longitudes_deg_e
