"""Converts satellite data to daily NetCDF files."""

import os
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import twb_satellite_io
from ml4convection.io import satellite_io

# TODO(thunderhoser): Convert counts to brightness temperatures.

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

BAND_NUMBERS = satellite_io.BAND_NUMBERS

INPUT_DIR_ARG_NAME = 'input_satellite_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
ALLOW_MISSING_DAYS_ARG_NAME = 'allow_missing_days'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_satellite_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing raw data from Taiwanese Weather Bureau'
    '.  Files therein will be found by `twb_satellite_io.find_file` and read by'
    ' `twb_satellite_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Satellite data will be processed for all days '
    'in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

ALLOW_MISSING_DAYS_HELP_STRING = (
    'Boolean flag.  If 1, will gracefully skip days with no data.  If 0, will '
    'throw an error if this happens.'
)
TEMPORARY_DIR_HELP_STRING = (
    'Name of temporary directory.  Intermediate text files (between input '
    'binary files and output NetCDF files) will be stored here.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Daily NetCDF files will be written by '
    '`satellite_io.write_file`, to locations therein determined by '
    '`satellite_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ALLOW_MISSING_DAYS_ARG_NAME, type=int, required=False, default=0,
    help=ALLOW_MISSING_DAYS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _process_satellite_data_one_time(
        input_dir_name, valid_time_unix_sec, temporary_dir_name,
        output_file_name, append):
    """Processes satellite data for one time step.

    :param input_dir_name: See documentation at top of file.
    :param valid_time_unix_sec: Valid time.
    :param temporary_dir_name: See documentation at top of file.
    :param output_file_name: Path to output file.
    :param append: See documentation for `satellite_io.write_file`.
    :return: success: Boolean flag (True if this method wrote to the output
        file).
    """

    brightness_count_matrix = None
    latitudes_deg_n = None
    longitudes_deg_e = None

    num_bands = len(BAND_NUMBERS)

    for j in range(num_bands):
        this_file_name = twb_satellite_io.find_file(
            top_directory_name=input_dir_name,
            valid_time_unix_sec=valid_time_unix_sec,
            band_number=BAND_NUMBERS[j]
        )

        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_count_matrix, these_latitudes_deg_n, these_longitudes_deg_e = (
            twb_satellite_io.read_file(
                binary_file_name=this_file_name, return_brightness_temps=False,
                raise_fortran_errors=False,
                temporary_dir_name=temporary_dir_name
            )
        )

        if this_count_matrix is None:
            return False

        if j == 0:
            num_grid_rows = len(these_latitudes_deg_n)
            num_grid_columns = len(these_longitudes_deg_e)
            brightness_count_matrix = numpy.full(
                (num_grid_rows, num_grid_columns, num_bands), numpy.nan
            )

            latitudes_deg_n = these_latitudes_deg_n + 0.
            longitudes_deg_e = these_longitudes_deg_e + 0.
        else:
            assert numpy.allclose(
                latitudes_deg_n, these_latitudes_deg_n, atol=TOLERANCE
            )
            assert numpy.allclose(
                longitudes_deg_e, these_longitudes_deg_e, atol=TOLERANCE
            )

        brightness_count_matrix[..., j] = this_count_matrix

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    satellite_io.write_file(
        netcdf_file_name=output_file_name,
        latitudes_deg_n=latitudes_deg_n,
        longitudes_deg_e=longitudes_deg_e, band_numbers=BAND_NUMBERS,
        valid_time_unix_sec=valid_time_unix_sec, append=append,
        brightness_count_matrix=brightness_count_matrix
    )

    return True


def _process_satellite_data_one_day(
        input_dir_name, date_string, allow_missing_days, temporary_dir_name,
        output_file_name):
    """Processes satellite data for one day.

    :param input_dir_name: See documentation at top of file.
    :param date_string: Will process for this day (format "yyyymmdd").
    :param allow_missing_days: See documentation at top of file.
    :param temporary_dir_name: Same.
    :param output_file_name: Path to output file.
    """

    first_time_unix_sec = (
        time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)
    )
    last_time_unix_sec = (
        first_time_unix_sec +
        DAYS_TO_SECONDS - twb_satellite_io.TIME_INTERVAL_SEC
    )

    num_bands = len(BAND_NUMBERS)
    input_file_names_by_band = [[]] * num_bands
    valid_times_by_band_unix_sec = [numpy.array([], dtype=int)] * num_bands

    for k in range(num_bands):
        input_file_names_by_band[k] = twb_satellite_io.find_many_files(
            top_directory_name=input_dir_name,
            first_time_unix_sec=first_time_unix_sec,
            last_time_unix_sec=last_time_unix_sec,
            band_number=BAND_NUMBERS[k],
            raise_error_if_all_missing=not allow_missing_days
        )

        valid_times_by_band_unix_sec[k] = numpy.array([
            twb_satellite_io.file_name_to_time(f)
            for f in input_file_names_by_band[k]
        ], dtype=int)

    valid_times_unix_sec = set.intersection(*[
        set(t.tolist()) for t in valid_times_by_band_unix_sec
    ])
    valid_times_unix_sec = numpy.array(list(valid_times_unix_sec), dtype=int)
    valid_times_unix_sec = numpy.sort(valid_times_unix_sec)

    num_times = len(valid_times_unix_sec)
    append = False

    for i in range(num_times):
        success = _process_satellite_data_one_time(
            input_dir_name=input_dir_name,
            valid_time_unix_sec=valid_times_unix_sec[i],
            temporary_dir_name=temporary_dir_name,
            output_file_name=output_file_name, append=append
        )

        append = append or success

        if i != num_times - 1:
            print('\n')


def _run(input_dir_name, first_date_string, last_date_string,
         allow_missing_days, temporary_dir_name, output_dir_name):
    """Converts satellite data to daily NetCDF files.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param allow_missing_days: Same.
    :param temporary_dir_name: Same.
    :param output_dir_name: Same.
    """

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    for i in range(len(date_strings)):
        this_netcdf_file_name = satellite_io.find_file(
            top_directory_name=output_dir_name,
            valid_date_string=date_strings[i],
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        _process_satellite_data_one_day(
            input_dir_name=input_dir_name, date_string=date_strings[i],
            allow_missing_days=allow_missing_days,
            temporary_dir_name=temporary_dir_name,
            output_file_name=this_netcdf_file_name
        )

        satellite_io.compress_file(this_netcdf_file_name)
        os.remove(this_netcdf_file_name)

        if i != len(date_strings) - 1:
            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        allow_missing_days=bool(getattr(
            INPUT_ARG_OBJECT, ALLOW_MISSING_DAYS_ARG_NAME
        )),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
