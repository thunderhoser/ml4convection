"""Converts radar data to daily NetCDF files."""

import sys
import os.path
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import twb_radar_io
import radar_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

INPUT_DIR_ARG_NAME = 'input_radar_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
ALLOW_MISSING_DAYS_ARG_NAME = 'allow_missing_days'
WITH_3D_ARG_NAME = 'with_3d'
OUTPUT_DIR_ARG_NAME = 'output_radar_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing raw data from Taiwanese Weather Bureau'
    '.  Files therein will be found by `twb_radar_io.find_file` and read by '
    '`twb_radar_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Radar data will be processed for all days in '
    'the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

ALLOW_MISSING_DAYS_HELP_STRING = (
    'Boolean flag.  If 1, will gracefully skip days with no data.  If 0, will '
    'throw an error if this happens.'
)
WITH_3D_HELP_STRING = 'Boolean flag.  If 1 (0), will process 3-D (2-D) data.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Daily NetCDF files will be written by '
    '`radar_io.write_2d_file` or `radar_io.write_3d_file`, to locations therein'
    ' determined by `radar_io.find_file`.'
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
    '--' + WITH_3D_ARG_NAME, type=int, required=False, default=0,
    help=WITH_3D_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _process_radar_data_one_day(input_dir_name, date_string, allow_missing_days,
                                output_dir_name):
    """Processes radar data for one day.

    :param input_dir_name: See documentation at top of file.
    :param date_string: Will process for this day (format "yyyymmdd").
    :param allow_missing_days: See documentation at top of file.
    :param output_dir_name: Same.
    """

    first_time_unix_sec = (
        time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)
    )
    last_time_unix_sec = (
        first_time_unix_sec + DAYS_TO_SECONDS - twb_radar_io.TIME_INTERVAL_SEC
    )
    input_file_names = twb_radar_io.find_many_files(
        top_directory_name=input_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec, with_3d=False,
        raise_error_if_all_missing=not allow_missing_days
    )
    output_file_name = radar_io.find_file(
        top_directory_name=output_dir_name, valid_date_string=date_string,
        with_3d=False, raise_error_if_missing=False
    )

    for i in range(len(input_file_names)):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        composite_refl_matrix_dbz, latitudes_deg_n, longitudes_deg_e = (
            twb_radar_io.read_2d_file(binary_file_name=input_file_names[i])
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        radar_io.write_2d_file(
            netcdf_file_name=output_file_name,
            composite_refl_matrix_dbz=composite_refl_matrix_dbz,
            latitudes_deg_n=latitudes_deg_n, longitudes_deg_e=longitudes_deg_e,
            valid_time_unix_sec=
            twb_radar_io.file_name_to_time(input_file_names[i]),
            append=i > 0
        )


def _run(input_dir_name, first_date_string, last_date_string,
         allow_missing_days, with_3d, output_dir_name):
    """Converts radar data to daily NetCDF files.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param allow_missing_days: Same.
    :param with_3d: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if `with_3d == True`, since I still have not figured
        out how to read the raw files.
    """

    if with_3d:
        raise ValueError('Cannot read raw 3-D files yet.')

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    for i in range(len(date_strings)):
        _process_radar_data_one_day(
            input_dir_name=input_dir_name, date_string=date_strings[i],
            allow_missing_days=allow_missing_days,
            output_dir_name=output_dir_name
        )

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
        with_3d=bool(getattr(INPUT_ARG_OBJECT, WITH_3D_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
