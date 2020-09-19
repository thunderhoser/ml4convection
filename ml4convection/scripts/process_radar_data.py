"""Converts radar data to daily NetCDF files."""

import os
import argparse
import warnings
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import twb_radar_io
from ml4convection.io import radar_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

INPUT_DIR_ARG_NAME = 'input_radar_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
ALLOW_MISSING_DAYS_ARG_NAME = 'allow_missing_days'
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
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Daily NetCDF files will be written by '
    '`radar_io.write_reflectivity_file`, to locations therein determined by '
    '`radar_io.find_file`.'
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _process_radar_data_one_day(
        input_dir_name, date_string, allow_missing_days, output_file_name):
    """Processes radar data for one day.

    :param input_dir_name: See documentation at top of file.
    :param date_string: Will process for this day (format "yyyymmdd").
    :param allow_missing_days: See documentation at top of file.
    :param output_file_name: Path to output file.
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
        last_time_unix_sec=last_time_unix_sec, with_3d=True,
        raise_error_if_all_missing=not allow_missing_days
    )

    append = False

    for this_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))

        try:
            (
                reflectivity_matrix_dbz,
                latitudes_deg_n,
                longitudes_deg_e,
                heights_m_asl
            ) = twb_radar_io.read_file(this_file_name)
        except Exception as e:
            warning_string = 'WARNING: {0:s}'.format(str(e))
            warnings.warn(warning_string)
            continue

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        radar_io.write_reflectivity_file(
            netcdf_file_name=output_file_name,
            reflectivity_matrix_dbz=reflectivity_matrix_dbz,
            latitudes_deg_n=latitudes_deg_n,
            longitudes_deg_e=longitudes_deg_e,
            heights_m_asl=heights_m_asl,
            valid_time_unix_sec=twb_radar_io.file_name_to_time(this_file_name),
            append=append
        )

        append = True


def _run(input_dir_name, first_date_string, last_date_string,
         allow_missing_days, output_dir_name):
    """Converts radar data to daily NetCDF files.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param allow_missing_days: Same.
    :param output_dir_name: Same.
    """

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    for i in range(len(date_strings)):
        this_netcdf_file_name = radar_io.find_file(
            top_directory_name=output_dir_name,
            valid_date_string=date_strings[i],
            file_type_string=radar_io.REFL_TYPE_STRING,
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        _process_radar_data_one_day(
            input_dir_name=input_dir_name, date_string=date_strings[i],
            allow_missing_days=allow_missing_days,
            output_file_name=this_netcdf_file_name
        )

        radar_io.compress_file(this_netcdf_file_name)
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
