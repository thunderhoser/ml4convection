"""Compresses satellite data into daily gzip files."""

import os
import argparse
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import satellite_io

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400

BAND_NUMBERS = satellite_io.BAND_NUMBERS

SATELLITE_DIR_ARG_NAME = 'satellite_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'

SATELLITE_DIR_HELP_STRING = 'Name of input/output directory.'
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Satellite data will be processed for all days '
    'in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)


def _run(top_satellite_dir_name, first_date_string, last_date_string):
    """Compresses satellite data into daily gzip files.

    This is effectively the main method.

    :param top_satellite_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    """

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    for i in range(len(date_strings)):
        this_netcdf_file_name = satellite_io.find_file(
            top_directory_name=top_satellite_dir_name,
            valid_date_string=date_strings[i],
            prefer_zipped=False, allow_other_format=False,
            raise_error_if_missing=False
        )

        if not os.path.isfile(this_netcdf_file_name):
            continue

        print('Compressing file: "{0:s}"...'.format(this_netcdf_file_name))
        satellite_io.compress_file(this_netcdf_file_name)
        os.remove(this_netcdf_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME)
    )
