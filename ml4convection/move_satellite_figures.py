"""Moves satellite figures into new directory structure."""

import os
import sys
import shutil
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import file_system_utils
import satellite_io
import twb_satellite_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

BAND_NUMBERS = satellite_io.BAND_NUMBERS
TIME_INTERVAL_SEC = twb_satellite_io.TIME_INTERVAL_SEC

DAYS_TO_SECONDS = 86400
INPUT_DATE_FORMAT = '%Y%m%d'
TIME_FORMAT = '%Y-%m-%d-%H%M'

INPUT_DIR_ARG_NAME = 'input_figure_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_DIR_ARG_NAME = 'output_figure_dir_name'

INPUT_DIR_HELP_STRING = 'Name of top-level input directory.'
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will move figures for all days in the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = 'Name of top-level output directory.'

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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_input_dir_name, first_date_string, last_date_string,
         top_output_dir_name):
    """Moves satellite figures into new directory structure.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param top_output_dir_name: Same.
    :raises: ValueError: if file cannot be found.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_date_string, INPUT_DATE_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_date_string, INPUT_DATE_FORMAT
    )
    last_time_unix_sec += DAYS_TO_SECONDS - TIME_INTERVAL_SEC

    valid_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_time_unix_sec,
        end_time_unix_sec=last_time_unix_sec,
        time_interval_sec=TIME_INTERVAL_SEC, include_endpoint=True
    )
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in valid_times_unix_sec
    ]
    month_strings = [t[:7].replace('-', '') for t in valid_time_strings]

    num_times = len(valid_times_unix_sec)
    num_bands = len(BAND_NUMBERS)

    for i in range(num_times):
        this_input_file_name = (
            '{0:s}/{1:s}/brightness-temperature_{2:s}_band{3:02d}.jpg'
        ).format(
            top_output_dir_name, month_strings[i], valid_time_strings[i],
            BAND_NUMBERS[0]
        )

        if not os.path.isfile(this_input_file_name):
            continue

        print('\n')

        for j in range(num_bands):
            this_input_file_name = (
                '{0:s}/{1:s}/brightness-temperature_{2:s}_band{3:02d}.jpg'
            ).format(
                top_output_dir_name, month_strings[i], valid_time_strings[i],
                BAND_NUMBERS[j]
            )

            if not os.path.isfile(this_input_file_name):
                error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
                    this_input_file_name
                )
                raise ValueError(error_string)

            this_output_file_name = (
                '{0:s}/{1:s}/brightness-temperature_{2:s}_band{3:02d}.jpg'
            ).format(
                top_output_dir_name, valid_time_strings[i][:10],
                valid_time_strings[i], BAND_NUMBERS[j]
            )

            # file_system_utils.mkdir_recursive_if_necessary(
            #     file_name=this_output_file_name
            # )

            print('Moving file from "{0:s}" to "{1:s}"...'.format(
                this_input_file_name, this_output_file_name
            ))
            # shutil.move(this_input_file_name, this_output_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
