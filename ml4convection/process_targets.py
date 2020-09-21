"""Processes echo-classification data into target files."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_io

INPUT_DIR_ARG_NAME = 'input_echo_classifn_dir_name'
SPATIAL_DS_FACTOR_ARG_NAME = 'spatial_downsampling_factor'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_DIR_ARG_NAME = 'output_predictor_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with echo-classification data.  Files therein '
    'will be found by `radar_io.find_file` and read by '
    '`radar_io.read_echo_classifn_file`.'
)
SPATIAL_DS_FACTOR_HELP_STRING = (
    'Downsampling factor, used to coarsen spatial resolution.  If you do not '
    'want to coarsen spatial resolution, make this 1.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will process data for all days in the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Predictor files will be written by '
    '`example_io._write_target_file`, to exact locations therein determined '
    'by `example_io.find_target_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SPATIAL_DS_FACTOR_ARG_NAME, type=int, required=True,
    help=SPATIAL_DS_FACTOR_HELP_STRING
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


def _run(top_input_dir_name, spatial_downsampling_factor, first_date_string,
         last_date_string, top_output_dir_name):
    """Processes satellite data into predictor files.

    This is effectively the main method.

    :param top_input_dir_name: See documentation at top of file.
    :param spatial_downsampling_factor: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param top_output_dir_name: Same.
    """

    example_io.process_targets(
        top_input_dir_name=top_input_dir_name,
        spatial_downsampling_factor=spatial_downsampling_factor,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        top_output_dir_name=top_output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        spatial_downsampling_factor=getattr(
            INPUT_ARG_OBJECT, SPATIAL_DS_FACTOR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
