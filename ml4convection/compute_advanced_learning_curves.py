"""Computes advanced scores for learning curves."""

import os
import sys
import glob
import numpy
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import learning_curves
import radar_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

INPUT_DIR_ARG_NAME = 'input_basic_score_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (containing basic scores) will be '
    'found by `learning_curves.find_basic_score_file` and read by '
    '`learning_curves.read_basic_score_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results (advanced scores) will be written here by '
    '`learning_curves.write_scores`.'
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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_basic_score_dir_name, first_date_string, last_date_string,
         output_file_name):
    """Computes advanced scores for learning curves.

    This is effectively the main method.

    :param top_basic_score_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param output_file_name: Same.
    """

    if not os.path.isdir(top_basic_score_dir_name):
        top_basic_score_dir_names = glob.glob(top_basic_score_dir_name)
        validation_losses = numpy.full(len(top_basic_score_dir_names), numpy.nan)

        for i in range(len(top_basic_score_dir_names)):
            these_words = (
                top_basic_score_dir_names[i].replace('/', '_').split('_')
            )

            for this_word in these_words:
                if not this_word.startswith('val-loss='):
                    continue

                validation_losses[i] = float(this_word.replace('val-loss=', ''))

        min_index = numpy.nanargmin(validation_losses)
        top_basic_score_dir_name = top_basic_score_dir_names[min_index]

        output_file_name = '{0:s}/advanced_scores.nc'.format(
            top_basic_score_dir_name
        )

    date_strings = []
    basic_score_file_names = []

    for k in range(NUM_RADARS):
        if len(date_strings) == 0:
            basic_score_file_names += (
                learning_curves.find_many_basic_score_files(
                    top_directory_name=top_basic_score_dir_name,
                    first_date_string=first_date_string,
                    last_date_string=last_date_string, radar_number=k,
                    raise_error_if_any_missing=False,
                    raise_error_if_all_missing=k > 0
                )
            )

            if len(basic_score_file_names) == 0:
                continue

            date_strings = [
                learning_curves.basic_file_name_to_date(f)
                for f in basic_score_file_names
            ]
        else:
            basic_score_file_names += [
                learning_curves.find_basic_score_file(
                    top_directory_name=top_basic_score_dir_name,
                    valid_date_string=d, radar_number=k,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

    basic_score_tables = []

    for this_file_name in basic_score_file_names:
        print('Reading basic scores from: "{0:s}"...'.format(this_file_name))
        basic_score_tables.append(
            learning_curves.read_scores(this_file_name)
        )

    print(SEPARATOR_STRING)

    basic_score_table_xarray = (
        learning_curves.concat_basic_score_tables(basic_score_tables)
    )
    del basic_score_tables

    advanced_score_table_xarray = learning_curves.get_advanced_scores(
        basic_score_table_xarray
    )
    print(advanced_score_table_xarray)

    print('\nWriting advanced scores to: "{0:s}"...'.format(output_file_name))
    learning_curves.write_scores(
        score_table_xarray=advanced_score_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_basic_score_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
