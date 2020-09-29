"""Computes advanced evaluation scores."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_basic_score_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein (containing basic scores) will be '
    'found by `evaluation.find_file` and read by `evaluation.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results (advanced scores) will be written here by '
    '`evaluation.write_file`.'
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
    """Computes advanced evaluation scores.

    This is effectively the main method.

    :param top_basic_score_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param output_file_name: Same.
    """

    basic_score_file_names = evaluation.find_many_files(
        top_directory_name=top_basic_score_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        with_basic_scores=True, raise_error_if_any_missing=False
    )

    date_strings = [
        evaluation.file_name_to_date(f) for f in basic_score_file_names
    ]
    num_dates = len(date_strings)
    basic_score_tables_xarray = [None] * num_dates

    for i in range(num_dates):
        print('Reading basic scores from: "{0:s}"...'.format(
            basic_score_file_names[i]
        ))
        basic_score_tables_xarray[i] = evaluation.read_file(
            basic_score_file_names[i]
        )

    basic_score_table_xarray = evaluation.concat_basic_score_tables(
        basic_score_tables_xarray
    )

    print('\nComputing advanced scores...')
    advanced_score_table_xarray = evaluation.get_advanced_scores(
        basic_score_table_xarray
    )
    print(advanced_score_table_xarray)

    print('Writing advanced scores to: "{0:s}"...'.format(output_file_name))
    evaluation.write_file(
        score_table_xarray=advanced_score_table_xarray,
        pickle_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_basic_score_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
