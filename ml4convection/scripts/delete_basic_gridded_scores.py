"""Deletes files with basic gridded scores."""

import os
import argparse
from ml4convection.utils import evaluation

EVALUATION_DIR_ARG_NAME = 'evaluation_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'

EVALUATION_DIR_HELP_STRING = (
    'Name of top-level directory with score files.  Files therein will be '
    'found by `evaluation.find_basic_score_file` and read by '
    '`evaluation.read_basic_score_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will delete score files for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EVALUATION_DIR_ARG_NAME, type=str, required=True,
    help=EVALUATION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)


def _run(top_evaluation_dir_name, first_date_string, last_date_string):
    """Deletes files with basic gridded scores.

    This is effectively the main method.

    :param top_evaluation_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    """

    basic_score_file_names = evaluation.find_many_basic_score_files(
        top_directory_name=top_evaluation_dir_name,
        first_date_string=first_date_string, last_date_string=last_date_string,
        gridded=True, radar_number=None, raise_error_if_all_missing=True
    )

    for this_file_name in basic_score_file_names:
        print('Deleting file: "{0:s}"...'.format(this_file_name))
        os.remove(this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_evaluation_dir_name=getattr(
            INPUT_ARG_OBJECT, EVALUATION_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME)
    )
