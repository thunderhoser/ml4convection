"""Deletes files with basic scores."""

import os
import argparse
from ml4convection.utils import evaluation
from ml4convection.utils import radar_utils

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

EVALUATION_DIR_ARG_NAME = 'evaluation_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
GRIDDED_ARG_NAME = 'gridded'
PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'

EVALUATION_DIR_HELP_STRING = (
    'Name of top-level directory with score files.  Files therein will be '
    'found by `evaluation.find_basic_score_file` and read by '
    '`evaluation.read_basic_score_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will delete score files for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

GRIDDED_HELP_STRING = (
    'Boolean flag.  If 1 (0), will delete files with (un)gridded scores.'
)
PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will delete files on partial grids (full grid).'
)

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
INPUT_ARG_PARSER.add_argument(
    '--' + GRIDDED_ARG_NAME, type=int, required=True, help=GRIDDED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PARTIAL_GRIDS_ARG_NAME, type=int, required=True,
    help=PARTIAL_GRIDS_HELP_STRING
)


def _run(top_evaluation_dir_name, first_date_string, last_date_string, gridded,
         use_partial_grids):
    """Deletes files with basic gridded scores.

    This is effectively the main method.

    :param top_evaluation_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param gridded: Same.
    :param use_partial_grids: Same.
    """

    if use_partial_grids:
        date_strings = []
        basic_score_file_names = []

        for k in range(NUM_RADARS):
            if len(date_strings) == 0:
                basic_score_file_names += (
                    evaluation.find_many_basic_score_files(
                        top_directory_name=top_evaluation_dir_name,
                        first_date_string=first_date_string,
                        last_date_string=last_date_string,
                        gridded=gridded, radar_number=k,
                        raise_error_if_any_missing=False,
                        raise_error_if_all_missing=k > 0
                    )
                )

                if len(basic_score_file_names) > 0:
                    date_strings = [
                        evaluation.basic_file_name_to_date(f)
                        for f in basic_score_file_names
                    ]
            else:
                basic_score_file_names += [
                    evaluation.find_basic_score_file(
                        top_directory_name=top_evaluation_dir_name,
                        valid_date_string=d, gridded=gridded, radar_number=k,
                        raise_error_if_missing=True
                    ) for d in date_strings
                ]
    else:
        basic_score_file_names = evaluation.find_many_basic_score_files(
            top_directory_name=top_evaluation_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            gridded=gridded, radar_number=None, raise_error_if_all_missing=True
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
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        gridded=bool(getattr(INPUT_ARG_OBJECT, GRIDDED_ARG_NAME)),
        use_partial_grids=bool(
            getattr(INPUT_ARG_OBJECT, PARTIAL_GRIDS_ARG_NAME)
        )
    )
