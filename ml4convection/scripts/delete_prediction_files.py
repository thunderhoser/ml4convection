"""Deletes prediction files."""

import os
import argparse
from ml4convection.io import prediction_io
from ml4convection.utils import radar_utils

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

PREDICTION_DIR_ARG_NAME = 'prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with prediction files.  Files therein will be '
    'found by `prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will delete prediction files for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will delete files with predictions on partial '
    '(full) grid.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PARTIAL_GRIDS_ARG_NAME, type=int, required=True,
    help=PARTIAL_GRIDS_HELP_STRING
)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         use_partial_grids):
    """Deletes prediction files.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param use_partial_grids: Same.
    """

    prediction_file_names = []

    if not use_partial_grids:
        prediction_file_names += prediction_io.find_many_files(
            top_directory_name=top_prediction_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            prefer_zipped=True, allow_other_format=False, radar_number=None,
            raise_error_if_any_missing=False, raise_error_if_all_missing=False
        )

        prediction_file_names += prediction_io.find_many_files(
            top_directory_name=top_prediction_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            prefer_zipped=False, allow_other_format=False, radar_number=None,
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=len(prediction_file_names) == 0
        )
    else:
        for k in range(NUM_RADARS):
            prediction_file_names += prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                prefer_zipped=True, allow_other_format=False, radar_number=k,
                raise_error_if_any_missing=False,
                raise_error_if_all_missing=False
            )

            this_flag = len(prediction_file_names) == 0 and k == NUM_RADARS - 1

            prediction_file_names += prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                prefer_zipped=False, allow_other_format=False, radar_number=k,
                raise_error_if_any_missing=False,
                raise_error_if_all_missing=this_flag
            )

    for this_file_name in prediction_file_names:
        print('Deleting file: "{0:s}"...'.format(this_file_name))
        os.remove(this_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        use_partial_grids=bool(
            getattr(INPUT_ARG_OBJECT, PARTIAL_GRIDS_ARG_NAME)
        )
    )
