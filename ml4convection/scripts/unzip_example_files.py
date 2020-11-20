"""Unzips example files on partial grid."""

import os.path
import argparse
from gewittergefahr.gg_utils import unzipping
from ml4convection.io import example_io
from ml4convection.utils import radar_utils

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'

PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictor files.  Files therein will be '
    'found by `example_io.find_predictor_file` and read by '
    '`example_io.read_predictor_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with target files.  Files therein will be '
    'found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will unzip files for the period '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTOR_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)


def _run(top_predictor_dir_name, top_target_dir_name, first_date_string,
         last_date_string):
    """Unzips example files on partial grid.

    This is effectively the main method.

    :param top_predictor_dir_name: See documentation at top of file.
    :param top_target_dir_name: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    """

    predictor_file_names = example_io.find_many_predictor_files(
        top_directory_name=top_predictor_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string, radar_number=0,
        prefer_zipped=True, allow_other_format=False,
        raise_error_if_all_missing=False,
        raise_error_if_any_missing=False
    )

    predictor_date_strings = [
        example_io.file_name_to_date(f) for f in predictor_file_names
    ]

    for k in range(1, NUM_RADARS):
        predictor_file_names += [
            example_io.find_predictor_file(
                top_directory_name=top_predictor_dir_name,
                date_string=d, radar_number=k, prefer_zipped=True,
                allow_other_format=False, raise_error_if_missing=True
            ) for d in predictor_date_strings
        ]

    for this_zipped_file_name in predictor_file_names:
        this_directory_name, this_pathless_file_name = os.path.split(
            this_zipped_file_name
        )
        this_unzipped_file_name = '{0:s}/{1:s}'.format(
            this_directory_name,
            this_pathless_file_name.replace(example_io.GZIP_FILE_EXTENSION, '')
        )

        print('Unzipping file to: "{0:s}"...'.format(this_unzipped_file_name))
        unzipping.unzip_gzip(
            gzip_file_name=this_zipped_file_name,
            extracted_file_name=this_unzipped_file_name
        )

    target_file_names = example_io.find_many_target_files(
        top_directory_name=top_target_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string, radar_number=0,
        prefer_zipped=True, allow_other_format=False,
        raise_error_if_all_missing=False,
        raise_error_if_any_missing=False
    )

    target_date_strings = [
        example_io.file_name_to_date(f) for f in target_file_names
    ]

    for k in range(1, NUM_RADARS):
        target_file_names += [
            example_io.find_target_file(
                top_directory_name=top_target_dir_name,
                date_string=d, radar_number=k, prefer_zipped=True,
                allow_other_format=False, raise_error_if_missing=True
            ) for d in target_date_strings
        ]

    for this_zipped_file_name in target_file_names:
        this_directory_name, this_pathless_file_name = os.path.split(
            this_zipped_file_name
        )
        this_unzipped_file_name = '{0:s}/{1:s}'.format(
            this_directory_name,
            this_pathless_file_name.replace(example_io.GZIP_FILE_EXTENSION, '')
        )

        print('Unzipping file to: "{0:s}"...'.format(this_unzipped_file_name))
        unzipping.unzip_gzip(
            gzip_file_name=this_zipped_file_name,
            extracted_file_name=this_unzipped_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME)
    )
