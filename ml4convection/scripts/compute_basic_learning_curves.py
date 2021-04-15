"""Computes basic scores for learning curves."""

import os
import copy
import argparse
import numpy
from ml4convection.io import prediction_io
from ml4convection.utils import learning_curves
from ml4convection.utils import radar_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
VALID_DATES_ARG_NAME = 'valid_date_strings'
NEIGH_DISTANCES_ARG_NAME = 'neigh_distances_px'
MIN_RESOLUTIONS_ARG_NAME = 'min_fourier_resolutions_deg'
MAX_RESOLUTIONS_ARG_NAME = 'max_fourier_resolutions_deg'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
VALID_DATES_HELP_STRING = (
    'List of valid dates (format "yyyymmdd").  Will evaluate predictions for '
    'each day.'
)

NEIGH_DISTANCES_HELP_STRING = (
    'List of matching distances (pixels) for neighbourhood-based evaluation.  '
    'If you do not want neighbourhood-based evaluation, leave this alone.'
)
MIN_RESOLUTIONS_HELP_STRING = (
    'List of minimum resolutions (degrees) for band-pass filters.  If you do '
    'not want Fourier-based evaluation, leave this alone.'
)
MAX_RESOLUTIONS_HELP_STRING = (
    'Same as `{0:s}` but with max resolutions.'.format(MIN_RESOLUTIONS_ARG_NAME)
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`learning_curves.write_scores`, to exact locations determined by '
    '`learning_curves.find_basic_score_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=VALID_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEIGH_DISTANCES_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=NEIGH_DISTANCES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_RESOLUTIONS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=MIN_RESOLUTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RESOLUTIONS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=MAX_RESOLUTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_prediction_dir_name, valid_date_strings, neigh_distances_px,
         min_fourier_resolutions_deg, max_fourier_resolutions_deg,
         top_output_dir_name):
    """Computes basic scores for learning curves.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param valid_date_strings: Same.
    :param neigh_distances_px: Same.
    :param min_fourier_resolutions_deg: Same.
    :param max_fourier_resolutions_deg: Same.
    :param top_output_dir_name: Same.
    """

    if len(neigh_distances_px) == 1 and neigh_distances_px[0] < 0:
        neigh_distances_px = None

    if (
            len(min_fourier_resolutions_deg) == 1
            and min_fourier_resolutions_deg[0] < 0
    ):
        min_fourier_resolutions_deg = None

    if (
            len(max_fourier_resolutions_deg) == 1
            and max_fourier_resolutions_deg[0] < 0
    ):
        max_fourier_resolutions_deg = None

    all_date_strings = copy.deepcopy(valid_date_strings)

    for k in range(NUM_RADARS):
        prediction_file_names = [
            prediction_io.find_file(
                top_directory_name=top_prediction_dir_name,
                valid_date_string=d, radar_number=k,
                prefer_zipped=True, allow_other_format=True,
                raise_error_if_missing=False
            ) for d in all_date_strings
        ]

        prediction_file_names = [
            f for f in prediction_file_names if os.path.isfile(f)
        ]
        valid_date_strings = [
            prediction_io.file_name_to_date(f)
            for f in prediction_file_names
        ]

        num_dates = len(valid_date_strings)

        for i in range(num_dates):
            print('Reading data from: "{0:s}"...'.format(
                prediction_file_names[i]
            ))
            prediction_dict = prediction_io.read_file(prediction_file_names[i])

            basic_score_table_xarray = learning_curves.get_basic_scores(
                prediction_dict=prediction_dict,
                neigh_distances_px=neigh_distances_px,
                min_fourier_resolutions_deg=min_fourier_resolutions_deg,
                max_fourier_resolutions_deg=max_fourier_resolutions_deg
            )

            output_file_name = learning_curves.find_basic_score_file(
                top_directory_name=top_output_dir_name,
                valid_date_string=valid_date_strings[i],
                radar_number=k, raise_error_if_missing=False
            )

            print('Writing results to: "{0:s}"...'.format(output_file_name))
            learning_curves.write_scores(
                score_table_xarray=basic_score_table_xarray,
                netcdf_file_name=output_file_name
            )

            if not (i == num_dates - 1 and k == NUM_RADARS - 1):
                continue

            print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        valid_date_strings=getattr(INPUT_ARG_OBJECT, VALID_DATES_ARG_NAME),
        neigh_distances_px=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEIGH_DISTANCES_ARG_NAME), dtype=float
        ),
        min_fourier_resolutions_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_RESOLUTIONS_ARG_NAME), dtype=float
        ),
        max_fourier_resolutions_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_RESOLUTIONS_ARG_NAME), dtype=float
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
