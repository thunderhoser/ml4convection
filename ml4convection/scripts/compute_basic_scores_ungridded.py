"""Computes basic evaluation scores sans grid (combined over full domain)."""

import argparse
import numpy
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from ml4convection.io import prediction_io
from ml4convection.utils import evaluation
from ml4convection.utils import radar_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
USE_PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'
MATCHING_DISTANCES_ARG_NAME = 'matching_distances_px'
NUM_PROB_THRESHOLDS_ARG_NAME = 'num_prob_thresholds'
PROB_THRESHOLDS_ARG_NAME = 'prob_thresholds'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

USE_PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will compute scores for partial (full) grids.'
)
MATCHING_DISTANCES_HELP_STRING = (
    'List of matching distances (pixels).  Neighbourhood evaluation will be '
    'done for each matching distance.'
)
NUM_PROB_THRESHOLDS_HELP_STRING = (
    'Number of probability thresholds.  One contingency table will be created '
    'for each.  If you want to use specific thresholds, leave this argument '
    'alone and specify `{0:s}`.'
).format(PROB_THRESHOLDS_ARG_NAME)

PROB_THRESHOLDS_HELP_STRING = (
    'List of exact probability thresholds.  One contingency table will be '
    'created for each.  If you do not want to use specific thresholds, leave '
    'this argument alone and specify `{0:s}`.'
).format(NUM_PROB_THRESHOLDS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`evaluation.write_basic_score_file`, to exact locations determined by '
    '`evaluation.find_basic_score_file`.'
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
    '--' + USE_PARTIAL_GRIDS_ARG_NAME, type=int, required=False, default=0,
    help=USE_PARTIAL_GRIDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCES_ARG_NAME, type=float, nargs='+', required=True,
    help=MATCHING_DISTANCES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PROB_THRESHOLDS_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLDS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _compute_scores_full_grid(
        top_prediction_dir_name, first_date_string, last_date_string,
        matching_distances_px, prob_thresholds, top_output_dir_name):
    """Computes scores on full grid.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param matching_distances_px: Same.
    :param prob_thresholds: Same.
    :param top_output_dir_name: Same.
    """

    prediction_file_names = prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        radar_number=None, prefer_zipped=False, allow_other_format=True,
        raise_error_if_any_missing=False
    )
    date_strings = [
        prediction_io.file_name_to_date(f) for f in prediction_file_names
    ]

    num_dates = len(date_strings)
    num_matching_distances = len(matching_distances_px)

    for i in range(num_dates):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_dict = prediction_io.read_file(prediction_file_names[i])

        for j in range(num_matching_distances):
            print('\n')

            basic_score_table_xarray = evaluation.get_basic_scores_ungridded(
                prediction_dict=prediction_dict,
                matching_distance_px=matching_distances_px[j],
                probability_thresholds=prob_thresholds
            )

            output_dir_name = '{0:s}/matching_distance_px={1:.6f}'.format(
                top_output_dir_name, matching_distances_px[j]
            )
            output_file_name = evaluation.find_basic_score_file(
                top_directory_name=output_dir_name,
                valid_date_string=date_strings[i],
                gridded=False, radar_number=None, raise_error_if_missing=False
            )

            print('\nWriting results to: "{0:s}"...'.format(output_file_name))
            evaluation.write_basic_score_file(
                basic_score_table_xarray=basic_score_table_xarray,
                netcdf_file_name=output_file_name
            )

        if i == num_dates - 1:
            continue

        print(SEPARATOR_STRING)


def _compute_scores_partial_grids(
        top_prediction_dir_name, first_date_string, last_date_string,
        matching_distances_px, prob_thresholds, top_output_dir_name):
    """Computes scores on partial grids.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param matching_distances_px: Same.
    :param prob_thresholds: Same.
    :param top_output_dir_name: Same.
    """

    date_strings = []

    for k in range(NUM_RADARS):
        if k == 0:
            prediction_file_names = prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                radar_number=k, prefer_zipped=True, allow_other_format=True,
                raise_error_if_any_missing=False
            )

            date_strings = [
                prediction_io.file_name_to_date(f)
                for f in prediction_file_names
            ]
        else:
            prediction_file_names = [
                prediction_io.find_file(
                    top_directory_name=top_prediction_dir_name,
                    valid_date_string=d, radar_number=k,
                    prefer_zipped=True, allow_other_format=True,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

        num_dates = len(date_strings)
        num_matching_distances = len(matching_distances_px)

        for i in range(num_dates):
            print('Reading data from: "{0:s}"...'.format(
                prediction_file_names[i]
            ))
            prediction_dict = prediction_io.read_file(prediction_file_names[i])

            for j in range(num_matching_distances):
                print('\n')

                basic_score_table_xarray = (
                    evaluation.get_basic_scores_ungridded(
                        prediction_dict=prediction_dict,
                        matching_distance_px=matching_distances_px[j],
                        probability_thresholds=prob_thresholds
                    )
                )

                output_dir_name = '{0:s}/matching_distance_px={1:.6f}'.format(
                    top_output_dir_name, matching_distances_px[j]
                )
                output_file_name = evaluation.find_basic_score_file(
                    top_directory_name=output_dir_name,
                    valid_date_string=date_strings[i],
                    gridded=False, radar_number=k, raise_error_if_missing=False
                )

                print('\nWriting results to: "{0:s}"...'.format(
                    output_file_name
                ))
                evaluation.write_basic_score_file(
                    basic_score_table_xarray=basic_score_table_xarray,
                    netcdf_file_name=output_file_name
                )

            if not (i == num_dates - 1 and k == NUM_RADARS - 1):
                continue

            print(SEPARATOR_STRING)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         use_partial_grids, matching_distances_px, num_prob_thresholds,
         prob_thresholds, top_output_dir_name):
    """Computes basic evaluation scores sans grid (combined over full domain).

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param use_partial_grids: Same.
    :param matching_distances_px: Same.
    :param num_prob_thresholds: Same.
    :param prob_thresholds: Same.
    :param top_output_dir_name: Same.
    """

    if num_prob_thresholds > 0:
        prob_thresholds = gg_model_eval.get_binarization_thresholds(
            threshold_arg=num_prob_thresholds
        )

    if not use_partial_grids:
        _compute_scores_full_grid(
            top_prediction_dir_name=top_prediction_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            matching_distances_px=matching_distances_px,
            prob_thresholds=prob_thresholds,
            top_output_dir_name=top_output_dir_name
        )

        return

    _compute_scores_partial_grids(
        top_prediction_dir_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        matching_distances_px=matching_distances_px,
        prob_thresholds=prob_thresholds,
        top_output_dir_name=top_output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        use_partial_grids=bool(getattr(
            INPUT_ARG_OBJECT, USE_PARTIAL_GRIDS_ARG_NAME
        )),
        matching_distances_px=numpy.array(
            getattr(INPUT_ARG_OBJECT, MATCHING_DISTANCES_ARG_NAME), dtype=float
        ),
        num_prob_thresholds=getattr(
            INPUT_ARG_OBJECT, NUM_PROB_THRESHOLDS_ARG_NAME
        ),
        prob_thresholds=numpy.array(getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLDS_ARG_NAME
        )),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
