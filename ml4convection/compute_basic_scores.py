"""Computes basic evaluation scores."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import prediction_io
import climatology_io
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
GRIDDED_ARG_NAME = 'gridded'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_px'
NUM_PROB_THRESHOLDS_ARG_NAME = 'num_prob_thresholds'
PROB_THRESHOLDS_ARG_NAME = 'prob_thresholds'
CLIMO_FILE_ARG_NAME = 'input_climo_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

GRIDDED_HELP_STRING = (
    'Boolean flag.  If 1, scores will be gridded (one set for each pixel).  If '
    '0, scores will be aggregated (one set for the full domain).'
)
MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance (pixels) for neighbourhood evaluation.'
)
NUM_PROB_THRESHOLDS_HELP_STRING = (
    '[used only if `{0:s}` = 0] Number of probability thresholds.  One '
    'contingency table will be created for each.'
).format(GRIDDED_ARG_NAME)

PROB_THRESHOLDS_HELP_STRING = (
    '[used only if `{0:s}` = 1] List of exact probability thresholds.  One '
    'contingency table will be created for each.'
).format(GRIDDED_ARG_NAME)

CLIMO_FILE_HELP_STRING = (
    '[used only if `{0:s}` = 1] Path to file with climatology (event '
    'frequencies in training data).'
).format(GRIDDED_ARG_NAME)

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
    '--' + GRIDDED_ARG_NAME, type=int, required=True, help=GRIDDED_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCE_ARG_NAME, type=float, required=True,
    help=MATCHING_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PROB_THRESHOLDS_ARG_NAME, type=int, required=False,
    default=evaluation.DEFAULT_NUM_PROB_THRESHOLDS,
    help=NUM_PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLDS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLIMO_FILE_ARG_NAME, type=str, required=False, default='',
    help=CLIMO_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         gridded, matching_distance_px, num_prob_thresholds, prob_thresholds,
         climo_file_name, top_output_dir_name):
    """Computes basic evaluation scores.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param gridded: Same.
    :param matching_distance_px: Same.
    :param num_prob_thresholds: Same.
    :param prob_thresholds: Same.
    :param climo_file_name: Same.
    :param top_output_dir_name: Same.
    """

    prediction_file_names = prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        raise_error_if_any_missing=False
    )

    date_strings = [
        prediction_io.file_name_to_date(f) for f in prediction_file_names
    ]
    num_dates = len(date_strings)

    if gridded:
        print('Reading event frequencies from: "{0:s}"...'.format(
            climo_file_name
        ))
        climo_dict = climatology_io.read_file(climo_file_name)
        training_event_freq_matrix = (
            climo_dict[climatology_io.EVENT_FREQ_BY_PIXEL_KEY]
        )
    else:
        training_event_freq_matrix = None

    for i in range(num_dates):
        if gridded:
            this_score_table_xarray = evaluation.get_basic_scores_gridded(
                prediction_file_name=prediction_file_names[i],
                matching_distance_px=matching_distance_px,
                probability_thresholds=prob_thresholds,
                training_event_freq_matrix=training_event_freq_matrix
            )
        else:
            this_score_table_xarray = evaluation.get_basic_scores_ungridded(
                prediction_file_name=prediction_file_names[i],
                matching_distance_px=matching_distance_px,
                num_prob_thresholds=num_prob_thresholds
            )

        this_output_file_name = evaluation.find_basic_score_file(
            top_directory_name=top_output_dir_name,
            valid_date_string=date_strings[i],
            gridded=gridded, raise_error_if_missing=False
        )

        print('\nWriting results to file: "{0:s}"...'.format(
            this_output_file_name
        ))
        evaluation.write_basic_score_file(
            basic_score_table_xarray=this_score_table_xarray,
            netcdf_file_name=this_output_file_name
        )

        if i == num_dates - 1:
            continue

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        gridded=bool(getattr(INPUT_ARG_OBJECT, GRIDDED_ARG_NAME)),
        matching_distance_px=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME
        ),
        num_prob_thresholds=getattr(
            INPUT_ARG_OBJECT, NUM_PROB_THRESHOLDS_ARG_NAME
        ),
        prob_thresholds=numpy.array(getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLDS_ARG_NAME
        )),
        climo_file_name=getattr(INPUT_ARG_OBJECT, CLIMO_FILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
