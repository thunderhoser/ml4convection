"""Computes basic evaluation scores."""

import argparse
from ml4convection.io import prediction_io
from ml4convection.utils import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_px'
NUM_PROB_THRESHOLDS_ARG_NAME = 'num_prob_thresholds'

# TODO(thunderhoser): This will be non-trivial to compute for each matching
# distance.
TRAINING_EVENT_FREQ_ARG_NAME = 'training_event_frequency'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance (pixels) for neighbourhood evaluation.'
)
NUM_PROB_THRESHOLDS_HELP_STRING = (
    'Number of probability thresholds.  One contingency table will be created '
    'for each.'
)
TRAINING_EVENT_FREQ_HELP_STRING = (
    'Event frequency in training data for the given matching distance.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`evaluation.write_file`, to exact locations determined by '
    '`evaluation.find_file`.'
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
    '--' + MATCHING_DISTANCE_ARG_NAME, type=float, required=True,
    help=MATCHING_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PROB_THRESHOLDS_ARG_NAME, type=int, required=False,
    default=evaluation.DEFAULT_NUM_PROB_THRESHOLDS,
    help=NUM_PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_EVENT_FREQ_ARG_NAME, type=float, required=True,
    help=TRAINING_EVENT_FREQ_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         matching_distance_px, training_event_frequency, top_output_dir_name):
    """Computes basic evaluation scores.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param matching_distance_px: Same.
    :param training_event_frequency: Same.
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

    for i in range(num_dates):
        this_score_table_xarray = evaluation.get_basic_scores(
            prediction_file_name=prediction_file_names[i],
            matching_distance_px=matching_distance_px,
            training_event_frequency=training_event_frequency
        )

        this_output_file_name = evaluation.find_file(
            top_directory_name=top_output_dir_name,
            valid_date_string=date_strings[i], with_basic_scores=True,
            raise_error_if_missing=False
        )

        print('\nWriting results to file: "{0:s}"...'.format(
            this_output_file_name
        ))
        evaluation.write_file(
            score_table_xarray=this_score_table_xarray,
            pickle_file_name=this_output_file_name
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
        matching_distance_px=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME
        ),
        training_event_frequency=getattr(
            INPUT_ARG_OBJECT, TRAINING_EVENT_FREQ_ARG_NAME
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
