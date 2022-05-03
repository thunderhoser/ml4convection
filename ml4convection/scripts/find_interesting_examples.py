"""Finds interesting examples (valid times)."""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import example_io
from ml4convection.io import prediction_io
from ml4convection.utils import radar_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)
TIME_FORMAT = '%Y-%m-%d-%H%M'

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
GRID_ROW_ARG_NAME = 'grid_row'
GRID_COLUMN_ARG_NAME = 'grid_column'
FIND_HIGHEST_PROBS_ARG_NAME = 'find_highest_probs'
FIND_LOWEST_PROBS_ARG_NAME = 'find_lowest_probs'
MIN_PROB_ARG_NAME = 'min_probability'
MAX_PROB_ARG_NAME = 'max_probability'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with predictions.  Files therein will be found'
    ' by `prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will find interesting examples with valid date '
    'in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

GRID_ROW_HELP_STRING = (
    '"Interesting" examples will be based on probabilities at the grid point '
    'with this row index.'
)
GRID_COLUMN_HELP_STRING = (
    '"Interesting" examples will be based on probabilities at the grid point '
    'with this column index.'
)
FIND_HIGHEST_PROBS_HELP_STRING = (
    'Boolean flag.  If 1, "interesting" examples will be those with the highest'
    ' probabilities at the given grid point.'
)
FIND_LOWEST_PROBS_HELP_STRING = (
    'Boolean flag.  If 1, "interesting" examples will be those with the lowest '
    'probabilities at the given grid point.'
)
MINMAX_PROB_HELP_STRING = (
    'If this argument is non-empty, "interesting" examples will be those with '
    'probabilities from `{0:s}`...`{1:s}` at the given grid point.'
).format(MIN_PROB_ARG_NAME, MAX_PROB_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = 'Number of interesting examples to find.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  IDs of interesting examples will be written here by '
    '`example_io.write_example_ids`.'
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
    '--' + GRID_ROW_ARG_NAME, type=int, required=True, help=GRID_ROW_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRID_COLUMN_ARG_NAME, type=int, required=True,
    help=GRID_COLUMN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIND_HIGHEST_PROBS_ARG_NAME, type=int, required=False, default=0,
    help=FIND_HIGHEST_PROBS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIND_LOWEST_PROBS_ARG_NAME, type=int, required=False, default=0,
    help=FIND_LOWEST_PROBS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PROB_ARG_NAME, type=float, required=False, default=-1,
    help=MINMAX_PROB_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PROB_ARG_NAME, type=float, required=False, default=-1,
    help=MINMAX_PROB_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=100,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         grid_row, grid_column, find_highest_probs, find_lowest_probs,
         min_probability, max_probability, num_examples, output_file_name):
    """Finds interesting examples (valid times).

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param grid_row: Same.
    :param grid_column: Same.
    :param find_highest_probs: Same.
    :param find_lowest_probs: Same.
    :param min_probability: Same.
    :param max_probability: Same.
    :param num_examples: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq(grid_row, 0)
    error_checking.assert_is_geq(grid_column, 0)
    error_checking.assert_is_greater(num_examples, 0)

    find_probs_in_range = min_probability >= 0 and max_probability >= 0

    if find_highest_probs:
        find_lowest_probs = False
        find_probs_in_range = False
    if find_lowest_probs:
        find_probs_in_range = False

    if find_probs_in_range:
        error_checking.assert_is_greater(max_probability, min_probability)
        error_checking.assert_is_leq(max_probability, 1.)

    assert find_highest_probs or find_lowest_probs or find_probs_in_range

    # Find input files.
    date_strings = []
    prediction_file_names = []

    for k in range(NUM_RADARS):
        if len(date_strings) == 0:
            prediction_file_names = prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                radar_number=k, prefer_zipped=False, allow_other_format=True,
                raise_error_if_any_missing=False,
                raise_error_if_all_missing=k > 0
            )

            if len(prediction_file_names) == 0:
                continue

            date_strings = [
                prediction_io.file_name_to_date(f)
                for f in prediction_file_names
            ]
        else:
            prediction_file_names += [
                prediction_io.find_file(
                    top_directory_name=top_prediction_dir_name,
                    valid_date_string=d, radar_number=k,
                    prefer_zipped=False, allow_other_format=True,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

    # Do actual stuff.
    example_probs = numpy.array([], dtype=float)
    example_times_unix_sec = numpy.array([], dtype=int)
    example_radar_numbers = numpy.array([], dtype=int)

    for this_file_name in prediction_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = prediction_io.read_file(this_file_name)

        these_probs = prediction_io.get_mean_predictions(this_prediction_dict)[
            :, grid_row, grid_column
        ]

        these_radar_numbers = numpy.full(
            len(these_probs),
            prediction_io.file_name_to_radar_number(this_file_name),
            dtype=int
        )

        example_probs = numpy.concatenate((example_probs, these_probs))
        example_times_unix_sec = numpy.concatenate((
            example_times_unix_sec,
            this_prediction_dict[prediction_io.VALID_TIMES_KEY]
        ))
        example_radar_numbers = numpy.concatenate((
            example_radar_numbers, these_radar_numbers
        ))

    print(SEPARATOR_STRING)

    # Find interesting examples.
    if find_highest_probs:
        sort_indices = numpy.argsort(-1 * example_probs)
        good_indices = sort_indices[:num_examples]
    elif find_lowest_probs:
        sort_indices = numpy.argsort(example_probs)
        good_indices = sort_indices[:num_examples]
    else:
        good_indices = numpy.where(numpy.logical_and(
            example_probs >= min_probability, example_probs <= max_probability
        ))[0]

        example_probs = example_probs[good_indices]
        example_times_unix_sec = example_times_unix_sec[good_indices]
        example_radar_numbers = example_radar_numbers[good_indices]

        median_prob = (min_probability + max_probability) / 2
        sort_indices = numpy.argsort(
            numpy.absolute(example_probs - median_prob)
        )
        good_indices = sort_indices[:num_examples]

    example_probs = example_probs[good_indices]
    example_times_unix_sec = example_times_unix_sec[good_indices]
    example_radar_numbers = example_radar_numbers[good_indices]
    example_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in example_times_unix_sec
    ]

    num_examples = len(example_probs)

    for i in range(num_examples):
        print((
            '{0:d}th interesting example ... probability = {1:.4f} ... '
            'time = {2:s} ... radar number = {3:d}'
        ).format(
            i + 1, example_probs[i], example_time_strings[i],
            example_radar_numbers[i]
        ))

    print(SEPARATOR_STRING)
    print('Writing results to: "{0:s}"...'.format(output_file_name))

    example_io.write_example_ids(
        netcdf_file_name=output_file_name,
        valid_times_unix_sec=example_times_unix_sec,
        radar_numbers=example_radar_numbers
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME
        ),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        grid_row=getattr(INPUT_ARG_OBJECT, GRID_ROW_ARG_NAME),
        grid_column=getattr(INPUT_ARG_OBJECT, GRID_COLUMN_ARG_NAME),
        find_highest_probs=bool(
            getattr(INPUT_ARG_OBJECT, FIND_HIGHEST_PROBS_ARG_NAME)
        ),
        find_lowest_probs=bool(
            getattr(INPUT_ARG_OBJECT, FIND_LOWEST_PROBS_ARG_NAME)
        ),
        min_probability=getattr(INPUT_ARG_OBJECT, MIN_PROB_ARG_NAME),
        max_probability=getattr(INPUT_ARG_OBJECT, MAX_PROB_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
