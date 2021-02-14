"""Runs permutation-based importance test."""

import argparse
from ml4convection.machine_learning import permutation
from ml4convection.machine_learning import neural_net

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
VALID_DATES_ARG_NAME = 'valid_date_strings'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_px'
SQUARE_FILTER_ARG_NAME = 'square_fss_filter'
RUN_BACKWARDS_ARG_NAME = 'run_backwards_test'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained model on which to run permutation test.  Will be'
    ' read by `neural_net.read_model`.'
)
PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors to use.  Files therein will be'
    ' found by `example_io.find_predictor_file` and read by '
    '`example_io.read_predictor_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets to use.  Files therein will be '
    'found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
VALID_DATES_HELP_STRING = 'List of valid dates for targets (format "yyyymmdd").'
NUM_BOOTSTRAP_REPS_HELP_STRING = (
    'Number of replicates for bootstrapping cost function.'
)
MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance (pixels) for negative-FSS cost function.'
)
SQUARE_FILTER_HELP_STRING = (
    'Boolean flag.  If 1, will square filter for negative-FSS cost function.  '
    'If 0, will use circular filter.'
)
RUN_BACKWARDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will run backwards (forward) versions of test.  '
    'This script always runs both single- and multi-pass tests.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `permutation.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTOR_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=VALID_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_REPS_ARG_NAME, type=int, required=False,
    default=permutation.DEFAULT_NUM_BOOTSTRAP_REPS,
    help=NUM_BOOTSTRAP_REPS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCE_ARG_NAME, type=float, required=False,
    default=4, help=MATCHING_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SQUARE_FILTER_ARG_NAME, type=int, required=False,
    default=1, help=SQUARE_FILTER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RUN_BACKWARDS_ARG_NAME, type=int, required=False,
    default=0, help=RUN_BACKWARDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, top_predictor_dir_name, top_target_dir_name,
         valid_date_strings, num_bootstrap_reps, matching_distance_px,
         square_fss_filter, run_backwards_test, output_file_name):
    """Runs permutation-based importance test.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param valid_date_strings: Same.
    :param num_bootstrap_reps: Same.
    :param matching_distance_px: Same.
    :param square_fss_filter: Same.
    :param run_backwards_test: Same.
    :param output_file_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)

    cost_function = permutation.make_fss_cost_function(
        matching_distance_px=matching_distance_px,
        square_filter=square_fss_filter, model_metadata_dict=model_metadata_dict
    )

    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    data_option_dict = {
        neural_net.PREDICTOR_DIRECTORY_KEY: top_predictor_dir_name,
        neural_net.TARGET_DIRECTORY_KEY: top_target_dir_name,
        neural_net.BAND_NUMBERS_KEY:
            training_option_dict[neural_net.BAND_NUMBERS_KEY],
        neural_net.LEAD_TIME_KEY:
            training_option_dict[neural_net.LEAD_TIME_KEY],
        neural_net.LAG_TIMES_KEY:
            training_option_dict[neural_net.LAG_TIMES_KEY],
        neural_net.INCLUDE_TIME_DIM_KEY:
            training_option_dict[neural_net.INCLUDE_TIME_DIM_KEY],
        neural_net.OMIT_NORTH_RADAR_KEY:
            training_option_dict[neural_net.OMIT_NORTH_RADAR_KEY],
        neural_net.NORMALIZE_FLAG_KEY:
            training_option_dict[neural_net.NORMALIZE_FLAG_KEY],
        neural_net.UNIFORMIZE_FLAG_KEY:
            training_option_dict[neural_net.UNIFORMIZE_FLAG_KEY],
        neural_net.ADD_COORDS_KEY: False
    }

    result_dict = permutation.run_forward_test(
        model_object=model_object, data_option_dict=data_option_dict,
        valid_date_strings=valid_date_strings, cost_function=cost_function,
        num_bootstrap_reps=num_bootstrap_reps
    )
    print(result_dict)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        valid_date_strings=getattr(INPUT_ARG_OBJECT, VALID_DATES_ARG_NAME),
        num_bootstrap_reps=getattr(
            INPUT_ARG_OBJECT, NUM_BOOTSTRAP_REPS_ARG_NAME
        ),
        matching_distance_px=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME
        ),
        square_fss_filter=bool(
            getattr(INPUT_ARG_OBJECT, SQUARE_FILTER_ARG_NAME)
        ),
        run_backwards_test=bool(
            getattr(INPUT_ARG_OBJECT, RUN_BACKWARDS_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
