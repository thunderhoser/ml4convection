"""Contains list of input arguments for training a neural net."""

from ml4convection.io import satellite_io

TRAINING_PREDICTOR_DIR_ARG_NAME = 'training_predictor_dir_name'
TRAINING_TARGET_DIR_ARG_NAME = 'training_target_dir_name'
VALIDN_PREDICTOR_DIR_ARG_NAME = 'validn_predictor_dir_name'
VALIDN_TARGET_DIR_ARG_NAME = 'validn_target_dir_name'
INPUT_MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_MODEL_DIR_ARG_NAME = 'output_model_dir_name'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
LEAD_TIME_ARG_NAME = 'lead_time_seconds'
LAG_TIMES_ARG_NAME = 'lag_times_seconds'
INCLUDE_TIME_DIM_ARG_NAME = 'include_time_dimension'
FIRST_TRAIN_DATE_ARG_NAME = 'first_training_date_string'
LAST_TRAIN_DATE_ARG_NAME = 'last_training_date_string'
FIRST_VALIDN_DATE_ARG_NAME = 'first_validn_date_string'
LAST_VALIDN_DATE_ARG_NAME = 'last_validn_date_string'
NORMALIZE_ARG_NAME = 'normalize'
UNIFORMIZE_ARG_NAME = 'uniformize'
ADD_COORDS_ARG_NAME = 'add_coords'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
MAX_DAILY_EXAMPLES_ARG_NAME = 'max_examples_per_day_in_batch'
USE_PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'
OMIT_NORTH_RADAR_ARG_NAME = 'omit_north_radar'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validn_batches_per_epoch'
PLATEAU_LR_MULTIPLIER_ARG_NAME = 'plateau_lr_multiplier'

TRAINING_PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors for training.  Files therein '
    'will be found by `example_io.find_predictor_file` and read by '
    '`example_io.read_predictor_file`.'
)
TRAINING_TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets for training.  Files therein will'
    ' be found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)

VALIDN_PREDICTOR_DIR_HELP_STRING = (
    'Same as `{0:s}` but for validation (monitoring).'
).format(TRAINING_PREDICTOR_DIR_ARG_NAME)

VALIDN_TARGET_DIR_HELP_STRING = (
    'Same as `{0:s}` but for validation (monitoring).'
).format(TRAINING_TARGET_DIR_ARG_NAME)

INPUT_MODEL_FILE_HELP_STRING = (
    'Path to file with untrained model (defining architecture, optimizer, and '
    'loss function).  Will be read by `neural_net.read_model`.'
)
OUTPUT_MODEL_DIR_HELP_STRING = (
    'Name of output directory.  Model will be saved here.'
)

BAND_NUMBERS_HELP_STRING = (
    'List of band numbers (integers) for satellite data.  Will use only these '
    'spectral bands as predictors.'
)

LEAD_TIME_HELP_STRING = 'Lead time for prediction.'
LAG_TIMES_HELP_STRING = 'Lag times for prediction.'
INCLUDE_TIME_DIM_HELP_STRING = (
    'Boolean flag.  If 1, predictor matrix will have time dimension.  If 0, '
    'times and spectral bands will be combined into channel axis.'
)
TRAIN_DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  The training period will consist of valid times'
    ' (radar times) from `{0:s}`...`{1:s}`.'
).format(FIRST_TRAIN_DATE_ARG_NAME, LAST_TRAIN_DATE_ARG_NAME)

VALIDN_DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  The validation period will consist of valid '
    'times (radar times) from `{0:s}`...`{1:s}`.'
).format(FIRST_VALIDN_DATE_ARG_NAME, LAST_VALIDN_DATE_ARG_NAME)

NORMALIZE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use normalized (unnormalized) predictors.'
)
UNIFORMIZE_HELP_STRING = (
    'Boolean flag.  If True, will convert satellite values to uniform '
    'distribution before normal distribution.  If False, will go directly to '
    'normal distribution.'
)
ADD_COORDS_HELP_STRING = (
    'Boolean flag.  If 1, will use coordinates (lat/long) as predictors.'
)
BATCH_SIZE_HELP_STRING = (
    'Number of examples in each training and validation (monitoring) batch.'
)
MAX_DAILY_EXAMPLES_HELP_STRING = (
    'Max number of examples from the same day in one batch.'
)
USE_PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will train on partial radar-centered (full) '
    'grids.'
)
OMIT_NORTH_RADAR_HELP_STRING = (
    '[used only if {0:s} == 1] Boolean flag.  If 1 (0), will train without '
    '(with) north radar.'
).format(USE_PARTIAL_GRIDS_ARG_NAME)

NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDN_BATCHES_HELP_STRING = (
    'Number of validation (monitoring) batches per epoch.'
)
PLATEAU_LR_MULTIPLIER_HELP_STRING = (
    'Multiplier for learning rate.  Learning rate will be multiplied by this '
    'factor upon plateau in validation performance.'
)


def add_input_args(parser_object):
    """Adds input args to ArgumentParser object.

    :param parser_object: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: parser_object: Same as input but with new args added.
    """

    parser_object.add_argument(
        '--' + TRAINING_PREDICTOR_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_PREDICTOR_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_TARGET_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_TARGET_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDN_PREDICTOR_DIR_ARG_NAME, type=str, required=True,
        help=VALIDN_PREDICTOR_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDN_TARGET_DIR_ARG_NAME, type=str, required=True,
        help=VALIDN_TARGET_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + INPUT_MODEL_FILE_ARG_NAME, type=str, required=True,
        help=INPUT_MODEL_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + OUTPUT_MODEL_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_MODEL_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BAND_NUMBERS_ARG_NAME, type=int, nargs='+', required=False,
        default=satellite_io.BAND_NUMBERS, help=BAND_NUMBERS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LEAD_TIME_ARG_NAME, type=int, required=True,
        help=LEAD_TIME_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LAG_TIMES_ARG_NAME, type=int, nargs='+', required=False,
        default=[0], help=LAG_TIMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + INCLUDE_TIME_DIM_ARG_NAME, type=int, required=True,
        help=INCLUDE_TIME_DIM_HELP_STRING
    )
    parser_object.add_argument(
        '--' + FIRST_TRAIN_DATE_ARG_NAME, type=str, required=False,
        default='20160101', help=TRAIN_DATE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LAST_TRAIN_DATE_ARG_NAME, type=str, required=False,
        default='20161224', help=TRAIN_DATE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + FIRST_VALIDN_DATE_ARG_NAME, type=str, required=False,
        default='20170101', help=VALIDN_DATE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LAST_VALIDN_DATE_ARG_NAME, type=str, required=False,
        default='20171224', help=VALIDN_DATE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NORMALIZE_ARG_NAME, type=int, required=False, default=1,
        help=NORMALIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + UNIFORMIZE_ARG_NAME, type=int, required=False, default=1,
        help=UNIFORMIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + ADD_COORDS_ARG_NAME, type=int, required=False, default=0,
        help=ADD_COORDS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BATCH_SIZE_ARG_NAME, type=int, required=False, default=256,
        help=BATCH_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MAX_DAILY_EXAMPLES_ARG_NAME, type=int, required=False,
        default=64, help=MAX_DAILY_EXAMPLES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + USE_PARTIAL_GRIDS_ARG_NAME, type=int, required=True,
        help=USE_PARTIAL_GRIDS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + OMIT_NORTH_RADAR_ARG_NAME, type=int, required=False,
        help=OMIT_NORTH_RADAR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=1000,
        help=NUM_EPOCHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=64, help=NUM_TRAINING_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_VALIDN_BATCHES_ARG_NAME, type=int, required=False,
        default=32, help=NUM_VALIDN_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PLATEAU_LR_MULTIPLIER_ARG_NAME, type=float, required=False,
        default=0.6, help=PLATEAU_LR_MULTIPLIER_HELP_STRING
    )

    return parser_object
