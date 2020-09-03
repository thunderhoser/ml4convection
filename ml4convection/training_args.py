"""Contains list of input arguments for training a neural net."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import satellite_io

TRAINING_SATELLITE_DIR_ARG_NAME = 'training_satellite_dir_name'
TRAINING_RADAR_DIR_ARG_NAME = 'training_radar_dir_name'
VALIDN_SATELLITE_DIR_ARG_NAME = 'validn_satellite_dir_name'
VALIDN_RADAR_DIR_ARG_NAME = 'validn_radar_dir_name'
INPUT_MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_MODEL_DIR_ARG_NAME = 'output_model_dir_name'
SPATIAL_DS_FACTOR_ARG_NAME = 'spatial_downsampling_factor'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
REFL_THRESHOLD_ARG_NAME = 'reflectivity_threshold_dbz'
LEAD_TIME_ARG_NAME = 'lead_time_seconds'
FIRST_TRAIN_DATE_ARG_NAME = 'first_training_date_string'
LAST_TRAIN_DATE_ARG_NAME = 'last_training_date_string'
FIRST_VALIDN_DATE_ARG_NAME = 'first_validn_date_string'
LAST_VALIDN_DATE_ARG_NAME = 'last_validn_date_string'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
UNIFORMIZE_ARG_NAME = 'uniformize'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
MAX_DAILY_EXAMPLES_ARG_NAME = 'max_examples_per_day_in_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validn_batches_per_epoch'
PLATEAU_LR_MULTIPLIER_ARG_NAME = 'plateau_lr_multiplier'

TRAINING_SATELLITE_DIR_HELP_STRING = (
    'Name of top-level directory with satellite data (predictors) for training.'
    '  Files therein will be found by `satellite_io.find_file` and read by '
    '`satellite_io.read_file`.'
)
TRAINING_RADAR_DIR_HELP_STRING = (
    'Name of top-level directory with radar data (targets) for training.  Files'
    ' therein will be found by `radar_io.find_file` and read by '
    '`radar_io.read_2d_file`.'
)
VALIDN_SATELLITE_DIR_HELP_STRING = (
    'Same as `{0:s}` but for validation (monitoring) data.'
).format(TRAINING_SATELLITE_DIR_ARG_NAME)

VALIDN_RADAR_DIR_HELP_STRING = (
    'Same as `{0:s}` but for validation (monitoring) data.'
).format(TRAINING_RADAR_DIR_ARG_NAME)

INPUT_MODEL_FILE_HELP_STRING = (
    'Path to file with untrained model (defining architecture, optimizer, and '
    'loss function).  Will be read by `neural_net.read_model`.'
)
OUTPUT_MODEL_DIR_HELP_STRING = (
    'Name of output directory.  Model will be saved here.'
)
SPATIAL_DS_FACTOR_HELP_STRING = (
    'Downsampling factor, used to coarsen spatial resolution.  If you do not '
    'want to coarsen spatial resolution, make this 1.'
)
BAND_NUMBERS_HELP_STRING = (
    'List of band numbers (integers) for satellite data.  Will use only these '
    'spectral bands as predictors.'
)
REFL_THRESHOLD_HELP_STRING = (
    'Reflectivity threshold for convection.  Only grid cells with composite '
    '(column-max) reflectivity >= threshold will be called convective.'
)
LEAD_TIME_HELP_STRING = 'Lead time for prediction.'

TRAIN_DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  The training period will consist of valid times'
    ' (radar times) from `{0:s}`...`{1:s}`.'
).format(FIRST_TRAIN_DATE_ARG_NAME, LAST_TRAIN_DATE_ARG_NAME)

VALIDN_DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  The validation period will consist of valid '
    'times (radar times) from `{0:s}`...`{1:s}`.'
).format(FIRST_VALIDN_DATE_ARG_NAME, LAST_VALIDN_DATE_ARG_NAME)

NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  Will be read by `normalization.read_file`.'
)
UNIFORMIZE_HELP_STRING = (
    'Boolean flag.  If True, will convert satellite values to uniform '
    'distribution before normal distribution.  If False, will go directly to '
    'normal distribution.'
)
BATCH_SIZE_HELP_STRING = (
    'Number of examples in each training and validation (monitoring) batch.'
)
MAX_DAILY_EXAMPLES_HELP_STRING = (
    'Max number of examples from the same day in one batch.'
)
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
        '--' + TRAINING_SATELLITE_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_SATELLITE_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TRAINING_RADAR_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_RADAR_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDN_SATELLITE_DIR_ARG_NAME, type=str, required=True,
        help=VALIDN_SATELLITE_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDN_RADAR_DIR_ARG_NAME, type=str, required=True,
        help=VALIDN_RADAR_DIR_HELP_STRING
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
        '--' + SPATIAL_DS_FACTOR_ARG_NAME, type=int, required=False, default=1,
        help=SPATIAL_DS_FACTOR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BAND_NUMBERS_ARG_NAME, type=int, nargs='+', required=False,
        default=satellite_io.BAND_NUMBERS, help=BAND_NUMBERS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + REFL_THRESHOLD_ARG_NAME, type=float, required=False,
        default=35., help=REFL_THRESHOLD_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LEAD_TIME_ARG_NAME, type=int, required=True,
        help=LEAD_TIME_HELP_STRING
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
        '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
        help=NORMALIZATION_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + UNIFORMIZE_ARG_NAME, type=int, required=False, default=0,
        help=UNIFORMIZE_HELP_STRING
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
