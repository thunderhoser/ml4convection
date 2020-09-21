"""Contains list of input arguments for training a neural net."""

from ml4convection.io import satellite_io

USE_PREPROCESSED_FILES_ARG_NAME = 'use_preprocessed_files'
TRAINING_PREDICTOR_DIR_ARG_NAME = 'training_predictor_dir_name'
TRAINING_TARGET_DIR_ARG_NAME = 'training_target_dir_name'
VALIDN_PREDICTOR_DIR_ARG_NAME = 'validn_predictor_dir_name'
VALIDN_TARGET_DIR_ARG_NAME = 'validn_target_dir_name'
INPUT_MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_MODEL_DIR_ARG_NAME = 'output_model_dir_name'
SPATIAL_DS_FACTOR_ARG_NAME = 'spatial_downsampling_factor'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
LEAD_TIME_ARG_NAME = 'lead_time_seconds'
FIRST_TRAIN_DATE_ARG_NAME = 'first_training_date_string'
LAST_TRAIN_DATE_ARG_NAME = 'last_training_date_string'
FIRST_VALIDN_DATE_ARG_NAME = 'first_validn_date_string'
LAST_VALIDN_DATE_ARG_NAME = 'last_validn_date_string'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
NORMALIZE_ARG_NAME = 'normalize'
UNIFORMIZE_ARG_NAME = 'uniformize'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
MAX_DAILY_EXAMPLES_ARG_NAME = 'max_examples_per_day_in_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validn_batches_per_epoch'
PLATEAU_LR_MULTIPLIER_ARG_NAME = 'plateau_lr_multiplier'

USE_PREPROCESSED_FILES_HELP_STRING = (
    'Boolean flag.  If 1, will use pre-processed files, readable by '
    '`example_io.read_predictor_file` and `example_io.read_target_file`.  If 0,'
    ' will use raw files, readable by `satellite_io.read_file` and '
    '`radar_io.read_2d_file`.'
)
TRAINING_PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors for training.  For more '
    'details, see doc for argument `{0:s}`.'
).format(USE_PREPROCESSED_FILES_ARG_NAME)

TRAINING_TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets for training.  For more details, '
    'see doc for argument `{0:s}`.'
).format(USE_PREPROCESSED_FILES_ARG_NAME)

VALIDN_PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors for validation (monitoring).  '
    'For more details, see doc for argument `{0:s}`.'
).format(USE_PREPROCESSED_FILES_ARG_NAME)

VALIDN_TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets for validation (monitoring).  For'
    ' more details, see doc for argument `{0:s}`.'
).format(USE_PREPROCESSED_FILES_ARG_NAME)

INPUT_MODEL_FILE_HELP_STRING = (
    'Path to file with untrained model (defining architecture, optimizer, and '
    'loss function).  Will be read by `neural_net.read_model`.'
)
OUTPUT_MODEL_DIR_HELP_STRING = (
    'Name of output directory.  Model will be saved here.'
)
SPATIAL_DS_FACTOR_HELP_STRING = (
    '[used only if `{0:s} == 0`] '
    'Downsampling factor, used to coarsen spatial resolution.  If you do not '
    'want to coarsen spatial resolution, make this 1.'
).format(USE_PREPROCESSED_FILES_ARG_NAME)

BAND_NUMBERS_HELP_STRING = (
    'List of band numbers (integers) for satellite data.  Will use only these '
    'spectral bands as predictors.'
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
    '[used only if `{0:s} == 0`] '
    'Path to normalization file.  Will be read by `normalization.read_file`.'
).format(USE_PREPROCESSED_FILES_ARG_NAME)

NORMALIZE_HELP_STRING = (
    '[used only if `{0:s} == 1`] '
    'Boolean flag.  If 1 (0), will use normalized (unnormalized) predictors.'
).format(USE_PREPROCESSED_FILES_ARG_NAME)

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
        '--' + USE_PREPROCESSED_FILES_ARG_NAME, type=int, required=False,
        default=1, help=USE_PREPROCESSED_FILES_HELP_STRING
    )
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
        '--' + SPATIAL_DS_FACTOR_ARG_NAME, type=int, required=False, default=1,
        help=SPATIAL_DS_FACTOR_HELP_STRING
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
        '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=False,
        default='', help=NORMALIZATION_FILE_HELP_STRING
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
