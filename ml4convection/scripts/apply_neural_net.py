"""Applies trained neural net in inference mode."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4convection.io import prediction_io
from ml4convection.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_EXAMPLES_PER_BATCH = 32

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
OVERLAP_SIZE_ARG_NAME = 'overlap_size_px'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  If model was trained with '
    'pre-processed files, this directory must contain pre-processed files, '
    'readable by `example_io.read_predictor_file`.  If model was trained with '
    'raw files, this directory must contain raw files, readable by '
    '`satellite_io.read_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets.  If model was trained with '
    'pre-processed files, this directory must contain pre-processed files, '
    'readable by `example_io.read_target_file`.  If model was trained with raw '
    'files, this directory must contain raw files, readable by '
    '`radar_io.read_2d_file`.'
)
OVERLAP_SIZE_HELP_STRING = (
    '[used only if neural net was trained on partial grids] Amount of overlap '
    '(in pixels) between adjacent partial grids.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  The model will be applied to valid times (radar'
    ' times) from the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Predictions will be written by '
    '`prediction_io.write_file`, to exact locations therein determined by '
    '`prediction_io.find_file`.'
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
    '--' + OVERLAP_SIZE_ARG_NAME, type=int, required=False, default=-1,
    help=OVERLAP_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _apply_net_one_day(
        model_object, base_option_dict, use_partial_grids, overlap_size_px,
        valid_date_string, model_file_name, top_output_dir_name):
    """Applies trained neural net to one day.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param base_option_dict: Dictionary with data-processing options.  See doc
        for `neural_net.create_data`.
    :param use_partial_grids: Boolean flag.  If True (False), model was trained
        on partial (full) grids.
    :param overlap_size_px: See documentation at top of file.
    :param valid_date_string: Valid date (radar date), in format "yyyymmdd").
    :param model_file_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    option_dict = copy.deepcopy(base_option_dict)
    option_dict[neural_net.VALID_DATE_KEY] = valid_date_string

    data_dict = neural_net.create_data_full_grid(
        option_dict=option_dict, return_coords=True
    )

    if data_dict is None:
        return

    if use_partial_grids:
        forecast_probability_matrix = neural_net.apply_model_partial_grids(
            model_object=model_object,
            predictor_matrix=data_dict[neural_net.PREDICTOR_MATRIX_KEY],
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            overlap_size_px=overlap_size_px, verbose=True
        )
    else:
        forecast_probability_matrix = neural_net.apply_model_full_grid(
            model_object=model_object,
            predictor_matrix=data_dict[neural_net.PREDICTOR_MATRIX_KEY],
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
        )

    these_percentiles = numpy.array([90, 95, 96, 97, 98, 99, 100], dtype=float)
    print(numpy.percentile(forecast_probability_matrix, these_percentiles))

    output_file_name = prediction_io.find_file(
        top_directory_name=top_output_dir_name,
        valid_date_string=valid_date_string, raise_error_if_missing=False
    )

    print('Writing predictions to: "{0:s}"...'.format(output_file_name))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        target_matrix=data_dict[neural_net.TARGET_MATRIX_KEY][..., 0],
        forecast_probability_matrix=forecast_probability_matrix,
        valid_times_unix_sec=data_dict[neural_net.VALID_TIMES_KEY],
        latitudes_deg_n=data_dict[neural_net.LATITUDES_KEY],
        longitudes_deg_e=data_dict[neural_net.LONGITUDES_KEY],
        model_file_name=model_file_name
    )


def _run(model_file_name, top_predictor_dir_name, top_target_dir_name,
         overlap_size_px, first_valid_date_string, last_valid_date_string,
         top_output_dir_name):
    """Applies trained neural net in inference mode.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param overlap_size_px: Same.
    :param first_valid_date_string: Same.
    :param last_valid_date_string: Same.
    :param top_output_dir_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)
    use_partial_grids = metadata_dict[neural_net.USE_PARTIAL_GRIDS_KEY]
    training_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    base_option_dict = {
        neural_net.PREDICTOR_DIRECTORY_KEY: top_predictor_dir_name,
        neural_net.TARGET_DIRECTORY_KEY: top_target_dir_name,
        neural_net.BAND_NUMBERS_KEY:
            training_option_dict[neural_net.BAND_NUMBERS_KEY],
        neural_net.LEAD_TIME_KEY:
            training_option_dict[neural_net.LEAD_TIME_KEY],
        neural_net.LAG_TIMES_KEY:
            training_option_dict[neural_net.LAG_TIMES_KEY],
        neural_net.NORMALIZE_FLAG_KEY:
            training_option_dict[neural_net.NORMALIZE_FLAG_KEY],
        neural_net.UNIFORMIZE_FLAG_KEY:
            training_option_dict[neural_net.UNIFORMIZE_FLAG_KEY]
    }

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_valid_date_string, last_valid_date_string
    )
    print(SEPARATOR_STRING)

    for i in range(len(valid_date_strings)):
        _apply_net_one_day(
            model_object=model_object, base_option_dict=base_option_dict,
            use_partial_grids=use_partial_grids,
            overlap_size_px=overlap_size_px,
            valid_date_string=valid_date_strings[i],
            model_file_name=model_file_name,
            top_output_dir_name=top_output_dir_name
        )

        if i != len(valid_date_strings) - 1:
            print('\n')


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        overlap_size_px=getattr(INPUT_ARG_OBJECT, OVERLAP_SIZE_ARG_NAME),
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
