"""Applies trained neural net in inference mode."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import prediction_io
import normalization
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_EXAMPLES_PER_BATCH = 32

MODEL_FILE_ARG_NAME = 'input_model_file_name'
SATELLITE_DIR_ARG_NAME = 'input_satellite_dir_name'
RADAR_DIR_ARG_NAME = 'input_radar_dir_name'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
SATELLITE_DIR_HELP_STRING = (
    'Name of top-level directory with satellite data (predictors).  Files '
    'therein will be found by `satellite_io.find_file` and read by '
    '`satellite_io.read_file`.'
)
RADAR_DIR_HELP_STRING = (
    'Name of top-level directory with radar data (targets).  Files therein will'
    ' be found by `radar_io.find_file` and read by `radar_io.read_2d_file`.'
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
    '--' + SATELLITE_DIR_ARG_NAME, type=str, required=True,
    help=SATELLITE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_DIR_ARG_NAME, type=str, required=True,
    help=RADAR_DIR_HELP_STRING
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


def _apply_net_one_day(model_object, base_option_dict, valid_date_string,
                       model_file_name, top_output_dir_name):
    """Applies trained neural net to one day.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param base_option_dict: Dictionary with data-processing options.  See doc
        for `neural_net.create_data`.
    :param valid_date_string: Valid date (radar date), in format "yyyymmdd").
    :param model_file_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    """

    option_dict = copy.deepcopy(base_option_dict)
    option_dict[neural_net.VALID_DATE_KEY] = valid_date_string
    data_dict = neural_net.create_data(
        option_dict=option_dict, return_coords=True
    )

    if data_dict is None:
        return

    forecast_probability_matrix = neural_net.apply_model(
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
        target_matrix=data_dict[neural_net.TARGET_MATRIX_KEY],
        forecast_probability_matrix=forecast_probability_matrix,
        valid_times_unix_sec=data_dict[neural_net.VALID_TIMES_KEY],
        latitudes_deg_n=data_dict[neural_net.LATITUDES_KEY],
        longitudes_deg_e=data_dict[neural_net.LONGITUDES_KEY],
        model_file_name=model_file_name
    )


def _run(model_file_name, top_satellite_dir_name, top_radar_dir_name,
         first_valid_date_string, last_valid_date_string, top_output_dir_name):
    """Applies trained neural net in inference mode.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_satellite_dir_name: Same.
    :param top_radar_dir_name: Same.
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
    training_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    normalization_file_name = (
        training_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )

    if normalization_file_name is None:
        norm_dict_for_count = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            normalization_file_name
        ))
        norm_dict_for_count = (
            normalization.read_file(normalization_file_name)[1]
        )

    base_option_dict = {
        neural_net.SATELLITE_DIRECTORY_KEY: top_satellite_dir_name,
        neural_net.RADAR_DIRECTORY_KEY: top_radar_dir_name,
        neural_net.SPATIAL_DS_FACTOR_KEY:
            training_option_dict[neural_net.SPATIAL_DS_FACTOR_KEY],
        neural_net.BAND_NUMBERS_KEY:
            training_option_dict[neural_net.BAND_NUMBERS_KEY],
        neural_net.LEAD_TIME_KEY:
            training_option_dict[neural_net.LEAD_TIME_KEY],
        neural_net.REFL_THRESHOLD_KEY:
            training_option_dict[neural_net.REFL_THRESHOLD_KEY],
        neural_net.NORMALIZATION_DICT_KEY: norm_dict_for_count,
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
        top_satellite_dir_name=getattr(
            INPUT_ARG_OBJECT, SATELLITE_DIR_ARG_NAME
        ),
        top_radar_dir_name=getattr(INPUT_ARG_OBJECT, RADAR_DIR_ARG_NAME),
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
