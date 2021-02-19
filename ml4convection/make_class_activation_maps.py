"""Makes class-activation maps."""

import os
import sys
import argparse
import numpy
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gradcam
import neural_net

tensorflow.compat.v1.disable_eager_execution()

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
VALID_DATES_ARG_NAME = 'valid_date_strings'
ACTIVATION_LAYER_ARG_NAME = 'activation_layer_name'
OUTPUT_LAYER_ARG_NAME = 'output_layer_name'
OUTPUT_ROW_ARG_NAME = 'output_row'
OUTPUT_COLUMN_ARG_NAME = 'output_column'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained model for which to make class-activation maps.  '
    'Will be read by `neural_net.read_model`.'
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
ACTIVATION_LAYER_HELP_STRING = 'Name of activation layer.'
OUTPUT_LAYER_HELP_STRING = (
    'Name of output layer.  This layer should output either probabilities '
    '(activation outputs) or pseudo-probabilities (activation inputs).'
)
OUTPUT_ROW_HELP_STRING = (
    'Class activation will be computed with respect to output at this grid row '
    '(non-negative integer).'
)
OUTPUT_COLUMN_HELP_STRING = (
    'Class activation will be computed with respect to output at this grid '
    'column (non-negative integer).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`gradcam.write_file`, to exact locations determined by '
    '`gradcam.find_file`.'
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
    '--' + ACTIVATION_LAYER_ARG_NAME, type=str, required=True,
    help=ACTIVATION_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_LAYER_ARG_NAME, type=str, required=True,
    help=OUTPUT_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_ROW_ARG_NAME, type=int, required=True,
    help=OUTPUT_ROW_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_COLUMN_ARG_NAME, type=int, required=True,
    help=OUTPUT_COLUMN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _make_cams_one_day(
        model_object, data_option_dict, valid_date_string, model_file_name,
        activation_layer_name, output_layer_name, output_row, output_column,
        output_dir_name):
    """Computes class-activation maps for one day.

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param data_option_dict: See doc for `neural_net.create_data_partial_grids`.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param model_file_name: See documentation at top of file.
    :param activation_layer_name: Same.
    :param output_layer_name: Same.
    :param output_row: Same.
    :param output_column: Same.
    :param output_dir_name: Same.
    """

    data_option_dict[neural_net.VALID_DATE_KEY] = valid_date_string
    data_dict_by_radar = neural_net.create_data_partial_grids(
        option_dict=data_option_dict, return_coords=True
    )

    if data_dict_by_radar is None:
        print(MINOR_SEPARATOR_STRING)
        return

    num_radars = len(data_dict_by_radar)
    output_neuron_indices = numpy.array([output_row, output_column], dtype=int)

    for k in range(num_radars):
        if len(list(data_dict_by_radar[k].keys())) == 0:
            continue

        print(MINOR_SEPARATOR_STRING)

        print((
            'Computing class-activation maps for radar {0:d} of {1:d}, '
            'activation layer "{2:s}", output neuron {3:s} in layer "{4:s}"...'
        ).format(
            k + 1, num_radars, activation_layer_name,
            str(output_neuron_indices), output_layer_name
        ))

        predictor_matrix = (
            data_dict_by_radar[k][neural_net.PREDICTOR_MATRIX_KEY]
        )
        num_examples = predictor_matrix.shape[0]
        class_activation_matrix = numpy.full(
            predictor_matrix.shape[:3], numpy.nan
        )

        for i in range(num_examples):
            if numpy.mod(i, 10) == 0:
                print('Have computed CAM for {0:d} of {1:d} examples...'.format(
                    i, num_examples
                ))

            class_activation_matrix[i, ...] = gradcam.run_gradcam(
                model_object=model_object,
                predictor_matrix=predictor_matrix[i, ...],
                activation_layer_name=activation_layer_name,
                output_layer_name=output_layer_name,
                output_row=output_row, output_column=output_column
            )

        print('Have computed CAM for all {0:d} examples!'.format(num_examples))

        output_file_name = gradcam.find_file(
            top_directory_name=output_dir_name,
            valid_date_string=valid_date_string, radar_number=k,
            raise_error_if_missing=False
        )

        print('Writing class-activation maps to: "{0:s}"...'.format(
            output_file_name
        ))
        gradcam.write_file(
            netcdf_file_name=output_file_name,
            class_activation_matrix=class_activation_matrix,
            valid_times_unix_sec=
            data_dict_by_radar[k][neural_net.VALID_TIMES_KEY],
            latitudes_deg_n=data_dict_by_radar[k][neural_net.LATITUDES_KEY],
            longitudes_deg_e=data_dict_by_radar[k][neural_net.LONGITUDES_KEY],
            model_file_name=model_file_name,
            activation_layer_name=activation_layer_name,
            output_layer_name=output_layer_name,
            output_row=output_row, output_column=output_column
        )


def _run(model_file_name, top_predictor_dir_name, top_target_dir_name,
         valid_date_strings, activation_layer_name, output_layer_name,
         output_row, output_column, output_dir_name):
    """Makes class-activation maps.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param valid_date_strings: Same.
    :param activation_layer_name: Same.
    :param output_layer_name: Same.
    :param output_row: Same.
    :param output_column: Same.
    :param output_dir_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)
    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
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
        neural_net.ADD_COORDS_KEY:
            training_option_dict[neural_net.ADD_COORDS_KEY]
    }

    for this_date_string in valid_date_strings:
        print(SEPARATOR_STRING)

        _make_cams_one_day(
            model_object=model_object, data_option_dict=data_option_dict,
            valid_date_string=this_date_string, model_file_name=model_file_name,
            activation_layer_name=activation_layer_name,
            output_layer_name=output_layer_name, output_row=output_row,
            output_column=output_column, output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        valid_date_strings=getattr(INPUT_ARG_OBJECT, VALID_DATES_ARG_NAME),
        activation_layer_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_LAYER_ARG_NAME
        ),
        output_layer_name=getattr(INPUT_ARG_OBJECT, OUTPUT_LAYER_ARG_NAME),
        output_row=getattr(INPUT_ARG_OBJECT, OUTPUT_ROW_ARG_NAME),
        output_column=getattr(INPUT_ARG_OBJECT, OUTPUT_COLUMN_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
