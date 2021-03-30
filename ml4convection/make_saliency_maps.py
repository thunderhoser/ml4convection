"""Creates saliency maps."""

import os
import sys
import argparse
import numpy
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import saliency
import neural_net

tensorflow.compat.v1.disable_eager_execution()

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = '%Y%m%d'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
VALID_DATES_ARG_NAME = 'valid_date_strings'
VALID_TIMES_ARG_NAME = 'valid_time_strings'
LAYER_ARG_NAME = 'layer_name'
IS_LAYER_OUTPUT_ARG_NAME = 'is_layer_output'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
MULTIPLY_BY_INPUT_ARG_NAME = 'multiply_by_input'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to file with trained model for which to make saliency maps.  Will be'
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
VALID_DATES_HELP_STRING = (
    'List of valid dates for targets (format "yyyymmdd").  If you want to '
    'specify exact times instead, leave this argument alone.'
)
VALID_TIMES_HELP_STRING = (
    'List of valid times for targets (format "yyyy-mm-dd-HHMM").  If you want '
    'to specify dates instead, leave this argument alone.'
)
IS_LAYER_OUTPUT_HELP_STRING = (
    'Boolean flag.  If 1, `{0:s}` is an output layer.'
).format(LAYER_ARG_NAME)

LAYER_HELP_STRING = 'Name of layer with relevant neuron.'
NEURON_INDICES_HELP_STRING = (
    '1-D list with indices of relevant neuron.  Must have length D - 1, where '
    'D = number of dimensions in layer output.  The first dimension is the '
    'batch dimension, which always has length `None` in Keras.'
)
IDEAL_ACTIVATION_HELP_STRING = (
    'Ideal neuron activation, used to define loss function.  The loss function '
    'will be (neuron_activation - ideal_activation)**2.'
)
MULTIPLY_BY_INPUT_HELP_STRING = (
    'Boolean flag.  If 1, saliency will be multiplied by input, so the '
    'interpretation method will actually be input * gradient.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`saliency.write_file`, to exact locations determined by '
    '`saliency.find_file`.'
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
    '--' + VALID_DATES_ARG_NAME, type=str, nargs='+', required=False,
    default=['None'], help=VALID_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIMES_ARG_NAME, type=str, nargs='+', required=False,
    default=['None'], help=VALID_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_ARG_NAME, type=str, required=True, help=LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IS_LAYER_OUTPUT_ARG_NAME, type=int, required=True,
    help=IS_LAYER_OUTPUT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=True,
    help=NEURON_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=True,
    help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MULTIPLY_BY_INPUT_ARG_NAME, type=int, required=False, default=0,
    help=MULTIPLY_BY_INPUT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_saliency_one_day(
        model_object, data_option_dict, valid_date_string, valid_times_unix_sec,
        model_file_name, layer_name, is_layer_output, neuron_indices,
        ideal_activation, multiply_by_input, output_dir_name):
    """Creates saliency maps for one day.

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param data_option_dict: See doc for `neural_net.create_data_partial_grids`.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param valid_times_unix_sec: 1-D numpy array of valid times.  May be None,
        in which case all valid times in the day will be used.
    :param model_file_name: See documentation at top of file.
    :param layer_name: Same.
    :param is_layer_output: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param multiply_by_input: Same.
    :param output_dir_name: Same.
    """

    data_option_dict[neural_net.VALID_DATE_KEY] = valid_date_string
    data_dict_by_radar = neural_net.create_data_partial_grids(
        option_dict=data_option_dict, return_coords=True
    )
    print(MINOR_SEPARATOR_STRING)

    if data_dict_by_radar is None:
        return

    num_radars = len(data_dict_by_radar)

    for k in range(num_radars):
        if len(list(data_dict_by_radar[k].keys())) == 0:
            continue

        print((
            'Computing saliency maps for radar {0:d} of {1:d}, neuron {2:s} of '
            'layer "{3:s}"...'
        ).format(
            k + 1, num_radars, str(neuron_indices), layer_name
        ))

        if valid_times_unix_sec is not None:
            all_times_unix_sec = (
                data_dict_by_radar[k][neural_net.VALID_TIMES_KEY]
            )

            good_indices = numpy.array([
                numpy.where(all_times_unix_sec == t)[0][0]
                for t in valid_times_unix_sec
            ], dtype=int)

            these_keys = [
                neural_net.VALID_TIMES_KEY, neural_net.PREDICTOR_MATRIX_KEY,
                neural_net.TARGET_MATRIX_KEY
            ]

            for this_key in these_keys:
                data_dict_by_radar[k][this_key] = (
                    data_dict_by_radar[k][this_key][good_indices, ...]
                )

        saliency_matrix = saliency.get_saliency_one_neuron(
            model_object=model_object,
            predictor_matrix=
            data_dict_by_radar[k][neural_net.PREDICTOR_MATRIX_KEY],
            layer_name=layer_name, neuron_indices=neuron_indices,
            ideal_activation=ideal_activation,
            multiply_by_input=multiply_by_input
        )

        output_file_name = saliency.find_file(
            top_directory_name=output_dir_name,
            valid_date_string=valid_date_string, radar_number=k,
            raise_error_if_missing=False
        )

        print('Writing saliency maps to: "{0:s}"...'.format(output_file_name))
        saliency.write_file(
            netcdf_file_name=output_file_name, saliency_matrix=saliency_matrix,
            valid_times_unix_sec=
            data_dict_by_radar[k][neural_net.VALID_TIMES_KEY],
            latitudes_deg_n=data_dict_by_radar[k][neural_net.LATITUDES_KEY],
            longitudes_deg_e=data_dict_by_radar[k][neural_net.LONGITUDES_KEY],
            model_file_name=model_file_name, is_layer_output=is_layer_output,
            layer_name=layer_name, neuron_indices=neuron_indices,
            ideal_activation=ideal_activation,
            multiply_by_input=multiply_by_input
        )


def _run(model_file_name, top_predictor_dir_name, top_target_dir_name,
         valid_date_strings, valid_time_strings, layer_name, is_layer_output,
         neuron_indices, ideal_activation, multiply_by_input, output_dir_name):
    """Runs permutation-based importance test.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param valid_date_strings: Same.
    :param valid_time_strings: Same.
    :param layer_name: Same.
    :param is_layer_output: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param multiply_by_input: Same.
    :param output_dir_name: Same.
    """

    if len(valid_date_strings) == 1 and valid_date_strings[0] == 'None':
        valid_date_strings = None
    if len(valid_time_strings) == 1 and valid_time_strings[0] == 'None':
        valid_time_strings = None

    if valid_date_strings is None:
        valid_times_unix_sec = numpy.array([
            time_conversion.string_to_unix_sec(t, TIME_FORMAT)
            for t in valid_time_strings
        ], dtype=int)

        all_valid_date_strings = [
            time_conversion.unix_sec_to_string(t, DATE_FORMAT)
            for t in valid_times_unix_sec
        ]

        valid_date_strings = list(set(all_valid_date_strings))
        valid_date_strings.sort()
    else:
        valid_times_unix_sec = None
        all_valid_date_strings = None

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

        if valid_times_unix_sec is None:
            these_times_unix_sec = None
        else:
            these_indices = numpy.where(
                numpy.array(all_valid_date_strings) == this_date_string
            )[0]

            these_times_unix_sec = valid_times_unix_sec[these_indices]

        _get_saliency_one_day(
            model_object=model_object, data_option_dict=data_option_dict,
            valid_date_string=this_date_string,
            valid_times_unix_sec=these_times_unix_sec,
            model_file_name=model_file_name,
            layer_name=layer_name, is_layer_output=is_layer_output,
            neuron_indices=neuron_indices, ideal_activation=ideal_activation,
            multiply_by_input=multiply_by_input, output_dir_name=output_dir_name
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
        valid_time_strings=getattr(INPUT_ARG_OBJECT, VALID_TIMES_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_ARG_NAME),
        is_layer_output=bool(
            getattr(INPUT_ARG_OBJECT, IS_LAYER_OUTPUT_ARG_NAME)
        ),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int
        ),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        multiply_by_input=bool(
            getattr(INPUT_ARG_OBJECT, MULTIPLY_BY_INPUT_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
