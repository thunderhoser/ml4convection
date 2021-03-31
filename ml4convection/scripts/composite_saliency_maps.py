"""Composites saliency maps via probability-matched means (PMM)."""

import os
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import prob_matched_means as pmm
from ml4convection.utils import radar_utils
from ml4convection.machine_learning import saliency
from ml4convection.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

SALIENCY_DIR_ARG_NAME = 'input_saliency_dir_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
MAX_PERCENTILE_ARG_NAME = 'max_pmm_percentile_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

SALIENCY_DIR_HELP_STRING = (
    'Name of top-level directory with input saliency maps (one for each pair of'
    ' valid date and radar).  Files therein will be found by '
    '`saliency.find_file` and read by `saliency.read_file`.'
)
PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors.  Files therein will be found '
    'by `example_io.find_predictor_file` and read by '
    '`example_io.read_predictor_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets.  Files therein will be found by '
    '`example_io.find_target_file` and read by `example_io.read_target_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  This script will composite saliency maps for '
    'valid dates in the period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

MAX_PERCENTILE_HELP_STRING = (
    'Max percentile level for the probability-matched means (PMM).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `saliency.write_composite_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_DIR_ARG_NAME, type=str, required=True,
    help=SALIENCY_DIR_HELP_STRING
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
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _read_saliency_one_file(saliency_file_name, top_predictor_dir_name,
                            top_target_dir_name):
    """Reads saliency maps from one file.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of lag times
    C = number of channels

    :param saliency_file_name: Path to input file.  Will be read by
        `saliency.read_file`.
    :param top_predictor_dir_name: See documentation at top of file.
    :param top_target_dir_name: Same.
    :return: saliency_matrix: numpy array of saliency values
        (either E x M x N x C or E x M x N x T x C).
    :return: denorm_predictor_matrix: numpy array of denormalized predictors
        (same shape as `saliency_matrix`).
    """

    print('Reading data from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency.read_file(saliency_file_name)

    model_metafile_name = neural_net.find_metafile(
        model_file_name=saliency_dict[saliency.MODEL_FILE_KEY],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    valid_date_string = saliency.file_name_to_date(saliency_file_name)
    radar_number = saliency.file_name_to_radar_num(saliency_file_name)

    predictor_option_dict = {
        neural_net.PREDICTOR_DIRECTORY_KEY: top_predictor_dir_name,
        neural_net.TARGET_DIRECTORY_KEY: top_target_dir_name,
        neural_net.VALID_DATE_KEY: valid_date_string,
        neural_net.BAND_NUMBERS_KEY:
            training_option_dict[neural_net.BAND_NUMBERS_KEY],
        neural_net.LEAD_TIME_KEY:
            training_option_dict[neural_net.LEAD_TIME_KEY],
        neural_net.LAG_TIMES_KEY:
            training_option_dict[neural_net.LAG_TIMES_KEY],
        neural_net.INCLUDE_TIME_DIM_KEY: False,
        neural_net.OMIT_NORTH_RADAR_KEY:
            training_option_dict[neural_net.OMIT_NORTH_RADAR_KEY],
        neural_net.NORMALIZE_FLAG_KEY: False,
        neural_net.UNIFORMIZE_FLAG_KEY: False,
        neural_net.ADD_COORDS_KEY:
            training_option_dict[neural_net.ADD_COORDS_KEY]
    }

    predictor_dict = neural_net.create_data_partial_grids(
        option_dict=predictor_option_dict, return_coords=False,
        radar_number=radar_number
    )[radar_number]

    good_indices = numpy.array([
        numpy.where(predictor_dict[neural_net.VALID_TIMES_KEY] == t)[0][0]
        for t in saliency_dict[saliency.VALID_TIMES_KEY]
    ], dtype=int)

    denorm_predictor_matrix = (
        predictor_dict[neural_net.PREDICTOR_MATRIX_KEY][good_indices, ...]
    )

    saliency_matrix = saliency_dict[saliency.SALIENCY_MATRIX_KEY]

    if len(saliency_matrix.shape) == 5:
        num_lag_times = len(training_option_dict[neural_net.LAG_TIMES_KEY])

        saliency_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=saliency_matrix, num_lag_times=num_lag_times
        )
        saliency_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=saliency_matrix, num_lag_times=num_lag_times,
            add_time_dimension=False
        )

    return saliency_matrix, denorm_predictor_matrix


def _run(top_saliency_dir_name, top_predictor_dir_name, top_target_dir_name,
         first_date_string, last_date_string, max_pmm_percentile_level,
         output_file_name):
    """Composites saliency maps via probability-matched means (PMM).

    This is effectively the main method.

    :param top_saliency_dir_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param max_pmm_percentile_level: Same.
    :param output_file_name: Same.
    """

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )
    saliency_file_names = []

    for this_date_string in valid_date_strings:
        for k in range(NUM_RADARS):
            this_file_name = saliency.find_file(
                top_directory_name=top_saliency_dir_name,
                valid_date_string=this_date_string,
                radar_number=k, raise_error_if_missing=False
            )

            if not os.path.isfile(this_file_name):
                continue

            saliency_file_names.append(this_file_name)

    saliency_matrix = None
    denorm_predictor_matrix = None

    for this_file_name in saliency_file_names:
        this_saliency_matrix, this_predictor_matrix = _read_saliency_one_file(
            saliency_file_name=this_file_name,
            top_predictor_dir_name=top_predictor_dir_name,
            top_target_dir_name=top_target_dir_name
        )

        if saliency_matrix is None:
            saliency_matrix = this_saliency_matrix + 0.
            denorm_predictor_matrix = this_predictor_matrix + 0.
        else:
            saliency_matrix = numpy.concatenate(
                (saliency_matrix, this_saliency_matrix), axis=0
            )
            denorm_predictor_matrix = numpy.concatenate(
                (denorm_predictor_matrix, this_predictor_matrix), axis=0
            )

    print(SEPARATOR_STRING)

    mean_saliency_matrix = pmm.run_pmm_many_variables(
        input_matrix=saliency_matrix,
        max_percentile_level=max_pmm_percentile_level
    )
    del saliency_matrix

    mean_denorm_predictor_matrix = pmm.run_pmm_many_variables(
        input_matrix=denorm_predictor_matrix,
        max_percentile_level=max_pmm_percentile_level
    )
    del denorm_predictor_matrix

    print('Reading data from: "{0:s}"...'.format(saliency_file_names[0]))
    first_saliency_dict = saliency.read_file(saliency_file_names[0])

    model_metafile_name = neural_net.find_metafile(
        model_file_name=first_saliency_dict[saliency.MODEL_FILE_KEY],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    if training_option_dict[neural_net.INCLUDE_TIME_DIM_KEY]:
        num_lag_times = len(training_option_dict[neural_net.LAG_TIMES_KEY])

        mean_saliency_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=numpy.expand_dims(mean_saliency_matrix, axis=0),
            num_lag_times=num_lag_times
        )
        mean_saliency_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=mean_saliency_matrix, num_lag_times=num_lag_times,
            add_time_dimension=True
        )[0, ...]

        mean_denorm_predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=
            numpy.expand_dims(mean_denorm_predictor_matrix, axis=0),
            num_lag_times=num_lag_times
        )
        mean_denorm_predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=mean_denorm_predictor_matrix,
            num_lag_times=num_lag_times, add_time_dimension=True
        )[0, ...]

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    saliency.write_composite_file(
        netcdf_file_name=output_file_name,
        mean_saliency_matrix=mean_saliency_matrix,
        mean_denorm_predictor_matrix=mean_denorm_predictor_matrix,
        model_file_name=first_saliency_dict[saliency.MODEL_FILE_KEY],
        is_layer_output=first_saliency_dict[saliency.IS_LAYER_OUTPUT_KEY],
        layer_name=first_saliency_dict[saliency.LAYER_NAME_KEY],
        neuron_indices=first_saliency_dict[saliency.NEURON_INDICES_KEY],
        ideal_activation=first_saliency_dict[saliency.IDEAL_ACTIVATION_KEY],
        multiply_by_input=first_saliency_dict[saliency.MULTIPLY_BY_INPUT_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_saliency_dir_name=getattr(INPUT_ARG_OBJECT, SALIENCY_DIR_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        max_pmm_percentile_level=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
