"""Computes spread vs. skill to determine quality of uncertainty estimates."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import uq_evaluation
import radar_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
TIME_INTERVAL_ARG_NAME = 'time_interval_steps'
MATCHING_DISTANCES_ARG_NAME = 'matching_distances_px'
BIN_EDGES_ARG_NAME = 'bin_edge_prediction_stdevs'
USE_FANCY_QUANTILES_ARG_NAME = 'use_fancy_quantile_method_for_stdev'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will use predictions for all days in the period'
    ' `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

TIME_INTERVAL_HELP_STRING = (
    'Will use predictions at every [k]th time step, where k = `{0:s}`.'
).format(TIME_INTERVAL_ARG_NAME)

MATCHING_DISTANCES_HELP_STRING = (
    'List of half-widths (pixels) for mean-smoother used in evaluation.  The '
    'spread-skill plot will be computed once for each half-width.'
)
BIN_EDGES_HELP_STRING = (
    'List of bin cutoffs -- ranging from (0, 1) -- each a standard deviation '
    'of the predictive distribution.  This script will automatically use 0 and '
    '1 as the lowest and highest cutoffs.'
)
USE_FANCY_QUANTILES_HELP_STRING = (
    '[used only if model does quantile regression] Boolean flag.  If 1, will '
    'use fancy quantile-based method to compute standard deviation of '
    'predictive distribution.  If False, will treat each quantile-based '
    'estimate as a Monte Carlo estimate.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  For each matching distance, one file will be '
    'written here by `uq_evaluation.write_spread_vs_skill`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TIME_INTERVAL_ARG_NAME, type=int, required=False, default=1,
    help=TIME_INTERVAL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCES_ARG_NAME, type=float, nargs='+', required=True,
    help=MATCHING_DISTANCES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BIN_EDGES_ARG_NAME, type=float, nargs='+', required=True,
    help=BIN_EDGES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_FANCY_QUANTILES_ARG_NAME, type=int, required=True, default=1,
    help=USE_FANCY_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         time_interval_steps, matching_distances_px, bin_edge_prediction_stdevs,
         use_fancy_quantile_method_for_stdev, output_dir_name):
    """Runs discard test to determine quality of uncertainty estimates.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param time_interval_steps: Same.
    :param matching_distances_px: Same.
    :param bin_edge_prediction_stdevs: Same.
    :param use_fancy_quantile_method_for_stdev: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    error_checking.assert_is_geq(time_interval_steps, 1)

    date_strings = []
    prediction_file_names = []

    for j in range(NUM_RADARS):
        if len(date_strings) == 0:
            prediction_file_names = prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                radar_number=j, prefer_zipped=False, allow_other_format=True,
                raise_error_if_any_missing=False,
                raise_error_if_all_missing=j > 0
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
                    valid_date_string=d, radar_number=j,
                    prefer_zipped=False, allow_other_format=True,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

    forecast_prob_matrices = []
    target_matrices = []
    quantile_levels = None
    model_file_name = ''

    for this_file_name in prediction_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        prediction_dict = prediction_io.read_file(this_file_name)

        num_times = len(prediction_dict[prediction_io.VALID_TIMES_KEY])
        desired_indices = numpy.linspace(
            0, num_times - 1, num=num_times, dtype=int
        )
        desired_indices = desired_indices[::time_interval_steps]

        prediction_dict = prediction_io.subset_by_index(
            prediction_dict=prediction_dict,
            desired_indices=desired_indices
        )
        forecast_prob_matrices.append(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY]
        )
        target_matrices.append(
            prediction_dict[prediction_io.TARGET_MATRIX_KEY]
        )

        quantile_levels = prediction_dict[prediction_io.QUANTILE_LEVELS_KEY]
        model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]

    prediction_dict = {
        prediction_io.PROBABILITY_MATRIX_KEY:
            numpy.concatenate(forecast_prob_matrices, axis=0),
        prediction_io.TARGET_MATRIX_KEY:
            numpy.concatenate(target_matrices, axis=0),
        prediction_io.QUANTILE_LEVELS_KEY: quantile_levels
    }
    del forecast_prob_matrices
    del target_matrices

    model_metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    print(SEPARATOR_STRING)
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    eval_mask_matrix = model_metadata_dict[neural_net.MASK_MATRIX_KEY]

    for this_matching_distance_px in matching_distances_px:
        result_dict = uq_evaluation.get_spread_vs_skill(
            prediction_dict=prediction_dict,
            bin_edge_prediction_stdevs=bin_edge_prediction_stdevs + 0.,
            half_window_size_px=this_matching_distance_px,
            eval_mask_matrix=copy.deepcopy(eval_mask_matrix),
            use_median=False,
            use_fancy_quantile_method_for_stdev=
            use_fancy_quantile_method_for_stdev
        )

        output_file_name = (
            '{0:s}/spread_vs_skill_matching-distance-px={1:.6f}.nc'
        ).format(output_dir_name, this_matching_distance_px)

        print('Writing results to: "{0:s}"...'.format(output_file_name))
        uq_evaluation.write_spread_vs_skill(
            netcdf_file_name=output_file_name, result_dict=result_dict,
            half_window_size_px=this_matching_distance_px, use_median=False,
            use_fancy_quantile_method_for_stdev=
            use_fancy_quantile_method_for_stdev
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        time_interval_steps=getattr(INPUT_ARG_OBJECT, TIME_INTERVAL_ARG_NAME),
        matching_distances_px=numpy.array(
            getattr(INPUT_ARG_OBJECT, MATCHING_DISTANCES_ARG_NAME), dtype=float
        ),
        bin_edge_prediction_stdevs=numpy.array(
            getattr(INPUT_ARG_OBJECT, BIN_EDGES_ARG_NAME), dtype=float
        ),
        use_fancy_quantile_method_for_stdev=bool(
            getattr(INPUT_ARG_OBJECT, USE_FANCY_QUANTILES_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
