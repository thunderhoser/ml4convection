"""Computes basic evaluation scores sans grid (combined over full domain)."""

import os
import sys
import argparse
import numpy
import tensorflow
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_model_evaluation as gg_model_eval
import error_checking
import prediction_io
import evaluation
import radar_utils
import fourier_utils
import wavelet_utils
import neural_net
from _wavetf import WaveTFFactory

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)
GRID_SPACING_DEG = 0.0125
NUM_EXAMPLES_PER_FOURIER_BATCH = 20
TARGET_PERCENTILE_LEVELS_TO_REPORT = numpy.array(
    [0, 25, 50, 75, 95, 99, 100], dtype=float
)

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
TIME_INTERVAL_ARG_NAME = 'time_interval_steps'
USE_PARTIAL_GRIDS_ARG_NAME = 'use_partial_grids'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
TRANSFORM_TARGETS_ARG_NAME = 'transform_targets_if_applicable'
MATCHING_DISTANCES_ARG_NAME = 'matching_distances_px'
NUM_PROB_THRESHOLDS_ARG_NAME = 'num_prob_thresholds'
PROB_THRESHOLDS_ARG_NAME = 'prob_thresholds'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory.  Files therein will be found by '
    '`prediction_io.find_file` and read by `prediction_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  Will evaluate predictions for all days in the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

TIME_INTERVAL_HELP_STRING = (
    'Will compute scores for every [k]th time step, where k = `{0:s}`.'
).format(TIME_INTERVAL_ARG_NAME)

USE_PARTIAL_GRIDS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will compute scores for partial (full) grids.'
)
SMOOTHING_RADIUS_HELP_STRING = (
    '[used only if {0:s} == 0] Radius for Gaussian smoother.  If you do not '
    'want to smooth predictions, leave this alone.'
).format(USE_PARTIAL_GRIDS_ARG_NAME)

TRANSFORM_TARGETS_HELP_STRING = (
    'Boolean flag.  If 1 and neural net was trained with wavelet- or Fourier-'
    'transformed target values, this script will use the same transform before '
    'computing scores.'
)
MATCHING_DISTANCES_HELP_STRING = (
    'List of matching distances (pixels).  Neighbourhood evaluation will be '
    'done for each matching distance.'
)
NUM_PROB_THRESHOLDS_HELP_STRING = (
    'Number of probability thresholds.  One contingency table will be created '
    'for each.  If you want to use specific thresholds, leave this argument '
    'alone and specify `{0:s}`.'
).format(PROB_THRESHOLDS_ARG_NAME)

PROB_THRESHOLDS_HELP_STRING = (
    'List of exact probability thresholds.  One contingency table will be '
    'created for each.  If you do not want to use specific thresholds, leave '
    'this argument alone and specify `{0:s}`.'
).format(NUM_PROB_THRESHOLDS_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Results will be written here by '
    '`evaluation.write_basic_score_file`, to exact locations determined by '
    '`evaluation.find_basic_score_file`.'
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
    '--' + USE_PARTIAL_GRIDS_ARG_NAME, type=int, required=False, default=0,
    help=USE_PARTIAL_GRIDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False, default=-1,
    help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRANSFORM_TARGETS_ARG_NAME, type=int, required=False, default=0,
    help=TRANSFORM_TARGETS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCES_ARG_NAME, type=float, nargs='+', required=True,
    help=MATCHING_DISTANCES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PROB_THRESHOLDS_ARG_NAME, type=int, required=False,
    default=-1, help=NUM_PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROB_THRESHOLDS_ARG_NAME, type=float, nargs='+', required=False,
    default=[-1], help=PROB_THRESHOLDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _transform_targets(prediction_dict):
    """Transforms target values.

    If the neural net was trained with wavelet- or Fourier-transformed target
    values, this method will apply the same transform.

    :param prediction_dict: Dictionary in format returned by
        `prediction_io.read_file`.
    :return: prediction_dict: Same but with transformed target values.
    """

    model_metafile_name = neural_net.find_metafile(
        model_file_name=prediction_dict[prediction_io.MODEL_FILE_KEY],
        raise_error_if_missing=True
    )
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = (
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )

    target_matrix = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].astype(float)
    )
    num_examples = target_matrix.shape[0]

    if training_option_dict[neural_net.FOURIER_TRANSFORM_KEY]:
        print(MINOR_SEPARATOR_STRING)
        for p in TARGET_PERCENTILE_LEVELS_TO_REPORT:
            print((
                '{0:.1f}th-percentile target value before Fourier '
                'transform = {1:.4f}'
            ).format(
                p, numpy.percentile(target_matrix, p)
            ))

        filtered_target_matrix = numpy.full(target_matrix.shape, numpy.nan)

        target_matrix = numpy.stack([
            fourier_utils.taper_spatial_data(target_matrix[i, ...])
            for i in range(num_examples)
        ], axis=0)

        blackman_matrix = fourier_utils.apply_blackman_window(
            numpy.ones(target_matrix.shape[1:])
        )
        target_matrix = numpy.stack([
            target_matrix[i, ...] * blackman_matrix
            for i in range(num_examples)
        ], axis=0)

        butterworth_matrix = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=numpy.ones(target_matrix.shape[1:]),
            filter_order=2, grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=
            training_option_dict[neural_net.MIN_TARGET_RESOLUTION_KEY],
            max_resolution_metres=
            training_option_dict[neural_net.MAX_TARGET_RESOLUTION_KEY]
        )

        for i in range(0, num_examples, NUM_EXAMPLES_PER_FOURIER_BATCH):
            first_index = i
            last_index = min([
                i + NUM_EXAMPLES_PER_FOURIER_BATCH, num_examples
            ])

            this_target_tensor = tensorflow.constant(
                target_matrix[first_index:last_index, ...],
                dtype=tensorflow.complex128
            )
            this_target_weight_tensor = tensorflow.signal.fft2d(
                this_target_tensor
            )
            this_target_weight_matrix = K.eval(this_target_weight_tensor)

            this_target_weight_matrix = numpy.stack([
                this_target_weight_matrix[i, ...] * butterworth_matrix
                for i in range(this_target_weight_matrix.shape[0])
            ], axis=0)

            this_target_weight_tensor = tensorflow.constant(
                this_target_weight_matrix, dtype=tensorflow.complex128
            )
            this_target_tensor = tensorflow.signal.ifft2d(
                this_target_weight_tensor
            )
            this_target_tensor = tensorflow.math.real(this_target_tensor)
            this_target_matrix = K.eval(this_target_tensor)

            filtered_target_matrix[first_index:last_index, ...] = numpy.stack([
                fourier_utils.untaper_spatial_data(this_target_matrix[i, ...])
                for i in range(this_target_matrix.shape[0])
            ], axis=0)

        target_matrix = numpy.maximum(filtered_target_matrix, 0.)
        target_matrix = numpy.minimum(target_matrix, 1.)
        del filtered_target_matrix

        print('\n')
        for p in TARGET_PERCENTILE_LEVELS_TO_REPORT:
            print((
                '{0:.1f}th-percentile target value after Fourier '
                'transform = {1:.4f}'
            ).format(
                p, numpy.percentile(target_matrix, p)
            ))

        print(MINOR_SEPARATOR_STRING)

    if training_option_dict[neural_net.WAVELET_TRANSFORM_KEY]:
        print(MINOR_SEPARATOR_STRING)
        for p in TARGET_PERCENTILE_LEVELS_TO_REPORT:
            print((
                '{0:.1f}th-percentile target value before wavelet '
                'transform = {1:.4f}'
            ).format(
                p, numpy.percentile(target_matrix, p)
            ))

        target_matrix, padding_arg = wavelet_utils.taper_spatial_data(
            target_matrix
        )

        coeff_tensor_by_level = wavelet_utils.do_forward_transform(
            target_matrix
        )
        coeff_tensor_by_level = wavelet_utils.filter_coefficients(
            coeff_tensor_by_level=coeff_tensor_by_level,
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=
            training_option_dict[neural_net.MIN_TARGET_RESOLUTION_KEY],
            max_resolution_metres=
            training_option_dict[neural_net.MAX_TARGET_RESOLUTION_KEY],
            verbose=True
        )

        inverse_dwt_object = WaveTFFactory().build(
            'haar', dim=2, inverse=True
        )
        target_tensor = inverse_dwt_object.call(
            coeff_tensor_by_level[0]
        )
        target_matrix = K.eval(target_tensor)[..., 0]

        target_matrix = wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=target_matrix,
            numpy_pad_width=padding_arg
        )
        target_matrix = numpy.maximum(target_matrix, 0.)
        target_matrix = numpy.minimum(target_matrix, 1.)

        print('\n')
        for p in TARGET_PERCENTILE_LEVELS_TO_REPORT:
            print((
                '{0:.1f}th-percentile target value after wavelet '
                'transform = {1:.4f}'
            ).format(
                p, numpy.percentile(target_matrix, p)
            ))

        print(MINOR_SEPARATOR_STRING)

    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = target_matrix
    return prediction_dict


def _compute_scores_full_grid(
        top_prediction_dir_name, first_date_string, last_date_string,
        time_interval_steps, smoothing_radius_px,
        transform_targets_if_applicable, matching_distances_px, prob_thresholds,
        top_output_dir_name):
    """Computes scores on full grid.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param time_interval_steps: Same.
    :param smoothing_radius_px: Same.
    :param transform_targets_if_applicable: Same.
    :param matching_distances_px: Same.
    :param prob_thresholds: Same.
    :param top_output_dir_name: Same.
    """

    prediction_file_names = prediction_io.find_many_files(
        top_directory_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        radar_number=None, prefer_zipped=False, allow_other_format=True,
        raise_error_if_any_missing=False
    )
    date_strings = [
        prediction_io.file_name_to_date(f) for f in prediction_file_names
    ]

    num_dates = len(date_strings)
    num_matching_distances = len(matching_distances_px)

    for i in range(num_dates):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        prediction_dict = prediction_io.read_file(prediction_file_names[i])

        num_times = len(prediction_dict[prediction_io.VALID_TIMES_KEY])
        desired_indices = numpy.linspace(
            0, num_times - 1, num=num_times, dtype=int
        )
        desired_indices = desired_indices[::time_interval_steps]

        prediction_dict = prediction_io.subset_by_index(
            prediction_dict=prediction_dict, desired_indices=desired_indices
        )

        if smoothing_radius_px is not None:
            prediction_dict = prediction_io.smooth_probabilities(
                prediction_dict=prediction_dict,
                smoothing_radius_px=smoothing_radius_px
            )

        if transform_targets_if_applicable:
            prediction_dict = _transform_targets(prediction_dict)

        for j in range(num_matching_distances):
            print('\n')

            basic_score_table_xarray = evaluation.get_basic_scores_ungridded(
                prediction_dict=prediction_dict,
                matching_distance_px=matching_distances_px[j],
                probability_thresholds=prob_thresholds
            )

            output_dir_name = '{0:s}/matching_distance_px={1:.6f}'.format(
                top_output_dir_name, matching_distances_px[j]
            )
            output_file_name = evaluation.find_basic_score_file(
                top_directory_name=output_dir_name,
                valid_date_string=date_strings[i],
                gridded=False, radar_number=None, raise_error_if_missing=False
            )

            print('\nWriting results to: "{0:s}"...'.format(output_file_name))
            evaluation.write_basic_score_file(
                basic_score_table_xarray=basic_score_table_xarray,
                netcdf_file_name=output_file_name
            )

        if i == num_dates - 1:
            continue

        print(SEPARATOR_STRING)


def _compute_scores_partial_grids(
        top_prediction_dir_name, first_date_string, last_date_string,
        time_interval_steps, transform_targets_if_applicable,
        matching_distances_px, prob_thresholds, top_output_dir_name):
    """Computes scores on partial grids.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param time_interval_steps: Same.
    :param transform_targets_if_applicable: Same.
    :param matching_distances_px: Same.
    :param prob_thresholds: Same.
    :param top_output_dir_name: Same.
    """

    date_strings = []

    for k in range(NUM_RADARS):
        if len(date_strings) == 0:
            prediction_file_names = prediction_io.find_many_files(
                top_directory_name=top_prediction_dir_name,
                first_date_string=first_date_string,
                last_date_string=last_date_string,
                radar_number=k, prefer_zipped=True, allow_other_format=True,
                raise_error_if_any_missing=False,
                raise_error_if_all_missing=k > 0
            )

            if len(prediction_file_names) == 0:
                continue

            date_strings = [
                prediction_io.file_name_to_date(f)
                for f in prediction_file_names
            ]
        else:
            prediction_file_names = [
                prediction_io.find_file(
                    top_directory_name=top_prediction_dir_name,
                    valid_date_string=d, radar_number=k,
                    prefer_zipped=True, allow_other_format=True,
                    raise_error_if_missing=True
                ) for d in date_strings
            ]

        num_dates = len(date_strings)
        num_matching_distances = len(matching_distances_px)

        for i in range(num_dates):
            print('Reading data from: "{0:s}"...'.format(
                prediction_file_names[i]
            ))
            prediction_dict = prediction_io.read_file(prediction_file_names[i])

            num_times = len(prediction_dict[prediction_io.VALID_TIMES_KEY])
            desired_indices = numpy.linspace(
                0, num_times - 1, num=num_times, dtype=int
            )
            desired_indices = desired_indices[::time_interval_steps]

            prediction_dict = prediction_io.subset_by_index(
                prediction_dict=prediction_dict, desired_indices=desired_indices
            )

            if transform_targets_if_applicable:
                prediction_dict = _transform_targets(prediction_dict)

            for j in range(num_matching_distances):
                print('\n')

                basic_score_table_xarray = (
                    evaluation.get_basic_scores_ungridded(
                        prediction_dict=prediction_dict,
                        matching_distance_px=matching_distances_px[j],
                        probability_thresholds=prob_thresholds
                    )
                )

                output_dir_name = '{0:s}/matching_distance_px={1:.6f}'.format(
                    top_output_dir_name, matching_distances_px[j]
                )
                output_file_name = evaluation.find_basic_score_file(
                    top_directory_name=output_dir_name,
                    valid_date_string=date_strings[i],
                    gridded=False, radar_number=k, raise_error_if_missing=False
                )

                print('\nWriting results to: "{0:s}"...'.format(
                    output_file_name
                ))
                evaluation.write_basic_score_file(
                    basic_score_table_xarray=basic_score_table_xarray,
                    netcdf_file_name=output_file_name
                )

            if not (i == num_dates - 1 and k == NUM_RADARS - 1):
                continue

            print(SEPARATOR_STRING)


def _run(top_prediction_dir_name, first_date_string, last_date_string,
         time_interval_steps, use_partial_grids, smoothing_radius_px,
         transform_targets_if_applicable, matching_distances_px,
         num_prob_thresholds, prob_thresholds, top_output_dir_name):
    """Computes basic evaluation scores sans grid (combined over full domain).

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param time_interval_steps: Same.
    :param use_partial_grids: Same.
    :param smoothing_radius_px: Same.
    :param transform_targets_if_applicable: Same.
    :param matching_distances_px: Same.
    :param num_prob_thresholds: Same.
    :param prob_thresholds: Same.
    :param top_output_dir_name: Same.
    """

    error_checking.assert_is_geq(time_interval_steps, 1)

    if num_prob_thresholds > 0:
        prob_thresholds = gg_model_eval.get_binarization_thresholds(
            threshold_arg=num_prob_thresholds
        )

    if not use_partial_grids:
        if smoothing_radius_px <= 0:
            smoothing_radius_px = None

        _compute_scores_full_grid(
            top_prediction_dir_name=top_prediction_dir_name,
            first_date_string=first_date_string,
            last_date_string=last_date_string,
            time_interval_steps=time_interval_steps,
            smoothing_radius_px=smoothing_radius_px,
            transform_targets_if_applicable=transform_targets_if_applicable,
            matching_distances_px=matching_distances_px,
            prob_thresholds=prob_thresholds,
            top_output_dir_name=top_output_dir_name
        )

        return

    _compute_scores_partial_grids(
        top_prediction_dir_name=top_prediction_dir_name,
        first_date_string=first_date_string,
        last_date_string=last_date_string,
        time_interval_steps=time_interval_steps,
        transform_targets_if_applicable=transform_targets_if_applicable,
        matching_distances_px=matching_distances_px,
        prob_thresholds=prob_thresholds,
        top_output_dir_name=top_output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        time_interval_steps=getattr(INPUT_ARG_OBJECT, TIME_INTERVAL_ARG_NAME),
        use_partial_grids=bool(getattr(
            INPUT_ARG_OBJECT, USE_PARTIAL_GRIDS_ARG_NAME
        )),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        transform_targets_if_applicable=bool(getattr(
            INPUT_ARG_OBJECT, TRANSFORM_TARGETS_ARG_NAME
        )),
        matching_distances_px=numpy.array(
            getattr(INPUT_ARG_OBJECT, MATCHING_DISTANCES_ARG_NAME), dtype=float
        ),
        num_prob_thresholds=getattr(
            INPUT_ARG_OBJECT, NUM_PROB_THRESHOLDS_ARG_NAME
        ),
        prob_thresholds=numpy.array(getattr(
            INPUT_ARG_OBJECT, PROB_THRESHOLDS_ARG_NAME
        )),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
