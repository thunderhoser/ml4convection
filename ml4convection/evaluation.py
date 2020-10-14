"""Model evaluation."""

import os
import sys
import pickle
import numpy
import xarray
from scipy.signal import convolve2d
from scipy.ndimage.morphology import binary_dilation, binary_erosion

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import histograms
import time_conversion
import number_rounding
import gg_model_evaluation as gg_model_eval
import file_system_utils
import error_checking
import prediction_io
import neural_net

TOLERANCE = 1e-6
DATE_FORMAT = '%Y%m%d'
TIME_FORMAT_FOR_MESSAGES = '%Y-%m-%d-%H%M'

DEFAULT_NUM_PROB_THRESHOLDS = 101
DEFAULT_NUM_BINS_FOR_RELIABILITY = 20

TIME_DIM = 'valid_time_unix_sec'
LATITUDE_DIM = 'latitude_deg_n'
LONGITUDE_DIM = 'longitude_deg_e'
PROBABILITY_THRESHOLD_DIM = 'probability_threshold'
RELIABILITY_BIN_DIM = 'reliability_bin'
SINGLETON_DIM = 'singleton'

NUM_ACTUAL_ORIENTED_TP_KEY = 'num_actual_oriented_true_positives'
NUM_PREDICTION_ORIENTED_TP_KEY = 'num_prediction_oriented_true_positives'
NUM_FALSE_NEGATIVES_KEY = 'num_false_negatives'
NUM_FALSE_POSITIVES_KEY = 'num_false_positives'

EXAMPLE_COUNT_KEY = 'num_examples'
SUMMED_FORECAST_PROB_KEY = 'summed_forecast_prob'
POSITIVE_EXAMPLE_COUNT_KEY = 'num_positive_examples'
MEAN_FORECAST_PROB_KEY = 'mean_forecast_prob'
EVENT_FREQUENCY_KEY = 'event_frequency'

ACTUAL_SSE_KEY = 'actual_sse'
REFERENCE_SSE_KEY = 'reference_sse'
BRIER_ACTUAL_SSE_KEY = 'actual_sse_for_brier'
CLIMO_SSE_KEY = 'climo_sse'

MODEL_FILE_KEY = 'model_file_name'
MATCHING_DISTANCE_KEY = 'matching_distance_px'
SQUARE_FSS_FILTER_KEY = 'square_fss_filter'
TRAINING_EVENT_FREQ_KEY = 'training_event_frequency'

POD_KEY = 'probability_of_detection'
SUCCESS_RATIO_KEY = 'success_ratio'
FREQUENCY_BIAS_KEY = 'frequency_bias'
CSI_KEY = 'critical_success_index'

BRIER_SKILL_SCORE_KEY = 'brier_skill_score'
BRIER_SCORE_KEY = 'brier_score'
RELIABILITY_KEY = 'reliability'
RESOLUTION_KEY = 'resolution'
FSS_KEY = 'fractions_skill_score'


def _check_2d_binary_matrix(binary_matrix):
    """Error-checks 2-D binary matrix.

    :param binary_matrix: 2-D numpy array, containing either Boolean flags or
        integers in 0...1.
    :return: is_boolean: Boolean flag, indicating whether or not matrix is
        Boolean.
    """

    error_checking.assert_is_numpy_array(binary_matrix, num_dimensions=2)

    try:
        error_checking.assert_is_boolean_numpy_array(binary_matrix)
        return True
    except TypeError:
        error_checking.assert_is_integer_numpy_array(binary_matrix)
        error_checking.assert_is_geq_numpy_array(binary_matrix, 0)
        error_checking.assert_is_leq_numpy_array(binary_matrix, 1)
        return False


def _get_structure_matrix(buffer_distance_px):
    """Creates structure matrix for dilation or erosion.

    :param buffer_distance_px: Buffer distance (number of pixels).
    :return: structure_matrix: 2-D numpy array of Boolean flags.
    """

    half_grid_size_px = int(numpy.ceil(buffer_distance_px))
    pixel_offsets = numpy.linspace(
        -half_grid_size_px, half_grid_size_px, num=2*half_grid_size_px + 1,
        dtype=float
    )

    column_offset_matrix, row_offset_matrix = numpy.meshgrid(
        pixel_offsets, pixel_offsets
    )
    distance_matrix_px = numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2
    )
    return distance_matrix_px <= buffer_distance_px


def _match_actual_convection_one_time(
        actual_target_matrix, predicted_target_matrix, matching_distance_px,
        eroded_eval_mask_matrix):
    """At one time, tries matching each actual convective px to predicted one.

    M = number of rows in grid
    N = number of columns in grid

    :param actual_target_matrix: M-by-N numpy array (see doc for
        `_check_2d_binary_matrix`), indicating where actual convection occurs.
    :param predicted_target_matrix: Same but for predicted convection.
    :param matching_distance_px: Matching distance (pixels).
    :param eroded_eval_mask_matrix: M-by-N numpy array (see doc for
        `_check_2d_binary_matrix`), indicating which pixels are to be used for
        evaluation.  This must be already eroded for the given matching
        distance.
    :return: fancy_prediction_matrix: M-by-N numpy array of integers (-1 where
        actual convection does not occur, 0 where actual convection occurs but
        is not predicted, 1 where actual convection occurs and is predicted).
    """

    fancy_prediction_matrix = dilate_binary_matrix(
        binary_matrix=predicted_target_matrix,
        buffer_distance_px=matching_distance_px
    ).astype(int)

    fancy_prediction_matrix[actual_target_matrix.astype(int) == 0] = -1
    fancy_prediction_matrix[eroded_eval_mask_matrix.astype(int) == 0] = -1

    return fancy_prediction_matrix


def _match_predicted_convection_one_time(
        actual_target_matrix, predicted_target_matrix, matching_distance_px,
        eroded_eval_mask_matrix):
    """At one time, tries matching each predicted convective px to actual one.

    :param actual_target_matrix: See doc for
        `_match_actual_convection_one_time`.
    :param predicted_target_matrix: Same.
    :param matching_distance_px: Same.
    :param eroded_eval_mask_matrix: Same.
    :return: fancy_target_matrix: M-by-N numpy array of integers (-1 where
        convection is not predicted, 0 where convection is predicted but does
        not occur, 1 where convection is predicted and occurs).
    """

    fancy_target_matrix = dilate_binary_matrix(
        binary_matrix=actual_target_matrix,
        buffer_distance_px=matching_distance_px
    ).astype(int)

    fancy_target_matrix[predicted_target_matrix.astype(int) == 0] = -1
    fancy_target_matrix[eroded_eval_mask_matrix.astype(int) == 0] = -1

    return fancy_target_matrix


def _get_reliability_components_one_time(
        actual_target_matrix, probability_matrix, matching_distance_px,
        num_bins, eroded_eval_mask_matrix):
    """Computes reliability components for one time step.

    M = number of rows in grid
    N = number of columns in grid
    B = number of bins for reliability curve

    :param actual_target_matrix: M-by-N numpy array (see doc for
        `_check_2d_binary_matrix`), indicating where actual convection occurs.
    :param probability_matrix: M-by-N numpy array of forecast convection
        probabilities.
    :param matching_distance_px: Matching distance (pixels).
    :param num_bins: Number of bins for reliability curve.
    :param eroded_eval_mask_matrix: See doc for
        `_match_actual_convection_one_time`.
    :return: example_count_matrix: M-by-N-by-B numpy array of total example
        counts.
    :return: summed_probability_matrix: M-by-N-by-B numpy array of summed
        forecast probabilities.
    :return: positive_example_count_matrix: M-by-N-by-B numpy array of positive-
        example counts.
    """

    dilated_actual_target_matrix = dilate_binary_matrix(
        binary_matrix=actual_target_matrix,
        buffer_distance_px=matching_distance_px
    ).astype(int)

    bin_indices = histograms.create_histogram(
        input_values=numpy.ravel(probability_matrix),
        num_bins=num_bins, min_value=0., max_value=1.
    )[0]

    bin_index_matrix = numpy.reshape(bin_indices, probability_matrix.shape)
    bin_index_matrix[eroded_eval_mask_matrix.astype(bool) == False] = -1

    these_dim = probability_matrix.shape + (num_bins,)
    example_count_matrix = numpy.full(these_dim, 0, dtype=int)
    summed_probability_matrix = numpy.full(these_dim, numpy.nan)
    positive_example_count_matrix = numpy.full(these_dim, 0, dtype=int)

    for j in range(num_bins):
        row_indices, column_indices = numpy.where(bin_index_matrix == j)
        example_count_matrix[..., j][row_indices, column_indices] = 1
        summed_probability_matrix[..., j][row_indices, column_indices] = (
            probability_matrix[row_indices, column_indices]
        )
        positive_example_count_matrix[..., j][row_indices, column_indices] = (
            dilated_actual_target_matrix[row_indices, column_indices]
        )

    return (
        example_count_matrix, summed_probability_matrix,
        positive_example_count_matrix
    )


def _get_fss_components_one_time(
        actual_target_matrix, probability_matrix, matching_distance_px,
        eroded_eval_mask_matrix, square_filter):
    """Computes FSS (fractions skill score) components for one time step.

    M = number of rows in grid
    N = number of columns in grid

    :param actual_target_matrix: See doc for
        `_get_reliability_components_one_time`.
    :param probability_matrix: Same.
    :param matching_distance_px: Same.
    :param eroded_eval_mask_matrix: Same.
    :param square_filter: See doc for `get_basic_scores_ungridded`.
    :return: actual_see_matrix: M-by-N numpy array with SSE (sum of squared
        errors) at each grid cell.
    :return: reference_sse_matrix: Same but for reference SSE.
    """

    # TODO(thunderhoser): This method does not deal with edge effects from
    # convolution, because I assume that the mask is all zero along the edge
    # anyways.

    if square_filter:
        structure_matrix = _get_structure_matrix(
            numpy.round(matching_distance_px)
        )
        weight_matrix = numpy.full(
            structure_matrix.shape, 1. / structure_matrix.size
        )
    else:
        structure_matrix = _get_structure_matrix(matching_distance_px)
        weight_matrix = structure_matrix.astype(float)
        weight_matrix = weight_matrix / numpy.sum(weight_matrix)

    smoothed_target_matrix = convolve2d(
        actual_target_matrix.astype(float), weight_matrix, mode='same'
    )
    smoothed_prob_matrix = convolve2d(
        probability_matrix, weight_matrix, mode='same'
    )
    smoothed_prob_matrix[eroded_eval_mask_matrix.astype(bool) == False] = (
        numpy.nan
    )

    actual_sse_matrix = (smoothed_target_matrix - smoothed_prob_matrix) ** 2
    reference_sse_matrix = (
        smoothed_target_matrix ** 2 + smoothed_prob_matrix ** 2
    )

    return actual_sse_matrix, reference_sse_matrix


def _get_bss_components_one_time(
        actual_target_matrix, probability_matrix, training_event_freq_matrix,
        matching_distance_px, eroded_eval_mask_matrix):
    """Computes BSS (Brier skill score) components for one time step.

    M = number of rows in grid
    N = number of columns in grid

    :param actual_target_matrix: See doc for
        `_get_reliability_components_one_time`.
    :param probability_matrix: Same.
    :param training_event_freq_matrix: M-by-N numpy array of event frequencies
        in training data ("climatologies").
    :param matching_distance_px: See doc for
        `_get_reliability_components_one_time`.
    :param eroded_eval_mask_matrix: Same.
    :return: actual_see_matrix: M-by-N numpy array with actual SSE (sum of
        squared errors) at each grid cell.
    :return: climo_sse_matrix: M-by-N numpy array with climatological SSE at
        each grid cell, obtained by always predicting climatology.
    """

    dilated_actual_target_matrix = dilate_binary_matrix(
        binary_matrix=actual_target_matrix,
        buffer_distance_px=matching_distance_px
    ).astype(int)

    actual_sse_matrix = (dilated_actual_target_matrix - probability_matrix) ** 2
    climo_sse_matrix = (
        (dilated_actual_target_matrix - training_event_freq_matrix) ** 2
    )

    actual_sse_matrix[eroded_eval_mask_matrix.astype(bool) == False] = numpy.nan
    climo_sse_matrix[eroded_eval_mask_matrix.astype(bool) == False] = numpy.nan

    return actual_sse_matrix, climo_sse_matrix


def _get_pod(contingency_table_dict):
    """Computes POD (probability of detection).

    :param contingency_table_dict: Dictionary with the following keys.
    contingency_table_dict["num_actual_oriented_true_positives"]
    contingency_table_dict["num_prediction_oriented_true_positives"]
    contingency_table_dict["num_false_positives"]
    contingency_table_dict["num_false_negatives"]

    :return: pod: Probability of detection.
    """

    numerator = contingency_table_dict[NUM_ACTUAL_ORIENTED_TP_KEY]
    denominator = (
        contingency_table_dict[NUM_ACTUAL_ORIENTED_TP_KEY] +
        contingency_table_dict[NUM_FALSE_NEGATIVES_KEY]
    )

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def _get_success_ratio(contingency_table_dict):
    """Computes success ratio.

    :param contingency_table_dict: See doc for `_get_pod`.
    :return: success_ratio: Success ratio.
    """

    numerator = contingency_table_dict[NUM_PREDICTION_ORIENTED_TP_KEY]
    denominator = (
        contingency_table_dict[NUM_FALSE_POSITIVES_KEY] +
        contingency_table_dict[NUM_PREDICTION_ORIENTED_TP_KEY]
    )

    try:
        return float(numerator) / denominator
    except ZeroDivisionError:
        return numpy.nan


def _get_csi(contingency_table_dict):
    """Computes CSI (critical success index).

    :param contingency_table_dict: See doc for `_get_pod`.
    :return: csi: Critical success index.
    """

    pod = _get_pod(contingency_table_dict)
    success_ratio = _get_success_ratio(contingency_table_dict)

    try:
        return (pod ** -1 + success_ratio ** -1 - 1) ** -1
    except ZeroDivisionError:
        return numpy.nan


def _get_frequency_bias(contingency_table_dict):
    """Computes frequency bias.

    :param contingency_table_dict: See doc for `_get_pod`.
    :return: frequency_bias: Frequency bias.
    """

    pod = _get_pod(contingency_table_dict)
    success_ratio = _get_success_ratio(contingency_table_dict)

    try:
        return pod / success_ratio
    except ZeroDivisionError:
        return numpy.nan


def _init_basic_score_table(
        valid_times_unix_sec, probability_thresholds, gridded,
        latitudes_deg_n=None, longitudes_deg_e=None,
        num_bins_for_reliability=None):
    """Initializes xarray table that will contain basic scores.

    :param valid_times_unix_sec: 1-D numpy array of valid times.
    :param probability_thresholds: 1-D numpy array of probability thresholds for
        contingency tables.
    :param gridded: Boolean flag.  If True, table will contain gridded scores
        (one set per grid cell).  If False, table will contain ungridded scores
        (aggregated over full domain).
    :param latitudes_deg_n: [used only if `gridded == True`]
        1-D numpy array of grid latitudes (deg N).
    :param longitudes_deg_e: [used only if `gridded == True`]
        1-D numpy array of grid longitudes (deg E).
    :param num_bins_for_reliability: [used only if `gridded == False`]
        Number of bins for reliability.
    :return: basic_score_table_xarray: xarray table (variable and dimension
        names should make the table self-explanatory).
    """

    metadata_dict = {
        TIME_DIM: valid_times_unix_sec,
        PROBABILITY_THRESHOLD_DIM: probability_thresholds
    }

    if gridded:
        metadata_dict[LATITUDE_DIM] = latitudes_deg_n
        metadata_dict[LONGITUDE_DIM] = longitudes_deg_e
    else:
        bin_indices = numpy.linspace(
            0, num_bins_for_reliability - 1, num=num_bins_for_reliability,
            dtype=int
        )
        metadata_dict[RELIABILITY_BIN_DIM] = bin_indices

    num_times = len(valid_times_unix_sec)
    num_prob_thresholds = len(probability_thresholds)

    if gridded:
        num_grid_rows = len(latitudes_deg_n)
        num_grid_columns = len(longitudes_deg_e)

        these_dim = (
            TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM, PROBABILITY_THRESHOLD_DIM
        )
        this_array = numpy.full(
            (num_times, num_grid_rows, num_grid_columns, num_prob_thresholds),
            0, dtype=int
        )
    else:
        num_grid_rows = 0
        num_grid_columns = 0

        these_dim = (TIME_DIM, PROBABILITY_THRESHOLD_DIM)
        this_array = numpy.full((num_times, num_prob_thresholds), 0, dtype=int)

    main_data_dict = {
        NUM_ACTUAL_ORIENTED_TP_KEY: (these_dim, this_array + 0),
        NUM_PREDICTION_ORIENTED_TP_KEY: (these_dim, this_array + 0),
        NUM_FALSE_NEGATIVES_KEY: (these_dim, this_array + 0),
        NUM_FALSE_POSITIVES_KEY: (these_dim, this_array + 0)
    }

    if gridded:
        these_dim_3d = (TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM)
        this_float_array_3d = numpy.full(
            (num_times, num_grid_rows, num_grid_columns), numpy.nan
        )
        this_integer_array_3d = numpy.full(
            (num_times, num_grid_rows, num_grid_columns), 0, dtype=int
        )

        these_dim_2d = (LATITUDE_DIM, LONGITUDE_DIM)
        this_array_2d = numpy.full((num_grid_rows, num_grid_columns), numpy.nan)

        new_dict = {
            EXAMPLE_COUNT_KEY: (these_dim_3d, this_integer_array_3d + 0),
            BRIER_ACTUAL_SSE_KEY: (these_dim_3d, this_float_array_3d + 0.),
            CLIMO_SSE_KEY: (these_dim_3d, this_float_array_3d + 0.),
            TRAINING_EVENT_FREQ_KEY: (these_dim_2d, this_array_2d + 0.)
        }
        main_data_dict.update(new_dict)
    else:
        these_dim = (TIME_DIM, RELIABILITY_BIN_DIM)
        this_integer_array = numpy.full(
            (num_times, num_bins_for_reliability), 0, dtype=int
        )
        this_float_array = numpy.full(
            (num_times, num_bins_for_reliability), numpy.nan
        )

        new_dict = {
            EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0),
            SUMMED_FORECAST_PROB_KEY: (these_dim, this_float_array + 0.),
            POSITIVE_EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0)
        }
        main_data_dict.update(new_dict)

    if gridded:
        these_dim = (TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM)
        this_array = numpy.full(
            (num_times, num_grid_rows, num_grid_columns), numpy.nan
        )
    else:
        these_dim = (TIME_DIM,)
        this_array = numpy.full(num_times, numpy.nan)

    new_dict = {
        ACTUAL_SSE_KEY: (these_dim, this_array + 0.),
        REFERENCE_SSE_KEY: (these_dim, this_array + 0.)
    }
    main_data_dict.update(new_dict)

    return xarray.Dataset(data_vars=main_data_dict, coords=metadata_dict)


def dilate_binary_matrix(binary_matrix, buffer_distance_px):
    """Dilates binary matrix.

    :param binary_matrix: See doc for `_check_2d_binary_matrix`.
    :param buffer_distance_px: Buffer distance (pixels).
    :return: dilated_binary_matrix: Dilated version of input.
    """

    _check_2d_binary_matrix(binary_matrix)
    error_checking.assert_is_geq(buffer_distance_px, 0.)

    structure_matrix = _get_structure_matrix(buffer_distance_px)
    dilated_binary_matrix = binary_dilation(
        binary_matrix.astype(int), structure=structure_matrix, iterations=1,
        border_value=0
    )
    return dilated_binary_matrix.astype(binary_matrix.dtype)


def erode_binary_matrix(binary_matrix, buffer_distance_px):
    """Erodes binary matrix.

    :param binary_matrix: See doc for `_check_2d_binary_matrix`.
    :param buffer_distance_px: Buffer distance (pixels).
    :return: eroded_binary_matrix: Eroded version of input.
    """

    _check_2d_binary_matrix(binary_matrix)
    error_checking.assert_is_geq(buffer_distance_px, 0.)

    structure_matrix = _get_structure_matrix(buffer_distance_px)
    eroded_binary_matrix = binary_erosion(
        binary_matrix.astype(int), structure=structure_matrix, iterations=1,
        border_value=1
    )
    return eroded_binary_matrix.astype(binary_matrix.dtype)


def get_basic_scores_ungridded(
        prediction_file_name, matching_distance_px, square_fss_filter=True,
        num_prob_thresholds=DEFAULT_NUM_PROB_THRESHOLDS,
        num_bins_for_reliability=DEFAULT_NUM_BINS_FOR_RELIABILITY,
        test_mode=False, prediction_dict=None, eval_mask_matrix=None,
        model_file_name=None):
    """Computes basic scores for full domain (aggregated in space).

    :param prediction_file_name: Path to input file (will be read by
        `prediction_io.read_file`).
    :param matching_distance_px: Matching distance (pixels) for neighbourhood
        evaluation.
    :param square_fss_filter: Boolean flag.  If True, the smoothing filter for
        FSS (fractions skill score) will be "squared" -- i.e., will be a square
        matrix with all non-zero values.  If False, the smoothing filter will
        not be "squared" -- i.e., some values in the matrix might be zero.
    :param num_prob_thresholds: Number of probability thresholds.  One
        contingency table will be created for each.
    :param num_bins_for_reliability: Number of bins for reliability curve.
    :param test_mode: Leave this alone.
    :param prediction_dict: Leave this alone.
    :param eval_mask_matrix: Leave this alone.
    :param model_file_name: Leave this alone.
    :return: basic_score_table_xarray: xarray table with results (variable
        and dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_boolean(test_mode)

    if not test_mode:
        print('Reading data from: "{0:s}"...'.format(prediction_file_name))
        prediction_dict = prediction_io.read_file(prediction_file_name)

        model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
        model_metafile_name = neural_net.find_metafile(
            model_file_name=model_file_name, raise_error_if_missing=True
        )

        print('Reading model metadata from: "{0:s}"...'.format(
            model_metafile_name
        ))
        model_metadata_dict = neural_net.read_metafile(model_metafile_name)
        eval_mask_matrix = model_metadata_dict[neural_net.MASK_MATRIX_KEY]

    # Check input args.
    _check_2d_binary_matrix(eval_mask_matrix)
    error_checking.assert_is_geq(matching_distance_px, 0.)
    error_checking.assert_is_boolean(square_fss_filter)
    error_checking.assert_is_integer(num_prob_thresholds)
    error_checking.assert_is_geq(num_prob_thresholds, 2)
    error_checking.assert_is_integer(num_bins_for_reliability)
    error_checking.assert_is_geq(num_bins_for_reliability, 10)

    # Create xarray table.
    valid_times_unix_sec = prediction_dict[prediction_io.VALID_TIMES_KEY]
    probability_thresholds = gg_model_eval.get_binarization_thresholds(
        threshold_arg=num_prob_thresholds
    )
    nan_array = numpy.full(1, numpy.nan)

    basic_score_table_xarray = _init_basic_score_table(
        valid_times_unix_sec=valid_times_unix_sec,
        probability_thresholds=probability_thresholds,
        num_bins_for_reliability=num_bins_for_reliability,
        latitudes_deg_n=nan_array, longitudes_deg_e=nan_array, gridded=False
    )

    basic_score_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    basic_score_table_xarray.attrs[MATCHING_DISTANCE_KEY] = matching_distance_px
    basic_score_table_xarray.attrs[SQUARE_FSS_FILTER_KEY] = square_fss_filter

    # Do actual stuff.
    eroded_eval_mask_matrix = erode_binary_matrix(
        binary_matrix=eval_mask_matrix, buffer_distance_px=matching_distance_px
    )
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_MESSAGES)
        for t in valid_times_unix_sec
    ]

    num_times = len(valid_times_unix_sec)
    num_prob_thresholds = len(probability_thresholds)

    for i in range(num_times):
        (
            this_example_count_matrix,
            this_summed_prob_matrix,
            this_pos_example_count_matrix
        ) = _get_reliability_components_one_time(
            actual_target_matrix=
            prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
            probability_matrix=
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...],
            matching_distance_px=matching_distance_px,
            num_bins=num_bins_for_reliability,
            eroded_eval_mask_matrix=eroded_eval_mask_matrix
        )

        basic_score_table_xarray[EXAMPLE_COUNT_KEY].values[i, :] = (
            numpy.sum(this_example_count_matrix, axis=(0, 1))
        )
        basic_score_table_xarray[SUMMED_FORECAST_PROB_KEY].values[i, :] = (
            numpy.nansum(this_summed_prob_matrix, axis=(0, 1))
        )
        basic_score_table_xarray[POSITIVE_EXAMPLE_COUNT_KEY].values[i, :] = (
            numpy.sum(this_pos_example_count_matrix, axis=(0, 1))
        )

        this_actual_sse_matrix, this_reference_sse_matrix = (
            _get_fss_components_one_time(
                actual_target_matrix=
                prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
                probability_matrix=
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...],
                matching_distance_px=matching_distance_px,
                eroded_eval_mask_matrix=eroded_eval_mask_matrix,
                square_filter=square_fss_filter
            )
        )

        basic_score_table_xarray[ACTUAL_SSE_KEY].values[i] = (
            numpy.nansum(this_actual_sse_matrix)
        )
        basic_score_table_xarray[REFERENCE_SSE_KEY].values[i] = (
            numpy.nansum(this_reference_sse_matrix)
        )

        for j in range(num_prob_thresholds):
            if numpy.mod(j, 10) == 0:
                print((
                    'Have computed contingency tables for {0:d} of {1:d} '
                    'probability thresholds at {2:s}...'
                ).format(
                    j, num_prob_thresholds, valid_time_strings[i]
                ))

            this_predicted_target_matrix = (
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...] >=
                probability_thresholds[j]
            )

            t = basic_score_table_xarray

            this_fancy_prediction_matrix = _match_actual_convection_one_time(
                actual_target_matrix=
                prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
                predicted_target_matrix=this_predicted_target_matrix,
                matching_distance_px=matching_distance_px,
                eroded_eval_mask_matrix=eroded_eval_mask_matrix
            )

            t[NUM_ACTUAL_ORIENTED_TP_KEY].values[i, j] = numpy.sum(
                this_fancy_prediction_matrix == 1
            )
            t[NUM_FALSE_NEGATIVES_KEY].values[i, j] = numpy.sum(
                this_fancy_prediction_matrix == 0
            )

            this_fancy_target_matrix = _match_predicted_convection_one_time(
                actual_target_matrix=
                prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
                predicted_target_matrix=this_predicted_target_matrix,
                matching_distance_px=matching_distance_px,
                eroded_eval_mask_matrix=eroded_eval_mask_matrix
            )

            t[NUM_PREDICTION_ORIENTED_TP_KEY].values[i, j] = numpy.sum(
                this_fancy_target_matrix == 1
            )
            t[NUM_FALSE_POSITIVES_KEY].values[i, j] = numpy.sum(
                this_fancy_target_matrix == 0
            )

            basic_score_table_xarray = t

        print((
            'Have computed contingency tables for all {0:d} probability '
            'thresholds at {1:s}!'
        ).format(
            num_prob_thresholds, valid_time_strings[i]
        ))

        if i != num_times - 1:
            print('\n')

    return basic_score_table_xarray


def get_basic_scores_gridded(
        prediction_file_name, matching_distance_px, probability_thresholds,
        training_event_freq_matrix, square_fss_filter=True,
        test_mode=False, prediction_dict=None, eval_mask_matrix=None,
        model_file_name=None):
    """Computes basic scores on grid (one set of scores for each grid point).

    M = number of rows in grid
    N = number of columns in grid

    :param prediction_file_name: See doc for `get_basic_scores_ungridded`.
    :param matching_distance_px: Same.
    :param probability_thresholds: 1-D numpy array of probability thresholds for
        contingency tables.
    :param training_event_freq_matrix: M-by-N numpy array of event frequencies
        ("climatologies") in training data.
    :param square_fss_filter: See doc for `get_basic_scores_ungridded`.
    :param test_mode: Leave this alone.
    :param prediction_dict: Leave this alone.
    :param eval_mask_matrix: Leave this alone.
    :param model_file_name: Leave this alone.
    :return: basic_score_table_xarray: xarray table with results (variable
        and dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_boolean(test_mode)

    if not test_mode:
        print('Reading data from: "{0:s}"...'.format(prediction_file_name))
        prediction_dict = prediction_io.read_file(prediction_file_name)

        model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
        model_metafile_name = neural_net.find_metafile(
            model_file_name=model_file_name, raise_error_if_missing=True
        )

        print('Reading model metadata from: "{0:s}"...'.format(
            model_metafile_name
        ))
        model_metadata_dict = neural_net.read_metafile(model_metafile_name)
        eval_mask_matrix = model_metadata_dict[neural_net.MASK_MATRIX_KEY]

    # Check input args.
    _check_2d_binary_matrix(eval_mask_matrix)
    error_checking.assert_is_geq(matching_distance_px, 0.)
    error_checking.assert_is_boolean(square_fss_filter)
    error_checking.assert_is_numpy_array(
        probability_thresholds, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(probability_thresholds, 0.)
    error_checking.assert_is_geq_numpy_array(
        training_event_freq_matrix, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        training_event_freq_matrix, 1., allow_nan=True
    )

    # Create xarray table.
    valid_times_unix_sec = prediction_dict[prediction_io.VALID_TIMES_KEY]

    basic_score_table_xarray = _init_basic_score_table(
        valid_times_unix_sec=valid_times_unix_sec,
        probability_thresholds=probability_thresholds,
        latitudes_deg_n=prediction_dict[prediction_io.LATITUDES_KEY],
        longitudes_deg_e=prediction_dict[prediction_io.LONGITUDES_KEY],
        gridded=True
    )

    basic_score_table_xarray[TRAINING_EVENT_FREQ_KEY].values = (
        training_event_freq_matrix
    )
    basic_score_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    basic_score_table_xarray.attrs[MATCHING_DISTANCE_KEY] = matching_distance_px
    basic_score_table_xarray.attrs[SQUARE_FSS_FILTER_KEY] = square_fss_filter

    # Do actual stuff.
    eroded_eval_mask_matrix = erode_binary_matrix(
        binary_matrix=eval_mask_matrix, buffer_distance_px=matching_distance_px
    )
    valid_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT_FOR_MESSAGES)
        for t in valid_times_unix_sec
    ]

    num_times = len(valid_times_unix_sec)
    num_prob_thresholds = len(probability_thresholds)

    for i in range(num_times):
        (
            basic_score_table_xarray[ACTUAL_SSE_KEY].values[i, ...],
            basic_score_table_xarray[REFERENCE_SSE_KEY].values[i, ...]
        ) = _get_fss_components_one_time(
            actual_target_matrix=
            prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
            probability_matrix=
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...],
            matching_distance_px=matching_distance_px,
            eroded_eval_mask_matrix=eroded_eval_mask_matrix,
            square_filter=square_fss_filter
        )

        (
            basic_score_table_xarray[BRIER_ACTUAL_SSE_KEY].values[i, ...],
            basic_score_table_xarray[CLIMO_SSE_KEY].values[i, ...]
        ) = _get_bss_components_one_time(
            actual_target_matrix=
            prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
            probability_matrix=
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...],
            training_event_freq_matrix=
            basic_score_table_xarray[TRAINING_EVENT_FREQ_KEY].values,
            matching_distance_px=matching_distance_px,
            eroded_eval_mask_matrix=eroded_eval_mask_matrix
        )

        this_count_matrix = numpy.invert(numpy.isnan(
            basic_score_table_xarray[BRIER_ACTUAL_SSE_KEY].values[i, ...]
        )).astype(int)

        basic_score_table_xarray[EXAMPLE_COUNT_KEY].values[i, ...] = (
            this_count_matrix + 0
        )

        for j in range(num_prob_thresholds):
            if numpy.mod(j, 10) == 0:
                print((
                    'Have computed contingency tables for {0:d} of {1:d} '
                    'probability thresholds at {2:s}...'
                ).format(
                    j, num_prob_thresholds, valid_time_strings[i]
                ))

            this_predicted_target_matrix = (
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...] >=
                probability_thresholds[j]
            )

            t = basic_score_table_xarray

            this_fancy_prediction_matrix = _match_actual_convection_one_time(
                actual_target_matrix=
                prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
                predicted_target_matrix=this_predicted_target_matrix,
                matching_distance_px=matching_distance_px,
                eroded_eval_mask_matrix=eroded_eval_mask_matrix
            )

            t[NUM_ACTUAL_ORIENTED_TP_KEY].values[i, ..., j] = (
                (this_fancy_prediction_matrix == 1).astype(int)
            )
            t[NUM_FALSE_NEGATIVES_KEY].values[i, ..., j] = (
                (this_fancy_prediction_matrix == 0).astype(int)
            )

            this_fancy_target_matrix = _match_predicted_convection_one_time(
                actual_target_matrix=
                prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
                predicted_target_matrix=this_predicted_target_matrix,
                matching_distance_px=matching_distance_px,
                eroded_eval_mask_matrix=eroded_eval_mask_matrix
            )

            t[NUM_PREDICTION_ORIENTED_TP_KEY].values[i, ..., j] = (
                (this_fancy_target_matrix == 1).astype(int)
            )
            t[NUM_FALSE_POSITIVES_KEY].values[i, ..., j] = (
                (this_fancy_target_matrix == 0).astype(int)
            )

            basic_score_table_xarray = t

        print((
            'Have computed contingency tables for all {0:d} probability '
            'thresholds at {1:s}!'
        ).format(
            num_prob_thresholds, valid_time_strings[i]
        ))

        if i != num_times - 1:
            print('\n')

    return basic_score_table_xarray


def concat_basic_score_tables(basic_score_tables_xarray):
    """Concatenates many tables along time dimension.

    :param basic_score_tables_xarray: 1-D list of xarray tables in format
        returned by `get_basic_scores_gridded` or `get_basic_scores_ungridded`.
    :return: basic_score_table_xarray: Single xarray table, containing data from
        all input tables.
    """

    model_file_names = [
        t.attrs[MODEL_FILE_KEY] for t in basic_score_tables_xarray
    ]
    matching_distances_px = numpy.array([
        t.attrs[MATCHING_DISTANCE_KEY] for t in basic_score_tables_xarray
    ])
    square_fss_filter_flags = numpy.array([
        t.attrs[SQUARE_FSS_FILTER_KEY] for t in basic_score_tables_xarray
    ], dtype=bool)

    unique_model_file_names = numpy.unique(numpy.array(model_file_names))
    unique_matching_distances_px = numpy.unique(
        number_rounding.round_to_nearest(matching_distances_px, TOLERANCE)
    )
    unique_square_flags = numpy.unique(square_fss_filter_flags)

    assert len(unique_model_file_names) == 1
    assert len(unique_matching_distances_px) == 1
    assert len(unique_square_flags) == 1

    non_empty_tables = [
        b for b in basic_score_tables_xarray
        if len(b.coords[TIME_DIM].values) > 0
    ]

    return xarray.concat(objs=non_empty_tables, dim=TIME_DIM)


def subset_basic_scores_by_hour(basic_score_table_xarray, desired_hour):
    """Subsets table by hour.

    :param basic_score_table_xarray: xarray table in format returned by
        `get_basic_scores_gridded` or `get_basic_scores_ungridded`.
    :param desired_hour: Desired hour (integer in range 0...23).
    :return: basic_score_table_xarray: Same as input but with fewer times.
    """

    error_checking.assert_is_integer(desired_hour)
    error_checking.assert_is_geq(desired_hour, 0)
    error_checking.assert_is_leq(desired_hour, 23)

    valid_times_unix_sec = basic_score_table_xarray.coords[TIME_DIM].values

    valid_hours = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%H'))
        for t in valid_times_unix_sec
    ], dtype=int)

    good_indices = numpy.where(valid_hours == desired_hour)[0]

    return basic_score_table_xarray.isel(
        indexers={TIME_DIM: good_indices}, drop=False
    )


def subset_basic_scores_by_space(
        basic_score_table_xarray, first_grid_row, last_grid_row,
        first_grid_column, last_grid_column):
    """Subsets table by space.

    :param basic_score_table_xarray: xarray table in format returned by
        `get_basic_scores_gridded`.
    :param first_grid_row: First row to keep (integer index).
    :param last_grid_row: Last row to keep (integer index).
    :param first_grid_column: First column to keep (integer index).
    :param last_grid_column: Last column to keep (integer index).
    :return: basic_score_table_xarray: Same as input but with fewer grid points.
    """

    num_grid_rows = len(basic_score_table_xarray.coords[LATITUDE_DIM].values)
    num_grid_columns = len(
        basic_score_table_xarray.coords[LONGITUDE_DIM].values
    )

    error_checking.assert_is_integer(first_grid_row)
    error_checking.assert_is_geq(first_grid_row, 0)
    error_checking.assert_is_integer(last_grid_row)
    error_checking.assert_is_less_than(last_grid_row, num_grid_rows)
    error_checking.assert_is_geq(last_grid_row, first_grid_row)

    error_checking.assert_is_integer(first_grid_column)
    error_checking.assert_is_geq(first_grid_column, 0)
    error_checking.assert_is_integer(last_grid_column)
    error_checking.assert_is_less_than(last_grid_column, num_grid_columns)
    error_checking.assert_is_geq(last_grid_column, first_grid_column)

    good_row_indices = numpy.linspace(
        first_grid_row, last_grid_row,
        num=last_grid_row - first_grid_row + 1, dtype=int
    )
    good_column_indices = numpy.linspace(
        first_grid_column, last_grid_column,
        num=last_grid_column - first_grid_column + 1, dtype=int
    )
    good_index_dict = {
        LATITUDE_DIM: good_row_indices,
        LONGITUDE_DIM: good_column_indices
    }

    return basic_score_table_xarray.isel(indexers=good_index_dict, drop=False)


def get_advanced_scores_gridded(basic_score_table_xarray):
    """Computes gridded advanced scores from gridded basic scores.

    :param basic_score_table_xarray: xarray table in format returned by
        `get_basic_scores_gridded`.
    :return: advanced_score_table_xarray: xarray table with advanced scores
        (variable and dimension names should make the table self-explanatory).
    """

    metadata_dict = dict()

    for this_key in [LATITUDE_DIM, LONGITUDE_DIM, PROBABILITY_THRESHOLD_DIM]:
        metadata_dict[this_key] = (
            basic_score_table_xarray.coords[this_key].values
        )

    num_grid_rows = len(metadata_dict[LATITUDE_DIM])
    num_grid_columns = len(metadata_dict[LONGITUDE_DIM])
    num_prob_thresholds = len(metadata_dict[PROBABILITY_THRESHOLD_DIM])

    these_dim = (LATITUDE_DIM, LONGITUDE_DIM, PROBABILITY_THRESHOLD_DIM)
    this_integer_array = numpy.full(
        (num_grid_rows, num_grid_columns, num_prob_thresholds),
        0, dtype=int
    )
    this_float_array = numpy.full(
        (num_grid_rows, num_grid_columns, num_prob_thresholds), numpy.nan
    )

    main_data_dict = {
        NUM_ACTUAL_ORIENTED_TP_KEY: (these_dim, this_integer_array + 0),
        NUM_PREDICTION_ORIENTED_TP_KEY: (these_dim, this_integer_array + 0),
        NUM_FALSE_NEGATIVES_KEY: (these_dim, this_integer_array + 0),
        NUM_FALSE_POSITIVES_KEY: (these_dim, this_integer_array + 0),
        POD_KEY: (these_dim, this_float_array + 0.),
        SUCCESS_RATIO_KEY: (these_dim, this_float_array + 0.),
        FREQUENCY_BIAS_KEY: (these_dim, this_float_array + 0.),
        CSI_KEY: (these_dim, this_float_array + 0.)
    }

    training_event_freq_matrix = (
        basic_score_table_xarray[TRAINING_EVENT_FREQ_KEY].values
    )

    try:
        if len(training_event_freq_matrix.shape) == 3:
            training_event_freq_matrix = training_event_freq_matrix[0, ...]
    except:
        training_event_freq_matrix = training_event_freq_matrix[0, ...]

    these_dim = (LATITUDE_DIM, LONGITUDE_DIM)
    this_integer_array = numpy.full(
        (num_grid_rows, num_grid_columns), 0, dtype=int
    )
    this_float_array = numpy.full(
        (num_grid_rows, num_grid_columns), numpy.nan
    )
    new_dict = {
        BRIER_ACTUAL_SSE_KEY: (these_dim, this_float_array + 0),
        CLIMO_SSE_KEY: (these_dim, this_float_array + 0.),
        EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0),
        BRIER_SCORE_KEY: (these_dim, this_float_array + 0.),
        BRIER_SKILL_SCORE_KEY: (these_dim, this_float_array + 0.),
        FSS_KEY: (these_dim, this_float_array + 0.),
        TRAINING_EVENT_FREQ_KEY: (these_dim, training_event_freq_matrix + 0.)
    }
    main_data_dict.update(new_dict)

    advanced_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict,
        attrs=basic_score_table_xarray.attrs
    )

    a = advanced_score_table_xarray
    b = basic_score_table_xarray

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            for k in range(num_prob_thresholds):
                this_contingency_table = {
                    NUM_ACTUAL_ORIENTED_TP_KEY: numpy.sum(
                        b[NUM_ACTUAL_ORIENTED_TP_KEY].values[:, i, j, k]
                    ),
                    NUM_PREDICTION_ORIENTED_TP_KEY: numpy.sum(
                        b[NUM_PREDICTION_ORIENTED_TP_KEY].values[:, i, j, k]
                    ),
                    NUM_FALSE_POSITIVES_KEY: numpy.sum(
                        b[NUM_FALSE_POSITIVES_KEY].values[:, i, j, k]
                    ),
                    NUM_FALSE_NEGATIVES_KEY: numpy.sum(
                        b[NUM_FALSE_NEGATIVES_KEY].values[:, i, j, k]
                    )
                }

                for this_key in [
                        NUM_ACTUAL_ORIENTED_TP_KEY,
                        NUM_PREDICTION_ORIENTED_TP_KEY,
                        NUM_FALSE_POSITIVES_KEY, NUM_FALSE_NEGATIVES_KEY
                ]:
                    a[this_key].values[i, j, k] = (
                        this_contingency_table[this_key]
                    )

                a[POD_KEY].values[i, j, k] = (
                    _get_pod(this_contingency_table)
                )
                a[SUCCESS_RATIO_KEY].values[
                    i, j, k
                ] = _get_success_ratio(this_contingency_table)

                a[CSI_KEY].values[i, j, k] = (
                    _get_csi(this_contingency_table)
                )
                a[FREQUENCY_BIAS_KEY].values[
                    i, j, k
                ] = _get_frequency_bias(this_contingency_table)

            a[BRIER_ACTUAL_SSE_KEY].values[i, j] = numpy.nansum(
                b[BRIER_ACTUAL_SSE_KEY].values[:, i, j]
            )
            a[CLIMO_SSE_KEY].values[i, j] = numpy.nansum(
                b[CLIMO_SSE_KEY].values[:, i, j]
            )
            a[EXAMPLE_COUNT_KEY].values[i, j] = numpy.sum(
                b[EXAMPLE_COUNT_KEY].values[:, i, j]
            )

            if a[EXAMPLE_COUNT_KEY].values[i, j] > 0:
                a[BRIER_SCORE_KEY].values[i, j] = (
                    a[BRIER_ACTUAL_SSE_KEY].values[i, j] /
                    a[EXAMPLE_COUNT_KEY].values[i, j]
                )
                a[BRIER_SKILL_SCORE_KEY].values[i, j] = 1. - (
                    a[BRIER_ACTUAL_SSE_KEY].values[i, j] /
                    a[CLIMO_SSE_KEY].values[i, j]
                )

            this_actual_sse = numpy.sum(b[ACTUAL_SSE_KEY].values[:, i, j])
            this_reference_sse = numpy.sum(b[REFERENCE_SSE_KEY].values[:, i, j])
            a[FSS_KEY].values[i, j] = 1. - this_actual_sse / this_reference_sse

    advanced_score_table_xarray = a
    return advanced_score_table_xarray


def get_advanced_scores_ungridded(basic_score_table_xarray,
                                  training_event_frequency):
    """Computes ungridded advanced scores from ungridded basic scores.

    :param basic_score_table_xarray: xarray table in format returned by
        `get_basic_scores_ungridded`.
    :param training_event_frequency: Event frequency in training data
        ("climatology").
    :return: advanced_score_table_xarray: xarray table with advanced scores
        (variable and dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_geq(training_event_frequency, 0.)
    error_checking.assert_is_leq(training_event_frequency, 1.)

    metadata_dict = dict()

    for this_key in [PROBABILITY_THRESHOLD_DIM, RELIABILITY_BIN_DIM]:
        metadata_dict[this_key] = (
            basic_score_table_xarray.coords[this_key].values
        )

    metadata_dict[SINGLETON_DIM] = numpy.array([0], dtype=int)

    num_prob_thresholds = len(metadata_dict[PROBABILITY_THRESHOLD_DIM])
    num_bins_for_reliability = len(metadata_dict[RELIABILITY_BIN_DIM])

    these_dim = (PROBABILITY_THRESHOLD_DIM,)
    this_integer_array = numpy.full(num_prob_thresholds, 0, dtype=int)
    this_float_array = numpy.full(num_prob_thresholds, numpy.nan)
    main_data_dict = {
        NUM_ACTUAL_ORIENTED_TP_KEY: (these_dim, this_integer_array + 0),
        NUM_PREDICTION_ORIENTED_TP_KEY: (these_dim, this_integer_array + 0),
        NUM_FALSE_NEGATIVES_KEY: (these_dim, this_integer_array + 0),
        NUM_FALSE_POSITIVES_KEY: (these_dim, this_integer_array + 0),
        POD_KEY: (these_dim, this_float_array + 0.),
        SUCCESS_RATIO_KEY: (these_dim, this_float_array + 0.),
        FREQUENCY_BIAS_KEY: (these_dim, this_float_array + 0.),
        CSI_KEY: (these_dim, this_float_array + 0.)
    }

    these_dim = (RELIABILITY_BIN_DIM,)
    this_integer_array = numpy.full(num_bins_for_reliability, 0, dtype=int)
    this_float_array = numpy.full(num_bins_for_reliability, numpy.nan)
    new_dict = {
        EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0),
        MEAN_FORECAST_PROB_KEY: (these_dim, this_float_array + 0.),
        EVENT_FREQUENCY_KEY: (these_dim, this_float_array + 0.)
    }
    main_data_dict.update(new_dict)

    these_dim = (SINGLETON_DIM,)
    this_float_array = numpy.full(1, numpy.nan)
    new_dict = {
        BRIER_SCORE_KEY: (these_dim, this_float_array + 0.),
        BRIER_SKILL_SCORE_KEY: (these_dim, this_float_array + 0.),
        RELIABILITY_KEY: (these_dim, this_float_array + 0.),
        RESOLUTION_KEY: (these_dim, this_float_array + 0.),
        FSS_KEY: (these_dim, this_float_array + 0.),
        TRAINING_EVENT_FREQ_KEY:
            (these_dim, numpy.full(1, training_event_frequency))
    }
    main_data_dict.update(new_dict)

    advanced_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict,
        attrs=basic_score_table_xarray.attrs
    )

    b = basic_score_table_xarray

    for k in range(num_prob_thresholds):
        this_contingency_table = {
            NUM_ACTUAL_ORIENTED_TP_KEY: numpy.sum(
                b[NUM_ACTUAL_ORIENTED_TP_KEY].values[:, k]
            ),
            NUM_PREDICTION_ORIENTED_TP_KEY: numpy.sum(
                b[NUM_PREDICTION_ORIENTED_TP_KEY].values[:, k]
            ),
            NUM_FALSE_POSITIVES_KEY: numpy.sum(
                b[NUM_FALSE_POSITIVES_KEY].values[:, k]
            ),
            NUM_FALSE_NEGATIVES_KEY: numpy.sum(
                b[NUM_FALSE_NEGATIVES_KEY].values[:, k]
            )
        }

        for this_key in [
                NUM_ACTUAL_ORIENTED_TP_KEY,
                NUM_PREDICTION_ORIENTED_TP_KEY,
                NUM_FALSE_POSITIVES_KEY, NUM_FALSE_NEGATIVES_KEY
        ]:
            advanced_score_table_xarray[this_key].values[k] = (
                this_contingency_table[this_key]
            )

        advanced_score_table_xarray[POD_KEY].values[k] = (
            _get_pod(this_contingency_table)
        )
        advanced_score_table_xarray[SUCCESS_RATIO_KEY].values[k] = (
            _get_success_ratio(this_contingency_table)
        )
        advanced_score_table_xarray[CSI_KEY].values[k] = (
            _get_csi(this_contingency_table)
        )
        advanced_score_table_xarray[FREQUENCY_BIAS_KEY].values[k] = (
            _get_frequency_bias(this_contingency_table)
        )

    for k in range(num_bins_for_reliability):
        this_num_examples = numpy.sum(b[EXAMPLE_COUNT_KEY].values[:, k])
        if this_num_examples == 0:
            continue

        this_sum_forecast_probs = numpy.nansum(
            b[SUMMED_FORECAST_PROB_KEY].values[:, k]
        )
        this_num_positive_examples = numpy.sum(
            b[POSITIVE_EXAMPLE_COUNT_KEY].values[:, k]
        )

        advanced_score_table_xarray[EXAMPLE_COUNT_KEY].values[k] = (
            this_num_examples
        )
        advanced_score_table_xarray[MEAN_FORECAST_PROB_KEY].values[k] = (
            this_sum_forecast_probs / this_num_examples
        )
        advanced_score_table_xarray[EVENT_FREQUENCY_KEY].values[k] = (
            float(this_num_positive_examples) / this_num_examples
        )

    if numpy.any(advanced_score_table_xarray[EXAMPLE_COUNT_KEY].values > 0):
        this_bss_dict = gg_model_eval.get_brier_skill_score(
            mean_forecast_prob_by_bin=
            advanced_score_table_xarray[MEAN_FORECAST_PROB_KEY].values,
            mean_observed_label_by_bin=
            advanced_score_table_xarray[EVENT_FREQUENCY_KEY].values,
            num_examples_by_bin=
            advanced_score_table_xarray[EXAMPLE_COUNT_KEY].values,
            climatology=
            advanced_score_table_xarray[TRAINING_EVENT_FREQ_KEY].values[0]
        )

        advanced_score_table_xarray[BRIER_SKILL_SCORE_KEY].values[0] = (
            this_bss_dict[gg_model_eval.BSS_KEY]
        )
        advanced_score_table_xarray[BRIER_SCORE_KEY].values[0] = (
            this_bss_dict[gg_model_eval.BRIER_SCORE_KEY]
        )
        advanced_score_table_xarray[RELIABILITY_KEY].values[0] = (
            this_bss_dict[gg_model_eval.RELIABILITY_KEY]
        )
        advanced_score_table_xarray[RESOLUTION_KEY].values[0] = (
            this_bss_dict[gg_model_eval.RESOLUTION_KEY]
        )

    actual_sse = numpy.sum(basic_score_table_xarray[ACTUAL_SSE_KEY].values)
    reference_sse = numpy.sum(
        basic_score_table_xarray[REFERENCE_SSE_KEY].values
    )
    advanced_score_table_xarray[FSS_KEY].values[0] = (
        1. - actual_sse / reference_sse
    )

    return advanced_score_table_xarray


def find_basic_score_file(top_directory_name, valid_date_string, gridded,
                          raise_error_if_missing=True):
    """Finds NetCDF file with basic evaluation scores.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param gridded: Boolean flag.  If True, will look for file with gridded
        scores.  If False, will look for file with scores aggregated over full
        domain.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: basic_score_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(gridded)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    basic_score_file_name = (
        '{0:s}/{1:s}/basic_scores_gridded={2:d}_{3:s}.nc'
    ).format(
        top_directory_name, valid_date_string[:4], int(gridded),
        valid_date_string
    )

    if os.path.isfile(basic_score_file_name) or not raise_error_if_missing:
        return basic_score_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        basic_score_file_name
    )
    raise ValueError(error_string)


def basic_file_name_to_date(basic_score_file_name):
    """Parses date from name of basic-score file.

    :param basic_score_file_name: Path to evaluation file (see
        `find_basic_score_file` for naming convention).
    :return: valid_date_string: Valid date (format "yyyymmdd").
    """

    error_checking.assert_is_string(basic_score_file_name)
    pathless_file_name = os.path.split(basic_score_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    valid_date_string = extensionless_file_name.split('_')[-1]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def find_many_basic_score_files(
        top_directory_name, first_date_string, last_date_string, gridded,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many NetCDF files with evaluation scores.

    :param top_directory_name: See doc for `find_basic_score_file`.
    :param first_date_string: First valid date (format "yyyymmdd").
    :param last_date_string: Last valid date (format "yyyymmdd").
    :param gridded: See doc for `find_basic_score_file`.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: basic_score_file_names: 1-D list of file paths.  This list does
        *not* contain expected paths to non-existent files.
    :raises: ValueError: if all files are missing and
        `raise_error_if_all_missing == True`.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    basic_score_file_names = []

    for this_date_string in valid_date_strings:
        this_file_name = find_basic_score_file(
            top_directory_name=top_directory_name,
            valid_date_string=this_date_string, gridded=gridded,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            basic_score_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(basic_score_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from dates {1:s} to '
            '{2:s}.'
        ).format(
            top_directory_name, first_date_string, last_date_string
        )
        raise ValueError(error_string)

    return basic_score_file_names


def write_basic_score_file(basic_score_table_xarray, netcdf_file_name):
    """Writes basic scores to NetCDF file.

    :param basic_score_table_xarray: xarray table created by
        `get_basic_scores_gridded` or `get_basic_scores_ungridded`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    basic_score_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_basic_score_file(netcdf_file_name):
    """Reads basic scores from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: basic_score_table_xarray: xarray table created by
        `get_basic_scores_gridded` or `get_basic_scores_ungridded`.
    """

    error_checking.assert_file_exists(netcdf_file_name)

    return xarray.open_dataset(netcdf_file_name)


def find_advanced_score_file(
        directory_name, gridded, month=None, hour=None,
        raise_error_if_missing=True):
    """Finds Pickle file with advanced evaluation scores.

    :param directory_name: Name of directory where file is expected.
    :param gridded: Boolean flag.  If True, will look for file with gridded
        scores (a different set of scores at each grid cell).  If False, will
        look for file with scores aggregated over the full domain.
    :param month: Month (integer in 1...12).  If None, will look for file that
        has all months.
    :param hour: Hour (integer in 0...23).  If None, will look for file that
        has all hours.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: advanced_score_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_boolean(gridded)
    error_checking.assert_is_boolean(raise_error_if_missing)

    advanced_score_file_name = '{0:s}/advanced_scores'.format(directory_name)

    if month is not None:
        error_checking.assert_is_integer(month)
        error_checking.assert_is_geq(month, 1)
        error_checking.assert_is_leq(month, 12)
        advanced_score_file_name += '_month={0:02d}'.format(month)
    elif hour is not None:
        error_checking.assert_is_integer(hour)
        error_checking.assert_is_geq(hour, 0)
        error_checking.assert_is_leq(hour, 23)
        advanced_score_file_name += '_hour={0:02d}'.format(hour)

    advanced_score_file_name += '_gridded={0:d}.p'.format(int(gridded))

    if os.path.isfile(advanced_score_file_name) or not raise_error_if_missing:
        return advanced_score_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        advanced_score_file_name
    )
    raise ValueError(error_string)


def write_advanced_score_file(advanced_score_table_xarray, pickle_file_name):
    """Writes advanced scores to Pickle file.

    :param advanced_score_table_xarray: xarray table created by
        `get_advanced_scores`.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(advanced_score_table_xarray, pickle_file_handle)
    pickle_file_handle.close()


def read_advanced_score_file(pickle_file_name):
    """Reads advanced scores from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: advanced_score_table_xarray: xarray table created by
        `get_advanced_scores`.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    advanced_score_table_xarray = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return advanced_score_table_xarray