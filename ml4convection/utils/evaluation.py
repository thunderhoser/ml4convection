"""Model evaluation."""

import pickle
import os.path
import numpy
import xarray
from scipy.signal import convolve2d
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from gewittergefahr.gg_utils import histograms
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import model_evaluation as gg_model_eval
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import prediction_io
from ml4convection.machine_learning import neural_net

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
    :param square_filter: See doc for `get_basic_scores`.
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


def get_basic_scores(
        prediction_file_name, matching_distance_px, square_fss_filter=True,
        num_prob_thresholds=DEFAULT_NUM_PROB_THRESHOLDS,
        num_bins_for_reliability=DEFAULT_NUM_BINS_FOR_RELIABILITY,
        test_mode=False, prediction_dict=None, eval_mask_matrix=None,
        model_file_name=None):
    """Computes basic scores.

    M = number of rows in grid
    N = number of columns in grid

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
    bin_indices = numpy.linspace(
        0, num_bins_for_reliability - 1, num=num_bins_for_reliability, dtype=int
    )
    metadata_dict = {
        TIME_DIM: valid_times_unix_sec,
        LATITUDE_DIM: prediction_dict[prediction_io.LATITUDES_KEY],
        LONGITUDE_DIM: prediction_dict[prediction_io.LONGITUDES_KEY],
        PROBABILITY_THRESHOLD_DIM: probability_thresholds,
        RELIABILITY_BIN_DIM: bin_indices
    }

    num_times = len(valid_times_unix_sec)
    num_grid_rows = len(prediction_dict[prediction_io.LATITUDES_KEY])
    num_grid_columns = len(prediction_dict[prediction_io.LONGITUDES_KEY])
    num_prob_thresholds = len(probability_thresholds)

    these_dim = (
        TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM, PROBABILITY_THRESHOLD_DIM
    )
    this_array = numpy.full(
        (num_times, num_grid_rows, num_grid_columns, num_prob_thresholds),
        0, dtype=int
    )
    main_data_dict = {
        NUM_ACTUAL_ORIENTED_TP_KEY: (these_dim, this_array + 0),
        NUM_PREDICTION_ORIENTED_TP_KEY: (these_dim, this_array + 0),
        NUM_FALSE_NEGATIVES_KEY: (these_dim, this_array + 0),
        NUM_FALSE_POSITIVES_KEY: (these_dim, this_array + 0)
    }

    these_dim = (TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM, RELIABILITY_BIN_DIM)
    this_integer_array = numpy.full(
        (num_times, num_grid_rows, num_grid_columns, num_bins_for_reliability),
        0, dtype=int
    )
    this_float_array = numpy.full(
        (num_times, num_grid_rows, num_grid_columns, num_bins_for_reliability),
        numpy.nan
    )
    new_dict = {
        EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0),
        SUMMED_FORECAST_PROB_KEY: (these_dim, this_float_array + 0.),
        POSITIVE_EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0)
    }
    main_data_dict.update(new_dict)

    these_dim = (TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM)
    this_array = numpy.full(
        (num_times, num_grid_rows, num_grid_columns), numpy.nan
    )
    new_dict = {
        ACTUAL_SSE_KEY: (these_dim, this_array + 0.),
        REFERENCE_SSE_KEY: (these_dim, this_array + 0.)
    }
    main_data_dict.update(new_dict)

    basic_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
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

    for i in range(num_times):
        (
            basic_score_table_xarray[EXAMPLE_COUNT_KEY].values[i, ...],
            basic_score_table_xarray[SUMMED_FORECAST_PROB_KEY].values[i, ...],
            basic_score_table_xarray[POSITIVE_EXAMPLE_COUNT_KEY].values[i, ...]
        ) = _get_reliability_components_one_time(
            actual_target_matrix=
            prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...],
            probability_matrix=
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...],
            matching_distance_px=matching_distance_px,
            num_bins=num_bins_for_reliability,
            eroded_eval_mask_matrix=eroded_eval_mask_matrix
        )

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
        returned by `get_basic_scores`.
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

    return xarray.concat(objs=basic_score_tables_xarray, dim=TIME_DIM)


def subset_basic_scores_by_hour(basic_score_table_xarray, desired_hour):
    """Subsets table by hour.

    :param basic_score_table_xarray: xarray table in format returned by
        `get_basic_scores`.
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
        `get_basic_scores`.
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


def aggregate_basic_scores_in_space(basic_score_table_xarray):
    """Aggregates basic scores in space.

    :param basic_score_table_xarray: xarray table in format returned by
        `get_basic_scores`, with different scores for each grid cell.
    :return: basic_score_table_xarray: Same as input but with the same scores
        for all grid cells.
    """

    # Create new table.
    metadata_dict = {
        LATITUDE_DIM: numpy.array([numpy.nan]),
        LONGITUDE_DIM: numpy.array([numpy.nan])
    }

    for this_key in [TIME_DIM, PROBABILITY_THRESHOLD_DIM, RELIABILITY_BIN_DIM]:
        metadata_dict[this_key] = (
            basic_score_table_xarray.coords[this_key].values
        )

    num_times = len(metadata_dict[TIME_DIM])
    num_prob_thresholds = len(metadata_dict[PROBABILITY_THRESHOLD_DIM])
    num_bins_for_reliability = len(metadata_dict[RELIABILITY_BIN_DIM])

    these_dim = (
        TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM, PROBABILITY_THRESHOLD_DIM
    )
    this_array = numpy.full(
        (num_times, 1, 1, num_prob_thresholds), 0, dtype=int
    )
    main_data_dict = {
        NUM_ACTUAL_ORIENTED_TP_KEY: (these_dim, this_array + 0),
        NUM_PREDICTION_ORIENTED_TP_KEY: (these_dim, this_array + 0),
        NUM_FALSE_NEGATIVES_KEY: (these_dim, this_array + 0),
        NUM_FALSE_POSITIVES_KEY: (these_dim, this_array + 0)
    }

    these_dim = (TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM, RELIABILITY_BIN_DIM)
    this_integer_array = numpy.full(
        (num_times, 1, 1, num_bins_for_reliability), 0, dtype=int
    )
    this_float_array = numpy.full(
        (num_times, 1, 1, num_bins_for_reliability), numpy.nan
    )
    new_dict = {
        EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0),
        SUMMED_FORECAST_PROB_KEY: (these_dim, this_float_array + 0.),
        POSITIVE_EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0)
    }
    main_data_dict.update(new_dict)

    these_dim = (TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM)
    this_array = numpy.full((num_times, 1, 1), numpy.nan)
    new_dict = {
        ACTUAL_SSE_KEY: (these_dim, this_array + 0.),
        REFERENCE_SSE_KEY: (these_dim, this_array + 0.)
    }
    main_data_dict.update(new_dict)

    new_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict,
        attrs=basic_score_table_xarray.attrs
    )

    main_keys = [
        NUM_ACTUAL_ORIENTED_TP_KEY, NUM_PREDICTION_ORIENTED_TP_KEY,
        NUM_FALSE_POSITIVES_KEY, NUM_FALSE_NEGATIVES_KEY,
        EXAMPLE_COUNT_KEY, SUMMED_FORECAST_PROB_KEY, POSITIVE_EXAMPLE_COUNT_KEY,
        ACTUAL_SSE_KEY, REFERENCE_SSE_KEY
    ]

    for this_key in main_keys:
        new_score_table_xarray[this_key].values = numpy.nansum(
            basic_score_table_xarray[this_key].values,
            axis=(1, 2), keepdims=True
        )

    return new_score_table_xarray


def get_advanced_scores(basic_score_table_xarray, training_event_freq_matrix):
    """Computes advanced scores from basic scores.

    M = number of rows in grid
    N = number of columns in grid

    :param basic_score_table_xarray: xarray table in format returned by
        `get_basic_scores`.
    :param training_event_freq_matrix: M-by-N numpy array of event frequencies
        ("climatologies") in training data.
    :return: advanced_score_table_xarray: xarray table with advanced scores
        (variable and dimension names should make the table self-explanatory).
    """

    metadata_dict = dict()

    for this_key in [
            TIME_DIM, LATITUDE_DIM, LONGITUDE_DIM,
            PROBABILITY_THRESHOLD_DIM, RELIABILITY_BIN_DIM
    ]:
        metadata_dict[this_key] = (
            basic_score_table_xarray.coords[this_key].values
        )

    num_grid_rows = len(metadata_dict[LATITUDE_DIM])
    num_grid_columns = len(metadata_dict[LONGITUDE_DIM])

    these_dim = numpy.array([num_grid_rows, num_grid_columns], dtype=int)
    error_checking.assert_is_numpy_array(
        training_event_freq_matrix, exact_dimensions=these_dim
    )
    error_checking.assert_is_geq_numpy_array(
        training_event_freq_matrix, 0., allow_nan=True
    )
    error_checking.assert_is_leq_numpy_array(
        training_event_freq_matrix, 1., allow_nan=True
    )

    num_prob_thresholds = len(metadata_dict[PROBABILITY_THRESHOLD_DIM])
    num_bins_for_reliability = len(metadata_dict[RELIABILITY_BIN_DIM])

    these_dim = (LATITUDE_DIM, LONGITUDE_DIM, PROBABILITY_THRESHOLD_DIM)
    this_integer_array = numpy.full(
        (num_grid_rows, num_grid_columns, num_prob_thresholds), 0, dtype=int)

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

    these_dim = (LATITUDE_DIM, LONGITUDE_DIM, RELIABILITY_BIN_DIM)
    this_integer_array = numpy.full(
        (num_grid_rows, num_grid_columns, num_bins_for_reliability),
        0, dtype=int
    )
    this_float_array = numpy.full(
        (num_grid_rows, num_grid_columns, num_bins_for_reliability), numpy.nan
    )
    new_dict = {
        EXAMPLE_COUNT_KEY: (these_dim, this_integer_array + 0),
        MEAN_FORECAST_PROB_KEY: (these_dim, this_float_array + 0.),
        EVENT_FREQUENCY_KEY: (these_dim, this_float_array + 0.)
    }
    main_data_dict.update(new_dict)

    these_dim = (LATITUDE_DIM, LONGITUDE_DIM)
    this_float_array = numpy.full((num_grid_rows, num_grid_columns), numpy.nan)
    new_dict = {
        BRIER_SCORE_KEY: (these_dim, this_float_array + 0.),
        BRIER_SKILL_SCORE_KEY: (these_dim, this_float_array + 0.),
        RELIABILITY_KEY: (these_dim, this_float_array + 0.),
        RESOLUTION_KEY: (these_dim, this_float_array + 0.),
        FSS_KEY: (these_dim, this_float_array + 0.),
        TRAINING_EVENT_FREQ_KEY: (these_dim, training_event_freq_matrix + 0.)
    }
    main_data_dict.update(new_dict)

    advanced_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict,
        attrs=basic_score_table_xarray.attrs
    )

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            for k in range(num_prob_thresholds):
                t = basic_score_table_xarray

                this_contingency_table = {
                    NUM_ACTUAL_ORIENTED_TP_KEY: numpy.sum(
                        t[NUM_ACTUAL_ORIENTED_TP_KEY].values[:, i, j, k]
                    ),
                    NUM_PREDICTION_ORIENTED_TP_KEY: numpy.sum(
                        t[NUM_PREDICTION_ORIENTED_TP_KEY].values[:, i, j, k]
                    ),
                    NUM_FALSE_POSITIVES_KEY: numpy.sum(
                        t[NUM_FALSE_POSITIVES_KEY].values[:, i, j, k]
                    ),
                    NUM_FALSE_NEGATIVES_KEY: numpy.sum(
                        t[NUM_FALSE_NEGATIVES_KEY].values[:, i, j, k]
                    )
                }

                for this_key in [
                        NUM_ACTUAL_ORIENTED_TP_KEY,
                        NUM_PREDICTION_ORIENTED_TP_KEY,
                        NUM_FALSE_POSITIVES_KEY, NUM_FALSE_NEGATIVES_KEY
                ]:
                    advanced_score_table_xarray[this_key].values[i, j, k] = (
                        this_contingency_table[this_key]
                    )

                advanced_score_table_xarray[POD_KEY].values[i, j, k] = (
                    _get_pod(this_contingency_table)
                )
                advanced_score_table_xarray[SUCCESS_RATIO_KEY].values[
                    i, j, k
                ] = _get_success_ratio(this_contingency_table)

                advanced_score_table_xarray[CSI_KEY].values[i, j, k] = (
                    _get_csi(this_contingency_table)
                )
                advanced_score_table_xarray[FREQUENCY_BIAS_KEY].values[
                    i, j, k
                ] = _get_frequency_bias(this_contingency_table)

            for k in range(num_bins_for_reliability):
                t = basic_score_table_xarray

                this_num_examples = numpy.sum(
                    t[EXAMPLE_COUNT_KEY].values[:, i, j, k]
                )
                if this_num_examples == 0:
                    continue

                this_sum_forecast_probs = numpy.nansum(
                    t[SUMMED_FORECAST_PROB_KEY].values[:, i, j, k]
                )
                this_num_positive_examples = numpy.sum(
                    t[POSITIVE_EXAMPLE_COUNT_KEY].values[:, i, j, k]
                )

                advanced_score_table_xarray[EXAMPLE_COUNT_KEY].values[
                    i, j, k
                ] = this_num_examples

                advanced_score_table_xarray[MEAN_FORECAST_PROB_KEY].values[
                    i, j, k
                ] = this_sum_forecast_probs / this_num_examples

                advanced_score_table_xarray[EVENT_FREQUENCY_KEY].values[
                    i, j, k
                ] = float(this_num_positive_examples) / this_num_examples

            a = advanced_score_table_xarray

            if numpy.any(a[EXAMPLE_COUNT_KEY].values[i, j, :] > 0):
                this_bss_dict = gg_model_eval.get_brier_skill_score(
                    mean_forecast_prob_by_bin=
                    a[MEAN_FORECAST_PROB_KEY].values[i, j, :],
                    mean_observed_label_by_bin=
                    a[EVENT_FREQUENCY_KEY].values[i, j, :],
                    num_examples_by_bin=a[EXAMPLE_COUNT_KEY].values[i, j, :],
                    climatology=a[TRAINING_EVENT_FREQ_KEY].values[i, j]
                )

                a[BRIER_SKILL_SCORE_KEY].values[i, j] = (
                    this_bss_dict[gg_model_eval.BSS_KEY]
                )
                a[BRIER_SCORE_KEY].values[i, j] = (
                    this_bss_dict[gg_model_eval.BRIER_SCORE_KEY]
                )
                a[RELIABILITY_KEY].values[i, j] = (
                    this_bss_dict[gg_model_eval.RELIABILITY_KEY]
                )
                a[RESOLUTION_KEY].values[i, j] = (
                    this_bss_dict[gg_model_eval.RESOLUTION_KEY]
                )

                advanced_score_table_xarray = a

            this_actual_sse = numpy.sum(
                basic_score_table_xarray[ACTUAL_SSE_KEY].values[:, i, j]
            )
            this_reference_sse = numpy.sum(
                basic_score_table_xarray[REFERENCE_SSE_KEY].values[:, i, j]
            )
            advanced_score_table_xarray[FSS_KEY].values[i, j] = (
                1. - this_actual_sse / this_reference_sse
            )

    return advanced_score_table_xarray


def find_basic_score_file(top_directory_name, valid_date_string,
                          raise_error_if_missing=True):
    """Finds Pickle file with basic evaluation scores.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: basic_score_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    error_checking.assert_is_boolean(raise_error_if_missing)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    basic_score_file_name = '{0:s}/{1:s}/basic_scores_{2:s}.p'.format(
        top_directory_name, valid_date_string[:4], valid_date_string
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
        top_directory_name, first_date_string, last_date_string,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False,
        test_mode=False):
    """Finds many Pickle files with evaluation scores.

    :param top_directory_name: See doc for `find_basic_score_file`.
    :param first_date_string: First valid date (format "yyyymmdd").
    :param last_date_string: Last valid date (format "yyyymmdd").
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
            valid_date_string=this_date_string,
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


def find_advanced_score_file(
        directory_name, month=None, hour=None, raise_error_if_missing=True):
    """Finds Pickle file with advanced evaluation scores.

    :param directory_name: Name of directory where file is expected.
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

    advanced_score_file_name += '.p'

    if os.path.isfile(advanced_score_file_name) or not raise_error_if_missing:
        return advanced_score_file_name

    error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
        advanced_score_file_name
    )
    raise ValueError(error_string)


def write_file(score_table_xarray, pickle_file_name):
    """Writes evaluation scores to Pickle file.

    :param score_table_xarray: xarray table created by `get_basic_scores` or
        `get_advanced_scores`.
    :param pickle_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(score_table_xarray, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads evaluation scores from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: score_table_xarray: xarray table created by `get_basic_scores` or
        `get_advanced_scores`.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    score_table_xarray = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    return score_table_xarray
