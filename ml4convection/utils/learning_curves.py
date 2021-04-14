"""Learning curves (simple model evaluation)."""

import copy
import numpy
import xarray
import tensorflow
from keras import backend as K
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import prediction_io
from ml4convection.utils import general_utils
from ml4convection.utils import fourier_utils
from ml4convection.machine_learning import neural_net

# EPSILON = numpy.finfo(float).eps
GRID_SPACING_DEG = 0.0125

TIME_DIM = 'valid_time_unix_sec'
MIN_RESOLUTION_DIM = 'min_fourier_resolution_deg'
MAX_RESOLUTION_DIM = 'max_fourier_resolution_deg'
NEIGH_DISTANCE_DIM = 'neigh_distance_px'

MODEL_FILE_KEY = 'model_file_name'

NEIGH_BRIER_SSE_KEY = 'neigh_brier_sse'
NEIGH_BRIER_NUM_VALS_KEY = 'neigh_brier_num_values'
NEIGH_FSS_ACTUAL_SSE_KEY = 'neigh_fss_actual_sse'
NEIGH_FSS_REFERENCE_SSE_KEY = 'neigh_fss_reference_sse'
NEIGH_IOU_INTERSECTION_KEY = 'neigh_iou_intersection'
NEIGH_IOU_UNION_KEY = 'neigh_iou_union'
NEIGH_DICE_INTERSECTION_KEY = 'neigh_dice_intersection'
NEIGH_DICE_NUM_PIX_KEY = 'neigh_dice_num_pixels'
NEIGH_PRED_ORIENTED_TP_KEY = 'neigh_pred_oriented_true_positives'
NEIGH_OBS_ORIENTED_TP_KEY = 'neigh_obs_oriented_true_positives'
NEIGH_FALSE_POSITIVES_KEY = 'neigh_false_positives'
NEIGH_FALSE_NEGATIVES_KEY = 'neigh_false_negatives'

FREQ_SSE_REAL_KEY = 'freq_space_sse_real'
FREQ_SSE_IMAGINARY_KEY = 'freq_space_sse_imaginary'
FREQ_SSE_TOTAL_KEY = 'freq_space_sse_total'
FREQ_SSE_NUM_WEIGHTS_KEY = 'freq_space_sse_num_weights'

FOURIER_BRIER_SSE_KEY = 'fourier_brier_sse'
FOURIER_BRIER_NUM_VALS_KEY = 'fourier_brier_num_values'
FOURIER_FSS_ACTUAL_SSE_KEY = 'fourier_fss_actual_sse'
FOURIER_FSS_REFERENCE_SSE_KEY = 'fourier_fss_reference_sse'
FOURIER_IOU_INTERSECTION_KEY = 'fourier_iou_intersection'
FOURIER_IOU_UNION_KEY = 'fourier_iou_union'
FOURIER_DICE_INTERSECTION_KEY = 'fourier_dice_intersection'
FOURIER_DICE_NUM_PIX_KEY = 'fourier_dice_num_pixels'
FOURIER_CSI_NUMERATOR_KEY = 'fourier_csi_numerator'
FOURIER_CSI_DENOMINATOR_KEY = 'fourier_csi_denominator'

NEIGH_BRIER_SCORE_KEY = 'neigh_brier_score'
NEIGH_FSS_KEY = 'neigh_fss'
NEIGH_IOU_KEY = 'neigh_iou'
NEIGH_DICE_COEFF_KEY = 'neigh_dice_coeff'
NEIGH_CSI_KEY = 'neigh_csi'

FREQ_MSE_REAL_KEY = 'freq_space_mse_real'
FREQ_MSE_IMAGINARY_KEY = 'freq_space_mse_imaginary'
FREQ_MSE_TOTAL_KEY = 'freq_space_mse_total'
FOURIER_BRIER_SCORE_KEY = 'fourier_brier_score'
FOURIER_FSS_KEY = 'fourier_fss'
FOURIER_IOU_KEY = 'fourier_iou'
FOURIER_DICE_COEFF_KEY = 'fourier_dice_coeff'
FOURIER_CSI_KEY = 'fourier_csi'


def _apply_max_filter(input_matrix, half_width_px):
    """Applies max-filter to 2-D matrix.

    :param input_matrix: 2-D numpy array.
    :param half_width_px: Half-width of filter (pixels).
    :return: output_matrix: Max-filtered version of `input_matrix` with same
        dimensions.
    """

    structure_matrix = general_utils.get_structure_matrix(half_width_px)
    structure_matrix = numpy.maximum(structure_matrix, 1)

    output_matrix = maximum_filter(
        input_matrix.astype(float),
        footprint=structure_matrix, mode='constant', cval=0.
    )

    return output_matrix.astype(input_matrix.dtype)


def _dilate_binary_matrix(binary_matrix, half_width_px):
    """Dilates binary matrix with a square filter.

    :param binary_matrix: See doc for `general_utils.check_2d_binary_matrix`.
    :param half_width_px: Half-width of filter (pixels).
    :return: dilated_binary_matrix: Dilated version of input.
    """

    general_utils.check_2d_binary_matrix(binary_matrix)
    structure_matrix = general_utils.get_structure_matrix(half_width_px)
    structure_matrix = numpy.maximum(structure_matrix, 1)

    dilated_binary_matrix = binary_dilation(
        binary_matrix.astype(int), structure=structure_matrix, iterations=1,
        border_value=0
    )
    return dilated_binary_matrix.astype(binary_matrix.dtype)


def _erode_binary_matrix(binary_matrix, half_width_px):
    """Erodes binary matrix with a square filter.

    :param binary_matrix: See doc for `general_utils.check_2d_binary_matrix`.
    :param half_width_px: Half-width of filter (pixels).
    :return: eroded_binary_matrix: Eroded version of input.
    """

    general_utils.check_2d_binary_matrix(binary_matrix)
    structure_matrix = general_utils.get_structure_matrix(half_width_px)
    structure_matrix = numpy.maximum(structure_matrix, 1)

    eroded_binary_matrix = binary_erosion(
        binary_matrix.astype(int), structure=structure_matrix, iterations=1,
        border_value=1
    )
    return eroded_binary_matrix.astype(binary_matrix.dtype)


def _apply_fourier_filter(orig_data_matrix, filter_matrix):
    """Filters spatial data via Fourier decomposition.

    M = number of rows in grid
    N = number of columns in grid

    :param orig_data_matrix: M-by-N numpy array of spatial data.
    :param filter_matrix: M-by-N numpy array of filter weights for Fourier
        coefficients.
    :return: filtered_data_matrix: M-by-N numpy array of filtered spatial data.
    :return: fourier_weight_matrix: M-by-N numpy array of complex weights.
    """

    orig_data_tensor = tensorflow.constant(
        numpy.expand_dims(orig_data_matrix, axis=0),
        dtype=tensorflow.complex128
    )

    fourier_weight_tensor = tensorflow.signal.fft2d(orig_data_tensor)
    fourier_weight_matrix = K.eval(fourier_weight_tensor)[0, ...]
    fourier_weight_matrix = fourier_weight_matrix * filter_matrix

    fourier_weight_tensor = tensorflow.constant(
        numpy.expand_dims(fourier_weight_matrix, axis=0),
        dtype=tensorflow.complex128
    )

    filtered_data_tensor = tensorflow.signal.ifft2d(fourier_weight_tensor)
    filtered_data_tensor = tensorflow.math.real(filtered_data_tensor)
    return K.eval(filtered_data_tensor)[0, ...], fourier_weight_matrix


def _get_freq_mse_components_one_time(
        actual_weight_matrix, predicted_weight_matrix):
    """Computes components of frequency-space MSE for one time step.

    M = number of rows in grid
    N = number of columns in grid

    :param actual_weight_matrix: M-by-N numpy array of complex Fourier weights
        for target values.
    :param predicted_weight_matrix: M-by-N numpy array of complex Fourier
        weights for predicted probabilities.
    :return: real_part_sse: Sum of squared errors (SSE) for real part of Fourier
        weights.
    :return: imaginary_part_sse: SSE for imaginary part.
    :return: total_sse: SSE for total (real and imaginary).
    :return: num_weights: Number of weights used to compute each SSE.
    """

    real_part_sse = numpy.sum(
        (numpy.real(actual_weight_matrix) -
         numpy.real(predicted_weight_matrix)) ** 2
    )
    imaginary_part_sse = numpy.sum(
        (numpy.imag(actual_weight_matrix) -
         numpy.imag(predicted_weight_matrix)) ** 2
    )
    total_sse = numpy.sum(
        numpy.abs(actual_weight_matrix - predicted_weight_matrix) ** 2
    )

    return (
        real_part_sse, imaginary_part_sse, total_sse, actual_weight_matrix.size
    )


def _get_brier_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix,
        matching_distance_px):
    """Computes Brier-score components for one time step.

    M = number of rows in grid
    N = number of columns in grid

    If doing neighbourhood-based evaluation, this method assumes that
    `eval_mask_matrix` is already eroded for the given matching distance.

    If doing Fourier-based evaluation, this method assumes that
    `actual_target_matrix` and `probability_matrix` have already gone through
    Fourier decomposition and recomposition.

    :param actual_target_matrix: M-by-N numpy array (see doc for
        `general_utils.check_2d_binary_matrix`), indicating where actual
        convection occurs.
    :param probability_matrix: M-by-N numpy array of forecast convection
        probabilities.
    :param eval_mask_matrix: M-by-N numpy array (see doc for
        `general_utils.check_2d_binary_matrix`), indicating which pixels are to
        be used for evaluation.
    :param matching_distance_px: Matching distance (pixels) for
        neighbourhood-based evaluation.  If doing Fourier-based
        evaluation, make this None.
    :return: sum_of_squared_errors: Sum of squared errors (SSE).
    :return: num_values: Number of values used to compute SSE.
    """

    if matching_distance_px is None:
        this_actual_target_matrix = actual_target_matrix
    else:
        this_actual_target_matrix = _dilate_binary_matrix(
            binary_matrix=actual_target_matrix,
            half_width_px=matching_distance_px
        ).astype(int)

    squared_error_matrix = (this_actual_target_matrix - probability_matrix) ** 2
    squared_error_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan

    sum_of_squared_errors = numpy.nansum(squared_error_matrix)
    num_values = numpy.sum(numpy.invert(numpy.isnan(squared_error_matrix)))

    return sum_of_squared_errors, num_values


def _get_fss_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix,
        matching_distance_px):
    """Computes components of fractions skill score (FSS) for one time step.

    See notes on neighbourhood-based vs. Fourier-based evaluation in
    `_get_brier_components_one_time`.

    :param actual_target_matrix: See doc for `_get_brier_components_one_time`.
    :param probability_matrix: Same.
    :param eval_mask_matrix: Same.
    :param matching_distance_px: Same.
    :return: actual_sse: Actual sum of squared errors.
    :return: reference_sse: Reference sum of squared errors.
    """

    if matching_distance_px is None:
        this_target_matrix = copy.deepcopy(actual_target_matrix)
        this_prob_matrix = copy.deepcopy(probability_matrix)
    else:
        structure_matrix = general_utils.get_structure_matrix(
            numpy.round(matching_distance_px)
        )
        weight_matrix = numpy.full(
            structure_matrix.shape, 1. / structure_matrix.size
        )

        this_target_matrix = convolve2d(
            actual_target_matrix.astype(float), weight_matrix, mode='same'
        )
        this_prob_matrix = convolve2d(
            probability_matrix, weight_matrix, mode='same'
        )

    this_prob_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    actual_sse = numpy.nansum((this_target_matrix - this_prob_matrix) ** 2)
    reference_sse = numpy.nansum(
        this_target_matrix ** 2 + this_prob_matrix ** 2
    )

    return actual_sse, reference_sse


def _get_iou_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix,
        matching_distance_px):
    """Computes components of intersection over union (IOU) for one time step.

    See notes on neighbourhood-based vs. Fourier-based evaluation in
    `_get_brier_components_one_time`.

    :param actual_target_matrix: See doc for `_get_brier_components_one_time`.
    :param probability_matrix: Same.
    :param eval_mask_matrix: Same.
    :param matching_distance_px: Same.
    :return: intersection: Intersection (numerator of IOU).
    :return: union: Union (denominator of IOU).
    """

    if matching_distance_px is None:
        this_actual_target_matrix = actual_target_matrix
    else:
        this_actual_target_matrix = _dilate_binary_matrix(
            binary_matrix=actual_target_matrix,
            half_width_px=matching_distance_px
        ).astype(int)

    product_matrix = probability_matrix * this_actual_target_matrix
    product_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    intersection = numpy.nansum(product_matrix)

    max_matrix = numpy.maximum(probability_matrix, this_actual_target_matrix)
    max_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    union = numpy.nansum(max_matrix)

    return intersection, union


def _get_dice_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix,
        matching_distance_px):
    """Computes components of Dice coefficient for one time step.

    See notes on neighbourhood-based vs. Fourier-based evaluation in
    `_get_brier_components_one_time`.

    :param actual_target_matrix: See doc for `_get_brier_components_one_time`.
    :param probability_matrix: Same.
    :param eval_mask_matrix: Same.
    :param matching_distance_px: Same.
    :return: intersection: Intersection (numerator of Dice coefficient).
    :return: num_pixels: Number of pixels (denominator of Dice coefficient).
    """

    if matching_distance_px is None:
        this_actual_target_matrix = actual_target_matrix
    else:
        this_actual_target_matrix = _dilate_binary_matrix(
            binary_matrix=actual_target_matrix,
            half_width_px=matching_distance_px
        ).astype(int)

    positive_product_matrix = probability_matrix * this_actual_target_matrix
    positive_product_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan

    negative_product_matrix = (
        (1 - probability_matrix) * (1 - this_actual_target_matrix)
    )
    negative_product_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan

    intersection = (
        numpy.nansum(positive_product_matrix) +
        numpy.nansum(negative_product_matrix)
    )
    num_pixels = numpy.sum(numpy.invert(numpy.isnan(negative_product_matrix)))

    return intersection, num_pixels


def _get_fourier_csi_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix):
    """Computes components of Fourier-based CSI for one time step.

    See notes on neighbourhood-based vs. Fourier-based evaluation in
    `_get_brier_components_one_time`.

    :param actual_target_matrix: See doc for `_get_brier_components_one_time`.
    :param probability_matrix: Same.
    :param eval_mask_matrix: Same.
    :return: numerator: Number of Fourier-based CSI (number of true
        positives).
    :return: denominator: Denominator of Fourier-based CSI (number of
        true positives + false positives + false negatives).
    """

    this_matrix = probability_matrix * actual_target_matrix
    this_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    num_true_positives = numpy.nansum(this_matrix)

    this_matrix = probability_matrix * (1 - actual_target_matrix)
    this_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    num_false_positives = numpy.nansum(this_matrix)

    this_matrix = (1 - probability_matrix) * actual_target_matrix
    this_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    num_false_negatives = numpy.nansum(this_matrix)

    denominator = num_true_positives + num_false_positives + num_false_negatives

    return num_true_positives, denominator


def _get_neigh_csi_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix,
        matching_distance_px):
    """Computes components of neighbourhood-based CSI for one time step.

    See notes on neighbourhood-based vs. Fourier-based evaluation in
    `_get_brier_components_one_time`.

    :param actual_target_matrix: See doc for `_get_brier_components_one_time`.
    :param probability_matrix: Same.
    :param eval_mask_matrix: Same.
    :param matching_distance_px: Same.
    :return: num_prediction_oriented_tp: Number of prediction-oriented true
        positives.
    :return: num_obs_oriented_tp: Number of observation-oriented true positives.
    :return: num_false_positives: Number of false positives.
    :return: num_false_negatives: Number of false negatives.
    """

    filtered_target_matrix = _apply_max_filter(
        input_matrix=actual_target_matrix, half_width_px=matching_distance_px
    )
    filtered_prob_matrix = _apply_max_filter(
        input_matrix=probability_matrix, half_width_px=matching_distance_px
    )

    filtered_prob_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    num_obs_oriented_tp = numpy.nansum(
        filtered_prob_matrix * actual_target_matrix
    )
    num_false_negatives = numpy.nansum(
        (1 - filtered_prob_matrix) * actual_target_matrix
    )

    filtered_target_matrix = filtered_target_matrix.astype(float)
    filtered_target_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    num_prediction_oriented_tp = numpy.nansum(
        filtered_target_matrix * probability_matrix
    )
    num_false_positives = numpy.nansum(
        (1 - filtered_target_matrix) * probability_matrix
    )

    return (
        num_prediction_oriented_tp, num_obs_oriented_tp,
        num_false_positives, num_false_negatives
    )


def _find_eval_mask(prediction_dict):
    """Finds evaluation mask for set of predictions.

    M = number of rows in grid
    N = number of columns in grid

    :param prediction_dict: Dictionary with predicted and actual values (in
        format returned by `prediction_io.read_file`).
    :return: mask_matrix: M-by-N numpy array of Boolean flags, where True means
        the grid cell is unmasked.
    :return: model_file_name: Path to model that generated predictions.
    """

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    metafile_name = neural_net.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(
        metafile_name
    ))
    metadata_dict = neural_net.read_metafile(metafile_name)
    mask_matrix = metadata_dict[neural_net.MASK_MATRIX_KEY]

    return mask_matrix, model_file_name


def get_basic_scores(
        prediction_dict, neigh_distances_px, min_fourier_resolutions_deg,
        max_fourier_resolutions_deg, test_mode=False, eval_mask_matrix=None,
        model_file_name=None):
    """Computes basic scores.

    D = number of matching distances for neighbourhood-based evaluation
    R = number of resolution bands for Fourier-based evaluation

    :param prediction_dict: Dictionary with predicted and actual values (in
        format returned by `prediction_io.read_file`).
    :param neigh_distances_px: length-D numpy array of matching distances.  If
        you do not want neighbourhood-based evaluation, make this None.
    :param min_fourier_resolutions_deg: length-R numpy array of minimum
        resolutions (degrees).  If you do not want Fourier-based
        evaluation, make this None.
    :param max_fourier_resolutions_deg: Same but with max resolutions.
    :param test_mode: Leave this alone.
    :param eval_mask_matrix: Leave this alone.
    :param model_file_name: Leave this alone.
    :return: basic_score_table_xarray: xarray table with results (variable
        and dimension names should make the table self-explanatory).
    """

    if neigh_distances_px is None:
        num_neigh_distances = 0
    else:
        error_checking.assert_is_numpy_array(
            neigh_distances_px, num_dimensions=1
        )
        error_checking.assert_is_geq_numpy_array(neigh_distances_px, 0.)

        num_neigh_distances = len(neigh_distances_px)

    if (
            min_fourier_resolutions_deg is None
            or max_fourier_resolutions_deg is None
    ):
        num_fourier_bands = 0
    else:
        error_checking.assert_is_numpy_array(
            min_fourier_resolutions_deg, num_dimensions=1
        )

        num_fourier_bands = len(min_fourier_resolutions_deg)
        expected_dim = numpy.array([num_fourier_bands], dtype=int)
        error_checking.assert_is_numpy_array(
            max_fourier_resolutions_deg, exact_dimensions=expected_dim
        )

    error_checking.assert_is_boolean(test_mode)

    if not test_mode:
        eval_mask_matrix, model_file_name = _find_eval_mask(prediction_dict)

    valid_times_unix_sec = prediction_dict[prediction_io.VALID_TIMES_KEY]
    num_times = len(valid_times_unix_sec)

    metadata_dict = {TIME_DIM: valid_times_unix_sec}
    main_data_dict = dict()

    if num_neigh_distances > 0:
        metadata_dict[NEIGH_DISTANCE_DIM] = neigh_distances_px

        these_dim = (TIME_DIM, NEIGH_DISTANCE_DIM)
        this_array = numpy.full((num_times, num_neigh_distances), numpy.nan)

        new_dict = {
            NEIGH_BRIER_SSE_KEY: (these_dim, this_array + 0),
            NEIGH_BRIER_NUM_VALS_KEY: (these_dim, this_array + 0.),
            NEIGH_FSS_ACTUAL_SSE_KEY: (these_dim, this_array + 0.),
            NEIGH_FSS_REFERENCE_SSE_KEY: (these_dim, this_array + 0.),
            NEIGH_IOU_INTERSECTION_KEY: (these_dim, this_array + 0.),
            NEIGH_IOU_UNION_KEY: (these_dim, this_array + 0.),
            NEIGH_DICE_INTERSECTION_KEY: (these_dim, this_array + 0.),
            NEIGH_DICE_NUM_PIX_KEY: (these_dim, this_array + 0.),
            NEIGH_PRED_ORIENTED_TP_KEY: (these_dim, this_array + 0.),
            NEIGH_OBS_ORIENTED_TP_KEY: (these_dim, this_array + 0.),
            NEIGH_FALSE_POSITIVES_KEY: (these_dim, this_array + 0.),
            NEIGH_FALSE_NEGATIVES_KEY: (these_dim, this_array + 0.)
        }
        main_data_dict.update(new_dict)

    if num_fourier_bands > 0:
        metadata_dict[MIN_RESOLUTION_DIM] = min_fourier_resolutions_deg
        metadata_dict[MAX_RESOLUTION_DIM] = max_fourier_resolutions_deg

        these_dim = (TIME_DIM, MIN_RESOLUTION_DIM)
        this_array = numpy.full((num_times, num_fourier_bands), numpy.nan)

        new_dict = {
            FREQ_SSE_REAL_KEY: (these_dim, this_array + 0),
            FREQ_SSE_IMAGINARY_KEY: (these_dim, this_array + 0),
            FREQ_SSE_TOTAL_KEY: (these_dim, this_array + 0),
            FREQ_SSE_NUM_WEIGHTS_KEY: (these_dim, this_array + 0),
            FOURIER_BRIER_SSE_KEY: (these_dim, this_array + 0),
            FOURIER_BRIER_NUM_VALS_KEY: (these_dim, this_array + 0.),
            FOURIER_FSS_ACTUAL_SSE_KEY: (these_dim, this_array + 0.),
            FOURIER_FSS_REFERENCE_SSE_KEY: (these_dim, this_array + 0.),
            FOURIER_IOU_INTERSECTION_KEY: (these_dim, this_array + 0.),
            FOURIER_IOU_UNION_KEY: (these_dim, this_array + 0.),
            FOURIER_DICE_INTERSECTION_KEY: (these_dim, this_array + 0.),
            FOURIER_DICE_NUM_PIX_KEY: (these_dim, this_array + 0.),
            FOURIER_CSI_NUMERATOR_KEY: (these_dim, this_array + 0.),
            FOURIER_CSI_DENOMINATOR_KEY: (these_dim, this_array + 0.)
        }
        main_data_dict.update(new_dict)

    basic_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    basic_score_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    b = basic_score_table_xarray

    if num_neigh_distances > 0:
        this_matrix = (
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...]
        )

        num_grid_rows = this_matrix.shape[0]
        num_grid_columns = this_matrix.shape[1]
        eroded_eval_mask_matrix = numpy.full(
            (num_neigh_distances, num_grid_rows, num_grid_columns),
            0, dtype=bool
        )

        for k in range(num_neigh_distances):
            eroded_eval_mask_matrix[k, ...] = _erode_binary_matrix(
                binary_matrix=eval_mask_matrix,
                half_width_px=neigh_distances_px[k]
            )

        for i in range(num_times):
            this_prob_matrix = (
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...]
                + 0.
            )
            this_target_matrix = (
                prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...] + 0
            )
            this_target_matrix = this_target_matrix.astype(float)

            for k in range(num_neigh_distances):
                (
                    b[NEIGH_BRIER_SSE_KEY].values[i, k],
                    b[NEIGH_BRIER_NUM_VALS_KEY].values[i, k]
                ) = _get_brier_components_one_time(
                    actual_target_matrix=this_target_matrix,
                    probability_matrix=this_prob_matrix,
                    eval_mask_matrix=eroded_eval_mask_matrix[k, ...],
                    matching_distance_px=neigh_distances_px[k]
                )

                (
                    b[NEIGH_FSS_ACTUAL_SSE_KEY].values[i, k],
                    b[NEIGH_FSS_REFERENCE_SSE_KEY].values[i, k]
                ) = _get_fss_components_one_time(
                    actual_target_matrix=this_target_matrix,
                    probability_matrix=this_prob_matrix,
                    eval_mask_matrix=eroded_eval_mask_matrix[k, ...],
                    matching_distance_px=neigh_distances_px[k]
                )

                (
                    b[NEIGH_IOU_INTERSECTION_KEY].values[i, k],
                    b[NEIGH_IOU_UNION_KEY].values[i, k]
                ) = _get_iou_components_one_time(
                    actual_target_matrix=this_target_matrix,
                    probability_matrix=this_prob_matrix,
                    eval_mask_matrix=eroded_eval_mask_matrix[k, ...],
                    matching_distance_px=neigh_distances_px[k]
                )

                (
                    b[NEIGH_DICE_INTERSECTION_KEY].values[i, k],
                    b[NEIGH_DICE_NUM_PIX_KEY].values[i, k]
                ) = _get_dice_components_one_time(
                    actual_target_matrix=this_target_matrix,
                    probability_matrix=this_prob_matrix,
                    eval_mask_matrix=eroded_eval_mask_matrix[k, ...],
                    matching_distance_px=neigh_distances_px[k]
                )

                (
                    b[NEIGH_PRED_ORIENTED_TP_KEY].values[i, k],
                    b[NEIGH_OBS_ORIENTED_TP_KEY].values[i, k],
                    b[NEIGH_FALSE_POSITIVES_KEY].values[i, k],
                    b[NEIGH_FALSE_NEGATIVES_KEY].values[i, k]
                ) = _get_neigh_csi_components_one_time(
                    actual_target_matrix=this_target_matrix,
                    probability_matrix=this_prob_matrix,
                    eval_mask_matrix=eroded_eval_mask_matrix[k, ...],
                    matching_distance_px=neigh_distances_px[k]
                )

    if num_fourier_bands > 0:
        this_matrix = fourier_utils.taper_spatial_data(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...]
        )

        num_tapered_rows = this_matrix.shape[0]
        num_tapered_columns = this_matrix.shape[1]
        these_dim = (num_fourier_bands, num_tapered_rows, num_tapered_columns)

        blackman_window_matrix = numpy.full(these_dim, numpy.nan)
        butterworth_filter_matrix = numpy.full(these_dim, numpy.nan)

        for k in range(num_fourier_bands):
            blackman_window_matrix[k, ...] = (
                fourier_utils.apply_blackman_window(
                    numpy.ones(blackman_window_matrix[k, ...].shape)
                )
            )

            butterworth_filter_matrix[k, ...] = (
                fourier_utils.apply_butterworth_filter(
                    coefficient_matrix=
                    numpy.ones(butterworth_filter_matrix[k, ...].shape),
                    filter_order=2., grid_spacing_metres=GRID_SPACING_DEG,
                    min_resolution_metres=min_fourier_resolutions_deg[k],
                    max_resolution_metres=max_fourier_resolutions_deg[k]
                )
            )

        for i in range(num_times):
            tapered_prob_matrix = fourier_utils.taper_spatial_data(
                prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][i, ...]
            )
            tapered_target_matrix = fourier_utils.taper_spatial_data(
                prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...]
            )
            tapered_target_matrix = tapered_target_matrix.astype(float)

            for k in range(num_fourier_bands):
                (
                    this_filtered_prob_matrix, this_predicted_weight_matrix
                ) = _apply_fourier_filter(
                    orig_data_matrix=
                    tapered_prob_matrix * blackman_window_matrix[k, ...],
                    filter_matrix=butterworth_filter_matrix[k, ...]
                )

                this_filtered_prob_matrix = fourier_utils.untaper_spatial_data(
                    this_filtered_prob_matrix
                )

                (
                    this_filtered_target_matrix, this_actual_weight_matrix
                ) = _apply_fourier_filter(
                    orig_data_matrix=
                    tapered_target_matrix * blackman_window_matrix[k, ...],
                    filter_matrix=butterworth_filter_matrix[k, ...]
                )

                this_filtered_target_matrix = (
                    fourier_utils.untaper_spatial_data(
                        this_filtered_target_matrix
                    )
                )

                (
                    b[FREQ_SSE_REAL_KEY].values[i, k],
                    b[FREQ_SSE_IMAGINARY_KEY].values[i, k],
                    b[FREQ_SSE_TOTAL_KEY].values[i, k],
                    b[FREQ_SSE_NUM_WEIGHTS_KEY].values[i, k]
                ) = _get_freq_mse_components_one_time(
                    actual_weight_matrix=this_actual_weight_matrix,
                    predicted_weight_matrix=this_predicted_weight_matrix
                )

                (
                    b[FOURIER_BRIER_SSE_KEY].values[i, k],
                    b[FOURIER_BRIER_NUM_VALS_KEY].values[i, k]
                ) = _get_brier_components_one_time(
                    actual_target_matrix=this_filtered_target_matrix,
                    probability_matrix=this_filtered_prob_matrix,
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[FOURIER_FSS_ACTUAL_SSE_KEY].values[i, k],
                    b[FOURIER_FSS_REFERENCE_SSE_KEY].values[i, k]
                ) = _get_fss_components_one_time(
                    actual_target_matrix=this_filtered_target_matrix,
                    probability_matrix=this_filtered_prob_matrix,
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[FOURIER_IOU_INTERSECTION_KEY].values[i, k],
                    b[FOURIER_IOU_UNION_KEY].values[i, k]
                ) = _get_iou_components_one_time(
                    actual_target_matrix=this_filtered_target_matrix,
                    probability_matrix=this_filtered_prob_matrix,
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[FOURIER_DICE_INTERSECTION_KEY].values[i, k],
                    b[FOURIER_DICE_NUM_PIX_KEY].values[i, k]
                ) = _get_dice_components_one_time(
                    actual_target_matrix=this_filtered_target_matrix,
                    probability_matrix=this_filtered_prob_matrix,
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[FOURIER_CSI_NUMERATOR_KEY].values[i, k],
                    b[FOURIER_CSI_DENOMINATOR_KEY].values[i, k]
                ) = _get_fourier_csi_components_one_time(
                    actual_target_matrix=this_filtered_target_matrix,
                    probability_matrix=this_filtered_prob_matrix,
                    eval_mask_matrix=eval_mask_matrix,
                )

    basic_score_table_xarray = b
    return basic_score_table_xarray


def get_advanced_scores(basic_score_table_xarray):
    """Computes advanced scores.

    :param basic_score_table_xarray: xarray table created by `get_basic_scores`.
    :return: advanced_score_table_xarray: xarray table with advanced scores
        (variable and dimension names should make the table self-explanatory).
    """

    metadata_dict = dict()

    for this_key in basic_score_table_xarray.coords:
        if this_key == TIME_DIM:
            continue

        metadata_dict[this_key] = (
            basic_score_table_xarray.coords[this_key].values
        )

    if NEIGH_DISTANCE_DIM in metadata_dict:
        num_neigh_distances = len(metadata_dict[NEIGH_DISTANCE_DIM])
    else:
        num_neigh_distances = 0

    if MIN_RESOLUTION_DIM in metadata_dict:
        num_fourier_bands = len(metadata_dict[MIN_RESOLUTION_DIM])
    else:
        num_fourier_bands = 0

    main_data_dict = dict()

    if num_neigh_distances > 0:
        these_dim = (NEIGH_DISTANCE_DIM,)
        this_array = numpy.full(num_neigh_distances, numpy.nan)

        new_dict = {
            NEIGH_BRIER_SCORE_KEY: (these_dim, this_array + 0.),
            NEIGH_FSS_KEY: (these_dim, this_array + 0.),
            NEIGH_IOU_KEY: (these_dim, this_array + 0.),
            NEIGH_DICE_COEFF_KEY: (these_dim, this_array + 0.),
            NEIGH_CSI_KEY: (these_dim, this_array + 0.)
        }
        main_data_dict.update(new_dict)

    if num_fourier_bands > 0:
        these_dim = (MIN_RESOLUTION_DIM,)
        this_array = numpy.full(num_fourier_bands, numpy.nan)

        new_dict = {
            FREQ_MSE_REAL_KEY: (these_dim, this_array + 0.),
            FREQ_MSE_IMAGINARY_KEY: (these_dim, this_array + 0.),
            FREQ_MSE_TOTAL_KEY: (these_dim, this_array + 0.),
            FOURIER_BRIER_SCORE_KEY: (these_dim, this_array + 0.),
            FOURIER_FSS_KEY: (these_dim, this_array + 0.),
            FOURIER_IOU_KEY: (these_dim, this_array + 0.),
            FOURIER_DICE_COEFF_KEY: (these_dim, this_array + 0.),
            FOURIER_CSI_KEY: (these_dim, this_array + 0.)
        }
        main_data_dict.update(new_dict)

    advanced_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    a = advanced_score_table_xarray
    b = basic_score_table_xarray

    if num_neigh_distances > 0:
        numerators = numpy.sum(b[NEIGH_BRIER_SSE_KEY].values, axis=0)
        denominators = numpy.sum(b[NEIGH_BRIER_NUM_VALS_KEY].values, axis=0)
        a[NEIGH_BRIER_SCORE_KEY].values = numerators / denominators

        numerators = numpy.sum(b[NEIGH_FSS_ACTUAL_SSE_KEY].values, axis=0)
        denominators = numpy.sum(b[NEIGH_FSS_REFERENCE_SSE_KEY].values, axis=0)
        a[NEIGH_FSS_KEY].values = 1. - numerators / denominators

        numerators = numpy.sum(b[NEIGH_IOU_INTERSECTION_KEY].values, axis=0)
        denominators = numpy.sum(b[NEIGH_IOU_UNION_KEY].values, axis=0)
        a[NEIGH_IOU_KEY].values = numerators / denominators

        numerators = numpy.sum(b[NEIGH_DICE_INTERSECTION_KEY].values, axis=0)
        denominators = numpy.sum(b[NEIGH_DICE_NUM_PIX_KEY].values, axis=0)
        a[NEIGH_DICE_COEFF_KEY].values = numerators / denominators

        a_values = numpy.sum(b[NEIGH_OBS_ORIENTED_TP_KEY].values, axis=0)
        c_values = numpy.sum(b[NEIGH_FALSE_NEGATIVES_KEY].values, axis=0)
        pod_values = a_values / (a_values + c_values)

        a_values = numpy.sum(b[NEIGH_PRED_ORIENTED_TP_KEY].values, axis=0)
        b_values = numpy.sum(b[NEIGH_FALSE_POSITIVES_KEY].values, axis=0)
        success_ratios = a_values / (a_values + b_values)

        a[NEIGH_CSI_KEY].values = numpy.power(
            pod_values ** -1 + success_ratios ** -1 - 1,
            -1.
        )

    if num_fourier_bands > 0:
        numerators = numpy.sum(b[FREQ_SSE_REAL_KEY].values, axis=0)
        denominators = numpy.sum(b[FREQ_SSE_NUM_WEIGHTS_KEY].values, axis=0)
        a[FREQ_MSE_REAL_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FREQ_SSE_IMAGINARY_KEY].values, axis=0)
        a[FREQ_MSE_IMAGINARY_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FREQ_SSE_TOTAL_KEY].values, axis=0)
        a[FREQ_MSE_TOTAL_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_BRIER_SSE_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_BRIER_NUM_VALS_KEY].values, axis=0)
        a[FOURIER_BRIER_SCORE_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_FSS_ACTUAL_SSE_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_FSS_REFERENCE_SSE_KEY].values, axis=0)
        a[FOURIER_FSS_KEY].values = 1. - numerators / denominators

        numerators = numpy.sum(b[FOURIER_IOU_INTERSECTION_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_IOU_UNION_KEY].values, axis=0)
        a[FOURIER_IOU_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_DICE_INTERSECTION_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_DICE_NUM_PIX_KEY].values, axis=0)
        a[FOURIER_DICE_COEFF_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_CSI_NUMERATOR_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_CSI_DENOMINATOR_KEY].values, axis=0)
        a[FOURIER_CSI_KEY].values = numerators / denominators

    advanced_score_table_xarray = a
    return advanced_score_table_xarray


def write_scores(score_table_xarray, netcdf_file_name):
    """Writes basic or advanced scores to NetCDF file.

    :param score_table_xarray: xarray table created by `get_basic_scores` or
        `get_advanced_scores`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    score_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_scores(netcdf_file_name):
    """Reads basic or advanced scores from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: score_table_xarray: xarray table created by `get_basic_scores` or
        `get_advanced_scores`.
    """

    error_checking.assert_file_exists(netcdf_file_name)

    return xarray.open_dataset(netcdf_file_name)
