"""Learning curves (simple model evaluation)."""

import os
import copy
import numpy
import xarray
import tensorflow
from keras import backend as K
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4convection.io import prediction_io
from ml4convection.utils import general_utils
from ml4convection.utils import fourier_utils
from ml4convection.utils import wavelet_utils
from ml4convection.machine_learning import neural_net
from wavetf import WaveTFFactory

# TODO(thunderhoser): This module contains some duplicated code from
# evaluation.py.

# EPSILON = numpy.finfo(float).eps
GRID_SPACING_DEG = 0.0125
DATE_FORMAT = '%Y%m%d'

TIME_DIM = 'valid_time_unix_sec'
MIN_RESOLUTION_DIM = 'min_resolution_deg'
MAX_RESOLUTION_DIM = 'max_resolution_deg'
NEIGH_DISTANCE_DIM = 'neigh_distance_px'

MODEL_FILE_KEY = 'model_file_name'

NEIGH_BRIER_SSE_KEY = 'neigh_brier_sse'
NEIGH_BRIER_NUM_VALS_KEY = 'neigh_brier_num_values'
NEIGH_FSS_ACTUAL_SSE_KEY = 'neigh_fss_actual_sse'
NEIGH_FSS_REFERENCE_SSE_KEY = 'neigh_fss_reference_sse'
NEIGH_IOU_POS_ISCTN_KEY = 'neigh_iou_positive_intersection'
NEIGH_IOU_POS_UNION_KEY = 'neigh_iou_positive_union'
NEIGH_IOU_NEG_ISCTN_KEY = 'neigh_iou_negative_intersection'
NEIGH_IOU_NEG_UNION_KEY = 'neigh_iou_negative_union'
NEIGH_DICE_INTERSECTION_KEY = 'neigh_dice_intersection'
NEIGH_DICE_NUM_PIX_KEY = 'neigh_dice_num_pixels'
NEIGH_PRED_ORIENTED_TP_KEY = 'neigh_pred_oriented_true_positives'
NEIGH_OBS_ORIENTED_TP_KEY = 'neigh_obs_oriented_true_positives'
NEIGH_FALSE_POSITIVES_KEY = 'neigh_false_positives'
NEIGH_FALSE_NEGATIVES_KEY = 'neigh_false_negatives'

FOURIER_COEFF_SSE_REAL_KEY = 'fourier_coeff_sse_real'
FOURIER_COEFF_SSE_IMAGINARY_KEY = 'fourier_coeff_sse_imaginary'
FOURIER_COEFF_SSE_TOTAL_KEY = 'fourier_coeff_sse_total'
FOURIER_COEFF_NUM_WEIGHTS_KEY = 'fourier_coeff_num_weights'

FOURIER_BRIER_SSE_KEY = 'fourier_brier_sse'
FOURIER_BRIER_NUM_VALS_KEY = 'fourier_brier_num_values'
FOURIER_FSS_ACTUAL_SSE_KEY = 'fourier_fss_actual_sse'
FOURIER_FSS_REFERENCE_SSE_KEY = 'fourier_fss_reference_sse'
FOURIER_IOU_POS_ISCTN_KEY = 'fourier_iou_positive_intersection'
FOURIER_IOU_POS_UNION_KEY = 'fourier_iou_positive_union'
FOURIER_IOU_NEG_ISCTN_KEY = 'fourier_iou_negative_intersection'
FOURIER_IOU_NEG_UNION_KEY = 'fourier_iou_negative_union'
FOURIER_DICE_INTERSECTION_KEY = 'fourier_dice_intersection'
FOURIER_DICE_NUM_PIX_KEY = 'fourier_dice_num_pixels'
FOURIER_TRUE_POSITIVES_KEY = 'fourier_num_true_positives'
FOURIER_FALSE_POSITIVES_KEY = 'fourier_num_false_positives'
FOURIER_FALSE_NEGATIVES_KEY = 'fourier_num_false_negatives'
FOURIER_TRUE_NEGATIVES_KEY = 'fourier_num_true_negatives'

WAVELET_SSE_MEAN_COEFFS_KEY = 'wavelet_sse_mean_coeffs'
WAVELET_SSE_DETAIL_COEFFS_KEY = 'wavelet_sse_detail_coeffs'
WAVELET_NUM_MEAN_COEFFS_KEY = 'wavelet_num_mean_coeffs'
WAVELET_NUM_DETAIL_COEFFS_KEY = 'wavelet_num_detail_coeffs'

WAVELET_BRIER_SSE_KEY = 'wavelet_brier_sse'
WAVELET_BRIER_NUM_VALS_KEY = 'wavelet_brier_num_values'
WAVELET_FSS_ACTUAL_SSE_KEY = 'wavelet_fss_actual_sse'
WAVELET_FSS_REFERENCE_SSE_KEY = 'wavelet_fss_reference_sse'
WAVELET_IOU_POS_ISCTN_KEY = 'wavelet_iou_positive_intersection'
WAVELET_IOU_POS_UNION_KEY = 'wavelet_iou_positive_union'
WAVELET_IOU_NEG_ISCTN_KEY = 'wavelet_iou_negative_intersection'
WAVELET_IOU_NEG_UNION_KEY = 'wavelet_iou_negative_union'
WAVELET_DICE_INTERSECTION_KEY = 'wavelet_dice_intersection'
WAVELET_DICE_NUM_PIX_KEY = 'wavelet_dice_num_pixels'
WAVELET_TRUE_POSITIVES_KEY = 'wavelet_num_true_positives'
WAVELET_FALSE_POSITIVES_KEY = 'wavelet_num_false_positives'
WAVELET_FALSE_NEGATIVES_KEY = 'wavelet_num_false_negatives'
WAVELET_TRUE_NEGATIVES_KEY = 'wavelet_num_true_negatives'

NEIGH_BRIER_SCORE_KEY = 'neigh_brier_score'
NEIGH_FSS_KEY = 'neigh_fss'
NEIGH_IOU_KEY = 'neigh_iou'
NEIGH_ALL_CLASS_IOU_KEY = 'neigh_all_class_iou'
NEIGH_DICE_COEFF_KEY = 'neigh_dice_coeff'
NEIGH_CSI_KEY = 'neigh_csi'

FOURIER_COEFF_MSE_REAL_KEY = 'fourier_coeff_mse_real'
FOURIER_COEFF_MSE_IMAGINARY_KEY = 'fourier_coeff_mse_imaginary'
FOURIER_COEFF_MSE_TOTAL_KEY = 'fourier_coeff_mse_total'
FOURIER_BRIER_SCORE_KEY = 'fourier_brier_score'
FOURIER_FSS_KEY = 'fourier_fss'
FOURIER_IOU_KEY = 'fourier_iou'
FOURIER_ALL_CLASS_IOU_KEY = 'fourier_all_class_iou'
FOURIER_DICE_COEFF_KEY = 'fourier_dice_coeff'
FOURIER_CSI_KEY = 'fourier_csi'
FOURIER_PEIRCE_SCORE_KEY = 'fourier_peirce_score'
FOURIER_HEIDKE_SCORE_KEY = 'fourier_heidke_score'
FOURIER_GERRITY_SCORE_KEY = 'fourier_gerrity_score'

WAVELET_COEFF_MSE_MEAN_KEY = 'wavelet_coeff_mse_real'
WAVELET_COEFF_MSE_DETAIL_KEY = 'wavelet_coeff_mse_imaginary'
WAVELET_BRIER_SCORE_KEY = 'wavelet_brier_score'
WAVELET_FSS_KEY = 'wavelet_fss'
WAVELET_IOU_KEY = 'wavelet_iou'
WAVELET_ALL_CLASS_IOU_KEY = 'wavelet_all_class_iou'
WAVELET_DICE_COEFF_KEY = 'wavelet_dice_coeff'
WAVELET_CSI_KEY = 'wavelet_csi'
WAVELET_PEIRCE_SCORE_KEY = 'wavelet_peirce_score'
WAVELET_HEIDKE_SCORE_KEY = 'wavelet_heidke_score'
WAVELET_GERRITY_SCORE_KEY = 'wavelet_gerrity_score'


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


def _apply_fourier_filter(orig_data_matrix, min_resolution_deg,
                          max_resolution_deg):
    """Filters spatial data via Fourier decomposition.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of rows after tapering = number of columns after tapering

    :param orig_data_matrix: E-by-M-by-N numpy array of spatial data.
    :param min_resolution_deg: Minimum resolution (degrees) allowed through
        band-pass filter.
    :param max_resolution_deg: Max resolution (degrees) allowed through
        band-pass filter.
    :return: filtered_data_matrix: E-by-M-by-N numpy array of filtered spatial
        data.
    :return: filtered_coeff_matrix: E-by-T-by-T numpy array of filtered Fourier
        coefficients.
    """

    orig_data_matrix = orig_data_matrix.astype(float)
    num_examples = orig_data_matrix.shape[0]

    orig_data_matrix = numpy.stack([
        fourier_utils.taper_spatial_data(orig_data_matrix[i, ...])
        for i in range(num_examples)
    ], axis=0)

    blackman_matrix = fourier_utils.apply_blackman_window(
        numpy.ones(orig_data_matrix.shape[1:])
    )
    orig_data_matrix = numpy.stack([
        orig_data_matrix[i, ...] * blackman_matrix for i in range(num_examples)
    ], axis=0)

    orig_data_tensor = tensorflow.constant(
        orig_data_matrix, dtype=tensorflow.complex128
    )
    coeff_tensor = tensorflow.signal.fft2d(orig_data_tensor)
    coeff_matrix = K.eval(coeff_tensor)

    butterworth_matrix = fourier_utils.apply_butterworth_filter(
        coefficient_matrix=numpy.ones(coeff_matrix.shape[1:]),
        filter_order=2, grid_spacing_metres=GRID_SPACING_DEG,
        min_resolution_metres=min_resolution_deg,
        max_resolution_metres=max_resolution_deg
    )

    coeff_matrix = numpy.stack([
        coeff_matrix[i, ...] * butterworth_matrix
        for i in range(num_examples)
    ], axis=0)

    coeff_tensor = tensorflow.constant(
        coeff_matrix, dtype=tensorflow.complex128
    )
    filtered_data_tensor = tensorflow.signal.ifft2d(coeff_tensor)
    filtered_data_tensor = tensorflow.math.real(filtered_data_tensor)
    filtered_data_matrix = K.eval(filtered_data_tensor)

    filtered_data_matrix = numpy.stack([
        fourier_utils.untaper_spatial_data(filtered_data_matrix[i, ...])
        for i in range(num_examples)
    ], axis=0)

    filtered_data_matrix = numpy.maximum(filtered_data_matrix, 0.)
    filtered_data_matrix = numpy.minimum(filtered_data_matrix, 1.)

    return filtered_data_matrix, coeff_matrix


def _apply_wavelet_filter(orig_data_matrix, min_resolution_deg,
                          max_resolution_deg):
    """Filters spatial data via wavelet decomposition.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    T = number of rows after tapering = number of columns after tapering

    :param orig_data_matrix: E-by-M-by-N numpy array of spatial data.
    :param min_resolution_deg: Minimum resolution (degrees) allowed through
        band-pass filter.
    :param max_resolution_deg: Max resolution (degrees) allowed through
        band-pass filter.
    :return: filtered_data_matrix: E-by-M-by-N numpy array of filtered spatial
        data.
    :return: filtered_mean_coeff_matrix: E-by-T-by-T numpy array of filtered
        wavelet coefficients for mean features.
    :return: filtered_detail_coeff_matrix: E-by-T-by-T-by-3 numpy array of
        filtered wavelet coefficients for detail features.
    """

    orig_data_matrix = orig_data_matrix.astype(float)
    orig_data_matrix, padding_arg = wavelet_utils.taper_spatial_data(
        orig_data_matrix
    )

    coeff_tensor_by_level = wavelet_utils.do_forward_transform(
        orig_data_matrix
    )
    coeff_tensor_by_level = wavelet_utils.filter_coefficients(
        coeff_tensor_by_level=coeff_tensor_by_level,
        grid_spacing_metres=GRID_SPACING_DEG,
        min_resolution_metres=min_resolution_deg,
        max_resolution_metres=max_resolution_deg, verbose=True
    )

    (
        filtered_mean_coeff_matrix,
        horizontal_coeff_matrix,
        vertical_coeff_matrix,
        diagonal_coeff_matrix
    ) = wavelet_utils.coeff_tensors_to_numpy(coeff_tensor_by_level)

    filtered_detail_coeff_matrix = numpy.stack(
        (horizontal_coeff_matrix, vertical_coeff_matrix, diagonal_coeff_matrix),
        axis=-1
    )

    inverse_dwt_object = WaveTFFactory().build('haar', dim=2, inverse=True)
    filtered_data_tensor = inverse_dwt_object.call(coeff_tensor_by_level[0])
    filtered_data_matrix = K.eval(filtered_data_tensor)[..., 0]

    filtered_data_matrix = wavelet_utils.untaper_spatial_data(
        spatial_data_matrix=filtered_data_matrix, numpy_pad_width=padding_arg
    )
    filtered_data_matrix = numpy.maximum(filtered_data_matrix, 0.)
    filtered_data_matrix = numpy.minimum(filtered_data_matrix, 1.)

    return (
        filtered_data_matrix, filtered_mean_coeff_matrix,
        filtered_detail_coeff_matrix
    )


def _get_fourier_coeff_sse_one_time(
        actual_weight_matrix, predicted_weight_matrix):
    """Computes SSE for Fourier coefficients at one time step.

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


def _get_wavelet_coeff_sse_one_time(
        actual_mean_coeff_matrix, actual_detail_coeff_matrix,
        predicted_mean_coeff_matrix, predicted_detail_coeff_matrix):
    """Computes SSE for Fourier coefficients at one time step.

    M = number of rows in grid
    N = number of columns in grid

    :param actual_mean_coeff_matrix: M-by-N numpy array of actual coefficients.
    :param actual_detail_coeff_matrix: M-by-N-by-3 numpy array of actual
        coefficients.
    :param predicted_mean_coeff_matrix: M-by-N numpy array of predicted
        coefficients.
    :param predicted_detail_coeff_matrix: M-by-N-by-3 numpy array of predicted
        coefficients.
    :return: mean_sse: Sum of squared errors (SSE) for mean coefficients.
    :return: detail_sse: SSE for detail coefficients.
    :return: num_mean_coeffs: Number of mean coefficients used in calculation.
    :return: num_detail_coeffs: Number of detail coefficients used in
        calculation.
    """

    this_error_matrix = (
        (actual_mean_coeff_matrix - predicted_mean_coeff_matrix) ** 2
    )
    mean_sse = numpy.nansum(this_error_matrix)
    num_mean_coeffs = numpy.sum(numpy.invert(numpy.isnan(this_error_matrix)))

    this_error_matrix = (
        (actual_detail_coeff_matrix - predicted_detail_coeff_matrix) ** 2
    )
    detail_sse = numpy.nansum(this_error_matrix)
    num_detail_coeffs = numpy.sum(numpy.invert(numpy.isnan(this_error_matrix)))

    return mean_sse, detail_sse, num_mean_coeffs, num_detail_coeffs


def _get_brier_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix,
        matching_distance_px):
    """Computes Brier-score components for one time step.

    M = number of rows in grid
    N = number of columns in grid

    If doing neighbourhood-based evaluation, this method assumes that
    `eval_mask_matrix` is already eroded for the given matching distance.

    If doing evaluation with band-pass filter (Fourier or wavelet), this method
    assumes that `actual_target_matrix` and `probability_matrix` have already
    gone through decomposition and recomposition.

    :param actual_target_matrix: M-by-N numpy array (see doc for
        `general_utils.check_2d_binary_matrix`), indicating where actual
        convection occurs.
    :param probability_matrix: M-by-N numpy array of forecast convection
        probabilities.
    :param eval_mask_matrix: M-by-N numpy array (see doc for
        `general_utils.check_2d_binary_matrix`), indicating which pixels are to
        be used for evaluation.
    :param matching_distance_px: Matching distance (pixels) for
        neighbourhood-based evaluation.  If doing evaluation with band-pass
        filter, make this None.
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

    See notes on neighbourhood vs. band-pass filter in
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

    See notes on neighbourhood vs. band-pass filter in
    `_get_brier_components_one_time`.

    :param actual_target_matrix: See doc for `_get_brier_components_one_time`.
    :param probability_matrix: Same.
    :param eval_mask_matrix: Same.
    :param matching_distance_px: Same.
    :return: positive_intersection: Intersection (numerator of IOU) for positive
        class.
    :return: positive_union: Union (denominator of IOU) for positive class.
    :return: negative_intersection: Intersection (numerator of IOU) for negative
        class.
    :return: negative_union: Union (denominator of IOU) for negative class.
    """

    if matching_distance_px is None:
        this_actual_target_matrix = actual_target_matrix
    else:
        this_actual_target_matrix = _dilate_binary_matrix(
            binary_matrix=actual_target_matrix,
            half_width_px=matching_distance_px
        ).astype(int)

    this_product_matrix = probability_matrix * this_actual_target_matrix
    this_product_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    positive_intersection = numpy.nansum(this_product_matrix)

    this_max_matrix = numpy.maximum(
        probability_matrix, this_actual_target_matrix
    )
    this_max_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    positive_union = numpy.nansum(this_max_matrix)

    this_product_matrix = (
        (1 - probability_matrix) * (1 - this_actual_target_matrix)
    )
    this_product_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    negative_intersection = numpy.nansum(this_product_matrix)

    this_max_matrix = numpy.maximum(
        1 - probability_matrix, 1 - this_actual_target_matrix
    )
    this_max_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    negative_union = numpy.nansum(this_max_matrix)

    return (
        positive_intersection, positive_union,
        negative_intersection, negative_union
    )


def _get_dice_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix,
        matching_distance_px):
    """Computes components of Dice coefficient for one time step.

    See notes on neighbourhood vs. band-pass filter in
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


def _get_neigh_csi_components_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix,
        matching_distance_px):
    """Computes components of neighbourhood-based CSI for one time step.

    See notes on neighbourhood vs. band-pass filter in
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
        input_matrix=actual_target_matrix.astype(float),
        half_width_px=matching_distance_px
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


def _get_band_pass_contingency_one_time(
        actual_target_matrix, probability_matrix, eval_mask_matrix):
    """Computes contingency table on band-pass-filtered data for one time step.

    See notes on neighbourhood vs. band-pass filter in
    `_get_brier_components_one_time`.

    :param actual_target_matrix: See doc for `_get_brier_components_one_time`.
    :param probability_matrix: Same.
    :param eval_mask_matrix: Same.
    :return: num_true_positives: a in contingency-table speak.
    :return: num_false_positives: b in contingency-table speak.
    :return: num_false_negatives: c in contingency-table speak.
    :return: num_true_negatives: d in contingency-table speak.
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

    this_matrix = (1 - probability_matrix) * (1 - actual_target_matrix)
    this_matrix[eval_mask_matrix.astype(bool) == False] = numpy.nan
    num_true_negatives = numpy.nansum(this_matrix)

    return (
        num_true_positives, num_false_positives,
        num_false_negatives, num_true_negatives
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
        prediction_dict, neigh_distances_px, min_resolutions_deg,
        max_resolutions_deg, test_mode=False, eval_mask_matrix=None,
        model_file_name=None):
    """Computes basic scores.

    D = number of matching distances for neighbourhood-based evaluation
    R = number of resolution bands for evaluation with band-pass filter

    :param prediction_dict: Dictionary with predicted and actual values (in
        format returned by `prediction_io.read_file`).
    :param neigh_distances_px: length-D numpy array of matching distances.  If
        you do not want neighbourhood-based evaluation, make this None.
    :param min_resolutions_deg: length-R numpy array of minimum
        resolutions (degrees).  If you do not want evaluation with band-pass
        filter, make this None.
    :param max_resolutions_deg: Same but with max resolutions.
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
            min_resolutions_deg is None
            or max_resolutions_deg is None
    ):
        num_filter_bands = 0
    else:
        error_checking.assert_is_numpy_array(
            min_resolutions_deg, num_dimensions=1
        )

        num_filter_bands = len(min_resolutions_deg)
        expected_dim = numpy.array([num_filter_bands], dtype=int)
        error_checking.assert_is_numpy_array(
            max_resolutions_deg, exact_dimensions=expected_dim
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
            NEIGH_IOU_POS_ISCTN_KEY: (these_dim, this_array + 0.),
            NEIGH_IOU_POS_UNION_KEY: (these_dim, this_array + 0.),
            NEIGH_IOU_NEG_ISCTN_KEY: (these_dim, this_array + 0.),
            NEIGH_IOU_NEG_UNION_KEY: (these_dim, this_array + 0.),
            NEIGH_DICE_INTERSECTION_KEY: (these_dim, this_array + 0.),
            NEIGH_DICE_NUM_PIX_KEY: (these_dim, this_array + 0.),
            NEIGH_PRED_ORIENTED_TP_KEY: (these_dim, this_array + 0.),
            NEIGH_OBS_ORIENTED_TP_KEY: (these_dim, this_array + 0.),
            NEIGH_FALSE_POSITIVES_KEY: (these_dim, this_array + 0.),
            NEIGH_FALSE_NEGATIVES_KEY: (these_dim, this_array + 0.)
        }
        main_data_dict.update(new_dict)

    if num_filter_bands > 0:
        metadata_dict[MIN_RESOLUTION_DIM] = min_resolutions_deg
        metadata_dict[MAX_RESOLUTION_DIM] = max_resolutions_deg

        these_dim = (TIME_DIM, MIN_RESOLUTION_DIM)
        this_array = numpy.full((num_times, num_filter_bands), numpy.nan)

        new_dict = {
            FOURIER_COEFF_SSE_REAL_KEY: (these_dim, this_array + 0),
            FOURIER_COEFF_SSE_IMAGINARY_KEY: (these_dim, this_array + 0),
            FOURIER_COEFF_SSE_TOTAL_KEY: (these_dim, this_array + 0),
            FOURIER_COEFF_NUM_WEIGHTS_KEY: (these_dim, this_array + 0),
            FOURIER_BRIER_SSE_KEY: (these_dim, this_array + 0),
            FOURIER_BRIER_NUM_VALS_KEY: (these_dim, this_array + 0.),
            FOURIER_FSS_ACTUAL_SSE_KEY: (these_dim, this_array + 0.),
            FOURIER_FSS_REFERENCE_SSE_KEY: (these_dim, this_array + 0.),
            FOURIER_IOU_POS_ISCTN_KEY: (these_dim, this_array + 0.),
            FOURIER_IOU_POS_UNION_KEY: (these_dim, this_array + 0.),
            FOURIER_IOU_NEG_ISCTN_KEY: (these_dim, this_array + 0.),
            FOURIER_IOU_NEG_UNION_KEY: (these_dim, this_array + 0.),
            FOURIER_DICE_INTERSECTION_KEY: (these_dim, this_array + 0.),
            FOURIER_DICE_NUM_PIX_KEY: (these_dim, this_array + 0.),
            FOURIER_TRUE_POSITIVES_KEY: (these_dim, this_array + 0.),
            FOURIER_FALSE_POSITIVES_KEY: (these_dim, this_array + 0.),
            FOURIER_FALSE_NEGATIVES_KEY: (these_dim, this_array + 0.),
            FOURIER_TRUE_NEGATIVES_KEY: (these_dim, this_array + 0.),
            WAVELET_SSE_MEAN_COEFFS_KEY: (these_dim, this_array + 0),
            WAVELET_SSE_DETAIL_COEFFS_KEY: (these_dim, this_array + 0),
            WAVELET_NUM_MEAN_COEFFS_KEY: (these_dim, this_array + 0),
            WAVELET_NUM_DETAIL_COEFFS_KEY: (these_dim, this_array + 0),
            WAVELET_BRIER_SSE_KEY: (these_dim, this_array + 0),
            WAVELET_BRIER_NUM_VALS_KEY: (these_dim, this_array + 0.),
            WAVELET_FSS_ACTUAL_SSE_KEY: (these_dim, this_array + 0.),
            WAVELET_FSS_REFERENCE_SSE_KEY: (these_dim, this_array + 0.),
            WAVELET_IOU_POS_ISCTN_KEY: (these_dim, this_array + 0.),
            WAVELET_IOU_POS_UNION_KEY: (these_dim, this_array + 0.),
            WAVELET_IOU_NEG_ISCTN_KEY: (these_dim, this_array + 0.),
            WAVELET_IOU_NEG_UNION_KEY: (these_dim, this_array + 0.),
            WAVELET_DICE_INTERSECTION_KEY: (these_dim, this_array + 0.),
            WAVELET_DICE_NUM_PIX_KEY: (these_dim, this_array + 0.),
            WAVELET_TRUE_POSITIVES_KEY: (these_dim, this_array + 0.),
            WAVELET_FALSE_POSITIVES_KEY: (these_dim, this_array + 0.),
            WAVELET_FALSE_NEGATIVES_KEY: (these_dim, this_array + 0.),
            WAVELET_TRUE_NEGATIVES_KEY: (these_dim, this_array + 0.)
        }
        main_data_dict.update(new_dict)

    basic_score_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    basic_score_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    b = basic_score_table_xarray

    mean_prob_matrix = prediction_io.get_mean_predictions(prediction_dict)

    if num_neigh_distances > 0:
        num_grid_rows = mean_prob_matrix.shape[1]
        num_grid_columns = mean_prob_matrix.shape[2]
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
            if numpy.mod(i, 10) == 0:
                print((
                    'Computing neighbourhood-based scores for {0:d}th of {1:d} '
                    'time steps...'
                ).format(
                    i + 1, num_times
                ))

            this_prob_matrix = mean_prob_matrix[i, ...] + 0.
            this_target_matrix = (
                prediction_dict[prediction_io.TARGET_MATRIX_KEY][i, ...] + 0
            )

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
                    b[NEIGH_IOU_POS_ISCTN_KEY].values[i, k],
                    b[NEIGH_IOU_POS_UNION_KEY].values[i, k],
                    b[NEIGH_IOU_NEG_ISCTN_KEY].values[i, k],
                    b[NEIGH_IOU_NEG_UNION_KEY].values[i, k]
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

        print((
            'Have computed neighbourhood-based scores for all {0:d} time steps!'
        ).format(
            num_times
        ))

    if num_filter_bands > 0:
        these_dim = (
            (num_filter_bands,) +
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape[:-1]
        )
        fourier_forecast_matrix = numpy.full(these_dim, numpy.nan)
        fourier_target_matrix = numpy.full(these_dim, numpy.nan)
        wavelet_forecast_matrix = numpy.full(these_dim, numpy.nan)
        wavelet_target_matrix = numpy.full(these_dim, numpy.nan)

        this_matrix = fourier_utils.taper_spatial_data(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ..., 0]
        )
        these_dim = (num_filter_bands, num_times) + this_matrix.shape
        fourier_forecast_coeff_matrix = numpy.full(these_dim, numpy.nan)
        fourier_target_coeff_matrix = numpy.full(these_dim, numpy.nan)

        this_matrix = wavelet_utils.taper_spatial_data(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][[0], ..., 0]
        )[0]
        these_dim = (num_filter_bands, num_times) + this_matrix.shape[1:]
        wavelet_forecast_mean_coeff_matrix = numpy.full(these_dim, numpy.nan)
        wavelet_target_mean_coeff_matrix = numpy.full(these_dim, numpy.nan)

        these_dim = these_dim + (3,)
        wavelet_forecast_detail_coeff_matrix = numpy.full(these_dim, numpy.nan)
        wavelet_target_detail_coeff_matrix = numpy.full(these_dim, numpy.nan)

        for k in range(num_filter_bands):
            (
                fourier_forecast_matrix[k, ...],
                fourier_forecast_coeff_matrix[k, ...]
            ) = _apply_fourier_filter(
                orig_data_matrix=mean_prob_matrix + 0.,
                min_resolution_deg=min_resolutions_deg[k],
                max_resolution_deg=max_resolutions_deg[k]
            )

            (
                fourier_target_matrix[k, ...],
                fourier_target_coeff_matrix[k, ...]
            ) = _apply_fourier_filter(
                orig_data_matrix=
                prediction_dict[prediction_io.TARGET_MATRIX_KEY] + 0.,
                min_resolution_deg=min_resolutions_deg[k],
                max_resolution_deg=max_resolutions_deg[k]
            )

            (
                wavelet_forecast_matrix[k, ...],
                wavelet_forecast_mean_coeff_matrix[k, ...],
                wavelet_forecast_detail_coeff_matrix[k, ...]
            ) = _apply_wavelet_filter(
                orig_data_matrix=mean_prob_matrix + 0.,
                min_resolution_deg=min_resolutions_deg[k],
                max_resolution_deg=max_resolutions_deg[k]
            )

            (
                wavelet_target_matrix[k, ...],
                wavelet_target_mean_coeff_matrix[k, ...],
                wavelet_target_detail_coeff_matrix[k, ...]
            ) = _apply_wavelet_filter(
                orig_data_matrix=
                prediction_dict[prediction_io.TARGET_MATRIX_KEY] + 0.,
                min_resolution_deg=min_resolutions_deg[k],
                max_resolution_deg=max_resolutions_deg[k]
            )

        for i in range(num_times):
            if numpy.mod(i, 10) == 0:
                print((
                    'Computing scores with band-pass filter for '
                    '{0:d}th of {1:d} time steps...'
                ).format(
                    i + 1, num_times
                ))

            for k in range(num_filter_bands):
                (
                    b[FOURIER_COEFF_SSE_REAL_KEY].values[i, k],
                    b[FOURIER_COEFF_SSE_IMAGINARY_KEY].values[i, k],
                    b[FOURIER_COEFF_SSE_TOTAL_KEY].values[i, k],
                    b[FOURIER_COEFF_NUM_WEIGHTS_KEY].values[i, k]
                ) = _get_fourier_coeff_sse_one_time(
                    actual_weight_matrix=fourier_target_coeff_matrix[k, i, ...],
                    predicted_weight_matrix=
                    fourier_forecast_coeff_matrix[k, i, ...]
                )

                (
                    b[FOURIER_BRIER_SSE_KEY].values[i, k],
                    b[FOURIER_BRIER_NUM_VALS_KEY].values[i, k]
                ) = _get_brier_components_one_time(
                    actual_target_matrix=fourier_target_matrix[k, i, ...],
                    probability_matrix=fourier_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[FOURIER_FSS_ACTUAL_SSE_KEY].values[i, k],
                    b[FOURIER_FSS_REFERENCE_SSE_KEY].values[i, k]
                ) = _get_fss_components_one_time(
                    actual_target_matrix=fourier_target_matrix[k, i, ...],
                    probability_matrix=fourier_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[FOURIER_IOU_POS_ISCTN_KEY].values[i, k],
                    b[FOURIER_IOU_POS_UNION_KEY].values[i, k],
                    b[FOURIER_IOU_NEG_ISCTN_KEY].values[i, k],
                    b[FOURIER_IOU_NEG_UNION_KEY].values[i, k]
                ) = _get_iou_components_one_time(
                    actual_target_matrix=fourier_target_matrix[k, i, ...],
                    probability_matrix=fourier_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[FOURIER_DICE_INTERSECTION_KEY].values[i, k],
                    b[FOURIER_DICE_NUM_PIX_KEY].values[i, k]
                ) = _get_dice_components_one_time(
                    actual_target_matrix=fourier_target_matrix[k, i, ...],
                    probability_matrix=fourier_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[FOURIER_TRUE_POSITIVES_KEY].values[i, k],
                    b[FOURIER_FALSE_POSITIVES_KEY].values[i, k],
                    b[FOURIER_FALSE_NEGATIVES_KEY].values[i, k],
                    b[FOURIER_TRUE_NEGATIVES_KEY].values[i, k]
                ) = _get_band_pass_contingency_one_time(
                    actual_target_matrix=fourier_target_matrix[k, i, ...],
                    probability_matrix=fourier_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                )

                (
                    b[WAVELET_SSE_MEAN_COEFFS_KEY].values[i, k],
                    b[WAVELET_SSE_DETAIL_COEFFS_KEY].values[i, k],
                    b[WAVELET_NUM_MEAN_COEFFS_KEY].values[i, k],
                    b[WAVELET_NUM_DETAIL_COEFFS_KEY].values[i, k]
                ) = _get_wavelet_coeff_sse_one_time(
                    actual_mean_coeff_matrix=
                    wavelet_target_mean_coeff_matrix[k, i, ...],
                    actual_detail_coeff_matrix=
                    wavelet_target_detail_coeff_matrix[k, i, ...],
                    predicted_mean_coeff_matrix=
                    wavelet_forecast_mean_coeff_matrix[k, i, ...],
                    predicted_detail_coeff_matrix=
                    wavelet_forecast_detail_coeff_matrix[k, i, ...]
                )

                (
                    b[WAVELET_BRIER_SSE_KEY].values[i, k],
                    b[WAVELET_BRIER_NUM_VALS_KEY].values[i, k]
                ) = _get_brier_components_one_time(
                    actual_target_matrix=wavelet_target_matrix[k, i, ...],
                    probability_matrix=wavelet_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[WAVELET_FSS_ACTUAL_SSE_KEY].values[i, k],
                    b[WAVELET_FSS_REFERENCE_SSE_KEY].values[i, k]
                ) = _get_fss_components_one_time(
                    actual_target_matrix=wavelet_target_matrix[k, i, ...],
                    probability_matrix=wavelet_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[WAVELET_IOU_POS_ISCTN_KEY].values[i, k],
                    b[WAVELET_IOU_POS_UNION_KEY].values[i, k],
                    b[WAVELET_IOU_NEG_ISCTN_KEY].values[i, k],
                    b[WAVELET_IOU_NEG_UNION_KEY].values[i, k]
                ) = _get_iou_components_one_time(
                    actual_target_matrix=wavelet_target_matrix[k, i, ...],
                    probability_matrix=wavelet_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[WAVELET_DICE_INTERSECTION_KEY].values[i, k],
                    b[WAVELET_DICE_NUM_PIX_KEY].values[i, k]
                ) = _get_dice_components_one_time(
                    actual_target_matrix=wavelet_target_matrix[k, i, ...],
                    probability_matrix=wavelet_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                    matching_distance_px=None
                )

                (
                    b[WAVELET_TRUE_POSITIVES_KEY].values[i, k],
                    b[WAVELET_FALSE_POSITIVES_KEY].values[i, k],
                    b[WAVELET_FALSE_NEGATIVES_KEY].values[i, k],
                    b[WAVELET_TRUE_NEGATIVES_KEY].values[i, k]
                ) = _get_band_pass_contingency_one_time(
                    actual_target_matrix=wavelet_target_matrix[k, i, ...],
                    probability_matrix=wavelet_forecast_matrix[k, i, ...],
                    eval_mask_matrix=eval_mask_matrix,
                )

        print((
            'Have computed scores with band-pass filter for all {0:d} time '
            'steps!'
        ).format(
            num_times
        ))

    basic_score_table_xarray = b
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
    unique_model_file_names = numpy.unique(numpy.array(model_file_names))
    assert len(unique_model_file_names) == 1

    non_empty_tables = [
        b for b in basic_score_tables_xarray
        if len(b.coords[TIME_DIM].values) > 0
    ]

    return xarray.concat(objs=non_empty_tables, dim=TIME_DIM)


def _compute_peirce_scores(
        basic_score_table_xarray, advanced_score_table_xarray, use_wavelets):
    """Computes Peirce score for each band-pass filter.

    :param basic_score_table_xarray: xarray table created by `get_basic_scores`.
    :param advanced_score_table_xarray: xarray table with advanced scores
        (variable and dimension names should make the table self-explanatory).
    :param use_wavelets: Boolean flag.  If True (False), will use
        wavelet- (Fourier-)based band-pass filters.
    :return: advanced_score_table_xarray: Same as input but with updated Peirce
        scores.
    """

    b = basic_score_table_xarray
    a = advanced_score_table_xarray

    if use_wavelets:
        num_true_positives_key = WAVELET_TRUE_POSITIVES_KEY
        num_false_positives_key = WAVELET_FALSE_POSITIVES_KEY
        num_false_negatives_key = WAVELET_FALSE_NEGATIVES_KEY
        num_true_negatives_key = WAVELET_TRUE_NEGATIVES_KEY
        peirce_score_key = WAVELET_PEIRCE_SCORE_KEY
    else:
        num_true_positives_key = FOURIER_TRUE_POSITIVES_KEY
        num_false_positives_key = FOURIER_FALSE_POSITIVES_KEY
        num_false_negatives_key = FOURIER_FALSE_NEGATIVES_KEY
        num_true_negatives_key = FOURIER_TRUE_NEGATIVES_KEY
        peirce_score_key = FOURIER_PEIRCE_SCORE_KEY

    numerators = numpy.sum(b[num_true_positives_key].values, axis=0)
    denominators = numpy.sum(
        b[num_true_positives_key].values + b[num_false_negatives_key].values,
        axis=0
    )
    pod_values = numerators / denominators

    numerators = numpy.sum(b[num_false_positives_key].values, axis=0)
    denominators = numpy.sum(
        b[num_false_positives_key].values + b[num_true_negatives_key].values,
        axis=0
    )
    pofd_values = numerators / denominators

    a[peirce_score_key].values = pod_values - pofd_values
    return a


def _compute_heidke_scores(
        basic_score_table_xarray, advanced_score_table_xarray, use_wavelets):
    """Computes Heidke score for each band-pass filter.

    :param basic_score_table_xarray: See doc for `_compute_peirce_scores`.
    :param advanced_score_table_xarray: Same.
    :param use_wavelets: Same.
    :return: advanced_score_table_xarray: Same.
    """

    b = basic_score_table_xarray
    a = advanced_score_table_xarray

    if use_wavelets:
        num_true_positives_key = WAVELET_TRUE_POSITIVES_KEY
        num_false_positives_key = WAVELET_FALSE_POSITIVES_KEY
        num_false_negatives_key = WAVELET_FALSE_NEGATIVES_KEY
        num_true_negatives_key = WAVELET_TRUE_NEGATIVES_KEY
        heidke_score_key = WAVELET_HEIDKE_SCORE_KEY
    else:
        num_true_positives_key = FOURIER_TRUE_POSITIVES_KEY
        num_false_positives_key = FOURIER_FALSE_POSITIVES_KEY
        num_false_negatives_key = FOURIER_FALSE_NEGATIVES_KEY
        num_true_negatives_key = FOURIER_TRUE_NEGATIVES_KEY
        heidke_score_key = FOURIER_HEIDKE_SCORE_KEY

    true_positive_counts = numpy.sum(b[num_true_positives_key].values, axis=0)
    false_positive_counts = numpy.sum(b[num_false_positives_key].values, axis=0)
    false_negative_counts = numpy.sum(b[num_false_negatives_key].values, axis=0)
    true_negative_counts = numpy.sum(b[num_true_negatives_key].values, axis=0)

    random_correct_counts = (
        (true_positive_counts + false_positive_counts) *
        (true_positive_counts + false_negative_counts) +
        (false_negative_counts + true_negative_counts) *
        (false_positive_counts + true_negative_counts)
    )
    example_counts = (
        true_positive_counts + false_positive_counts +
        false_negative_counts + true_negative_counts
    )
    random_correct_counts = random_correct_counts / example_counts

    numerators = (
        true_positive_counts + true_negative_counts - random_correct_counts
    )
    denominators = example_counts - random_correct_counts

    a[heidke_score_key].values = numerators / denominators
    return a


def _compute_gerrity_scores(
        basic_score_table_xarray, advanced_score_table_xarray, use_wavelets):
    """Computes Gerrity score for each band-pass filter.

    :param basic_score_table_xarray: See doc for `_compute_peirce_scores`.
    :param advanced_score_table_xarray: Same.
    :param use_wavelets: Same.
    :return: advanced_score_table_xarray: Same.
    """

    b = basic_score_table_xarray
    a = advanced_score_table_xarray

    if use_wavelets:
        num_true_positives_key = WAVELET_TRUE_POSITIVES_KEY
        num_false_positives_key = WAVELET_FALSE_POSITIVES_KEY
        num_false_negatives_key = WAVELET_FALSE_NEGATIVES_KEY
        num_true_negatives_key = WAVELET_TRUE_NEGATIVES_KEY
        gerrity_score_key = WAVELET_GERRITY_SCORE_KEY
    else:
        num_true_positives_key = FOURIER_TRUE_POSITIVES_KEY
        num_false_positives_key = FOURIER_FALSE_POSITIVES_KEY
        num_false_negatives_key = FOURIER_FALSE_NEGATIVES_KEY
        num_true_negatives_key = FOURIER_TRUE_NEGATIVES_KEY
        gerrity_score_key = FOURIER_GERRITY_SCORE_KEY

    true_positive_counts = numpy.sum(b[num_true_positives_key].values, axis=0)
    false_positive_counts = numpy.sum(b[num_false_positives_key].values, axis=0)
    false_negative_counts = numpy.sum(b[num_false_negatives_key].values, axis=0)
    true_negative_counts = numpy.sum(b[num_true_negatives_key].values, axis=0)

    example_counts = (
        true_positive_counts + false_positive_counts +
        false_negative_counts + true_negative_counts
    )
    event_ratios = (
        (false_positive_counts + true_negative_counts) /
        (true_positive_counts + false_negative_counts)
    )
    numerators = (
        true_positive_counts * event_ratios
        + true_negative_counts * (1. / event_ratios)
        - false_positive_counts - false_negative_counts
    )

    a[gerrity_score_key].values = numerators / example_counts
    return a


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
        num_filter_bands = len(metadata_dict[MIN_RESOLUTION_DIM])
    else:
        num_filter_bands = 0

    main_data_dict = dict()

    if num_neigh_distances > 0:
        these_dim = (NEIGH_DISTANCE_DIM,)
        this_array = numpy.full(num_neigh_distances, numpy.nan)

        new_dict = {
            NEIGH_BRIER_SCORE_KEY: (these_dim, this_array + 0.),
            NEIGH_FSS_KEY: (these_dim, this_array + 0.),
            NEIGH_IOU_KEY: (these_dim, this_array + 0.),
            NEIGH_ALL_CLASS_IOU_KEY: (these_dim, this_array + 0.),
            NEIGH_DICE_COEFF_KEY: (these_dim, this_array + 0.),
            NEIGH_CSI_KEY: (these_dim, this_array + 0.)
        }
        main_data_dict.update(new_dict)

    if num_filter_bands > 0:
        these_dim = (MIN_RESOLUTION_DIM,)
        this_array = numpy.full(num_filter_bands, numpy.nan)

        new_dict = {
            FOURIER_COEFF_MSE_REAL_KEY: (these_dim, this_array + 0.),
            FOURIER_COEFF_MSE_IMAGINARY_KEY: (these_dim, this_array + 0.),
            FOURIER_COEFF_MSE_TOTAL_KEY: (these_dim, this_array + 0.),
            FOURIER_BRIER_SCORE_KEY: (these_dim, this_array + 0.),
            FOURIER_FSS_KEY: (these_dim, this_array + 0.),
            FOURIER_IOU_KEY: (these_dim, this_array + 0.),
            FOURIER_ALL_CLASS_IOU_KEY: (these_dim, this_array + 0.),
            FOURIER_DICE_COEFF_KEY: (these_dim, this_array + 0.),
            FOURIER_CSI_KEY: (these_dim, this_array + 0.),
            FOURIER_PEIRCE_SCORE_KEY: (these_dim, this_array + 0.),
            FOURIER_GERRITY_SCORE_KEY: (these_dim, this_array + 0.),
            FOURIER_HEIDKE_SCORE_KEY: (these_dim, this_array + 0.),
            WAVELET_COEFF_MSE_MEAN_KEY: (these_dim, this_array + 0.),
            WAVELET_COEFF_MSE_DETAIL_KEY: (these_dim, this_array + 0.),
            WAVELET_BRIER_SCORE_KEY: (these_dim, this_array + 0.),
            WAVELET_FSS_KEY: (these_dim, this_array + 0.),
            WAVELET_IOU_KEY: (these_dim, this_array + 0.),
            WAVELET_ALL_CLASS_IOU_KEY: (these_dim, this_array + 0.),
            WAVELET_DICE_COEFF_KEY: (these_dim, this_array + 0.),
            WAVELET_CSI_KEY: (these_dim, this_array + 0.),
            WAVELET_PEIRCE_SCORE_KEY: (these_dim, this_array + 0.),
            WAVELET_GERRITY_SCORE_KEY: (these_dim, this_array + 0.),
            WAVELET_HEIDKE_SCORE_KEY: (these_dim, this_array + 0.)
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

        numerators = numpy.sum(b[NEIGH_IOU_POS_ISCTN_KEY].values, axis=0)
        denominators = numpy.sum(b[NEIGH_IOU_POS_UNION_KEY].values, axis=0)
        a[NEIGH_IOU_KEY].values = numerators / denominators

        numerators = numpy.sum(b[NEIGH_IOU_NEG_ISCTN_KEY].values, axis=0)
        denominators = numpy.sum(b[NEIGH_IOU_NEG_UNION_KEY].values, axis=0)
        a[NEIGH_ALL_CLASS_IOU_KEY].values = 0.5 * (
            numerators / denominators + a[NEIGH_IOU_KEY].values
        )

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

    if num_filter_bands > 0:
        numerators = numpy.sum(b[FOURIER_COEFF_SSE_REAL_KEY].values, axis=0)
        denominators = numpy.sum(
            b[FOURIER_COEFF_NUM_WEIGHTS_KEY].values, axis=0
        )
        a[FOURIER_COEFF_MSE_REAL_KEY].values = numerators / denominators

        numerators = numpy.sum(
            b[FOURIER_COEFF_SSE_IMAGINARY_KEY].values, axis=0
        )
        a[FOURIER_COEFF_MSE_IMAGINARY_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_COEFF_SSE_TOTAL_KEY].values, axis=0)
        a[FOURIER_COEFF_MSE_TOTAL_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_BRIER_SSE_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_BRIER_NUM_VALS_KEY].values, axis=0)
        a[FOURIER_BRIER_SCORE_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_FSS_ACTUAL_SSE_KEY].values, axis=0)
        denominators = numpy.sum(
            b[FOURIER_FSS_REFERENCE_SSE_KEY].values, axis=0
        )
        a[FOURIER_FSS_KEY].values = 1. - numerators / denominators

        numerators = numpy.sum(b[FOURIER_IOU_POS_ISCTN_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_IOU_POS_UNION_KEY].values, axis=0)
        a[FOURIER_IOU_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_IOU_NEG_ISCTN_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_IOU_NEG_UNION_KEY].values, axis=0)
        a[FOURIER_ALL_CLASS_IOU_KEY].values = 0.5 * (
            numerators / denominators + a[FOURIER_IOU_KEY].values
        )

        numerators = numpy.sum(b[FOURIER_DICE_INTERSECTION_KEY].values, axis=0)
        denominators = numpy.sum(b[FOURIER_DICE_NUM_PIX_KEY].values, axis=0)
        a[FOURIER_DICE_COEFF_KEY].values = numerators / denominators

        numerators = numpy.sum(b[FOURIER_TRUE_POSITIVES_KEY].values, axis=0)
        denominators = numpy.sum(
            b[FOURIER_TRUE_POSITIVES_KEY].values +
            b[FOURIER_FALSE_POSITIVES_KEY].values +
            b[FOURIER_FALSE_NEGATIVES_KEY].values,
            axis=0
        )
        a[FOURIER_CSI_KEY].values = numerators / denominators

        a = _compute_peirce_scores(
            basic_score_table_xarray=b, advanced_score_table_xarray=a,
            use_wavelets=False
        )
        a = _compute_heidke_scores(
            basic_score_table_xarray=b, advanced_score_table_xarray=a,
            use_wavelets=False
        )
        a = _compute_gerrity_scores(
            basic_score_table_xarray=b, advanced_score_table_xarray=a,
            use_wavelets=False
        )

        numerators = numpy.sum(b[WAVELET_SSE_MEAN_COEFFS_KEY].values, axis=0)
        denominators = numpy.sum(b[WAVELET_NUM_MEAN_COEFFS_KEY].values, axis=0)
        a[WAVELET_COEFF_MSE_MEAN_KEY].values = numerators / denominators

        numerators = numpy.sum(b[WAVELET_SSE_DETAIL_COEFFS_KEY].values, axis=0)
        denominators = numpy.sum(
            b[WAVELET_NUM_DETAIL_COEFFS_KEY].values, axis=0
        )
        a[WAVELET_COEFF_MSE_DETAIL_KEY].values = numerators / denominators

        numerators = numpy.sum(b[WAVELET_BRIER_SSE_KEY].values, axis=0)
        denominators = numpy.sum(b[WAVELET_BRIER_NUM_VALS_KEY].values, axis=0)
        a[WAVELET_BRIER_SCORE_KEY].values = numerators / denominators

        numerators = numpy.sum(b[WAVELET_FSS_ACTUAL_SSE_KEY].values, axis=0)
        denominators = numpy.sum(
            b[WAVELET_FSS_REFERENCE_SSE_KEY].values, axis=0
        )
        a[WAVELET_FSS_KEY].values = 1. - numerators / denominators

        numerators = numpy.sum(b[WAVELET_IOU_POS_ISCTN_KEY].values, axis=0)
        denominators = numpy.sum(b[WAVELET_IOU_POS_UNION_KEY].values, axis=0)
        a[WAVELET_IOU_KEY].values = numerators / denominators

        numerators = numpy.sum(b[WAVELET_IOU_NEG_ISCTN_KEY].values, axis=0)
        denominators = numpy.sum(b[WAVELET_IOU_NEG_UNION_KEY].values, axis=0)
        a[WAVELET_ALL_CLASS_IOU_KEY].values = 0.5 * (
            numerators / denominators + a[WAVELET_IOU_KEY].values
        )

        numerators = numpy.sum(b[WAVELET_DICE_INTERSECTION_KEY].values, axis=0)
        denominators = numpy.sum(b[WAVELET_DICE_NUM_PIX_KEY].values, axis=0)
        a[WAVELET_DICE_COEFF_KEY].values = numerators / denominators

        numerators = numpy.sum(b[WAVELET_TRUE_POSITIVES_KEY].values, axis=0)
        denominators = numpy.sum(
            b[WAVELET_TRUE_POSITIVES_KEY].values +
            b[WAVELET_FALSE_POSITIVES_KEY].values +
            b[WAVELET_FALSE_NEGATIVES_KEY].values,
            axis=0
        )
        a[WAVELET_CSI_KEY].values = numerators / denominators

        a = _compute_peirce_scores(
            basic_score_table_xarray=b, advanced_score_table_xarray=a,
            use_wavelets=True
        )
        a = _compute_heidke_scores(
            basic_score_table_xarray=b, advanced_score_table_xarray=a,
            use_wavelets=True
        )
        a = _compute_gerrity_scores(
            basic_score_table_xarray=b, advanced_score_table_xarray=a,
            use_wavelets=True
        )

    advanced_score_table_xarray = a
    return advanced_score_table_xarray


def find_basic_score_file(top_directory_name, valid_date_string,
                          radar_number=None, raise_error_if_missing=True):
    """Finds NetCDF file with basic scores.

    :param top_directory_name: Name of top-level directory where file is
        expected.
    :param valid_date_string: Valid date (format "yyyymmdd").
    :param radar_number: Radar number (non-negative integer).  If you are
        looking for data on the full grid, leave this alone.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: basic_score_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(top_directory_name)
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if radar_number is not None:
        error_checking.assert_is_integer(radar_number)
        error_checking.assert_is_geq(radar_number, 0)

    basic_score_file_name = (
        '{0:s}/{1:s}/basic_scores_{2:s}{3:s}.nc'
    ).format(
        top_directory_name, valid_date_string[:4], valid_date_string,
        '' if radar_number is None else '_radar{0:d}'.format(radar_number)
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

    valid_date_string = extensionless_file_name.split('_')[2]
    _ = time_conversion.string_to_unix_sec(valid_date_string, DATE_FORMAT)

    return valid_date_string


def find_many_basic_score_files(
        top_directory_name, first_date_string, last_date_string,
        radar_number=None, raise_error_if_all_missing=True,
        raise_error_if_any_missing=False, test_mode=False):
    """Finds many NetCDF files with basic scores.

    :param top_directory_name: See doc for `find_basic_score_file`.
    :param first_date_string: First valid date (format "yyyymmdd").
    :param last_date_string: Last valid date (format "yyyymmdd").
    :param radar_number: See doc for `find_basic_score_file`.
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
            radar_number=radar_number,
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
