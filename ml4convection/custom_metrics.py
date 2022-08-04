"""Custom metrics for Keras models."""

import os
import sys
import copy
import numpy
from tensorflow.keras import backend as K
from scipy.ndimage.morphology import binary_erosion

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import general_utils


def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return K.log(K.maximum(input_tensor, 1e-6)) / K.log(2.)


def _erode_mask(mask_matrix, half_window_size_px):
    """Erodes binary mask.

    :param mask_matrix: See doc for `pod`.
    :param half_window_size_px: Same.
    :return: eroded_mask_matrix: Eroded version of input.
    """

    window_size_px = 2 * half_window_size_px + 1
    structure_matrix = numpy.full(
        (window_size_px, window_size_px), 1, dtype=bool
    )

    eroded_mask_matrix = binary_erosion(
        mask_matrix.astype(int), structure=structure_matrix, iterations=1,
        border_value=1
    )

    return numpy.expand_dims(
        eroded_mask_matrix.astype(float), axis=(0, -1)
    )


def _check_input_args(half_window_size_px, mask_matrix, function_name,
                      test_mode):
    """Error-checks input arguments for any metric.

    :param half_window_size_px: See doc for `pod`.
    :param mask_matrix: Same.
    :param function_name: Same.
    :param test_mode: Same.
    :return: eroded_mask_matrix: Eroded version of input.
    """

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    return _erode_mask(
        mask_matrix=copy.deepcopy(mask_matrix),
        half_window_size_px=half_window_size_px
    )


def _apply_max_filter(input_tensor, half_window_size_px):
    """Applies maximum-filter to tensor.

    :param input_tensor: Keras tensor.
    :param half_window_size_px: Number of pixels in half of filter window (on
        either side of center).  If this argument is K, the window size will be
        (1 + 2 * K) by (1 + 2 * K).
    :return: output_tensor: Filtered version of `input_tensor`.
    """

    window_size_px = 2 * half_window_size_px + 1

    return K.pool2d(
        x=input_tensor, pool_mode='max',
        pool_size=(window_size_px, window_size_px), strides=(1, 1),
        padding='same', data_format='channels_last'
    )


def pod(half_window_size_px, mask_matrix, function_name=None, test_mode=False):
    """Creates function to compute probability of detection.

    M = number of rows in grid
    N = number of columns in grid

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Grid cells marked
        "False" are masked out and not used to compute the loss.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: pod_function: Function (defined below).
    """

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )
    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

    def pod_function(target_tensor, prediction_tensor):
        """Computes probability of detection.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: pod: Probability of detection.
        """

        filtered_prediction_tensor = _apply_max_filter(
            input_tensor=prediction_tensor,
            half_window_size_px=half_window_size_px
        )

        masked_prediction_tensor = (
            eroded_mask_matrix * filtered_prediction_tensor
        )
        masked_target_tensor = eroded_mask_matrix * target_tensor

        num_actual_oriented_true_positives = K.sum(
            masked_target_tensor * masked_prediction_tensor
        )
        num_false_negatives = K.sum(
            masked_target_tensor * (1 - masked_prediction_tensor)
        )

        denominator = (
            num_actual_oriented_true_positives + num_false_negatives +
            K.epsilon()
        )
        return num_actual_oriented_true_positives / denominator

    if function_name is not None:
        pod_function.__name__ = function_name

    return pod_function


def success_ratio(half_window_size_px, mask_matrix, function_name=None,
                  test_mode=False):
    """Creates function to compute success ratio.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: success_ratio_function: Function (defined below).
    """

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )
    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

    def success_ratio_function(target_tensor, prediction_tensor):
        """Computes success ratio.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: success_ratio: Success ratio.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor, half_window_size_px=half_window_size_px
        )

        masked_target_tensor = eroded_mask_matrix * filtered_target_tensor
        masked_prediction_tensor = eroded_mask_matrix * prediction_tensor

        num_prediction_oriented_true_positives = K.sum(
            masked_target_tensor * masked_prediction_tensor
        )
        num_false_positives = K.sum(
            (1 - masked_target_tensor) * masked_prediction_tensor
        )

        denominator = (
            num_prediction_oriented_true_positives + num_false_positives +
            K.epsilon()
        )
        return num_prediction_oriented_true_positives / denominator

    if function_name is not None:
        success_ratio_function.__name__ = function_name

    return success_ratio_function


def frequency_bias(half_window_size_px, mask_matrix, function_name=None,
                   test_mode=False):
    """Creates function to compute frequency bias.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: frequency_bias_function: Function (defined below).
    """

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def frequency_bias_function(target_tensor, prediction_tensor):
        """Computes frequency bias.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: frequency_bias: Frequency bias.
        """

        pod_function = pod(
            half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
            test_mode=test_mode
        )
        pod_value = pod_function(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor
        )

        success_ratio_function = success_ratio(
            half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
            test_mode=test_mode
        )
        success_ratio_value = success_ratio_function(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor
        )

        return pod_value / (success_ratio_value + K.epsilon())

    if function_name is not None:
        frequency_bias_function.__name__ = function_name

    return frequency_bias_function


def csi(half_window_size_px, mask_matrix, use_as_loss_function=False,
        function_name=None, test_mode=False):
    """Creates function to compute critical success index.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param use_as_loss_function: Boolean flag.  If True (False), will use CSI as
        loss function (metric).
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: csi_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)
    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def csi_function(target_tensor, prediction_tensor):
        """Computes critical success index.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: csi: Critical success index.
        """

        pod_function = pod(
            half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
            test_mode=test_mode
        )
        pod_value = K.epsilon() + pod_function(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor
        )

        success_ratio_function = success_ratio(
            half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
            test_mode=test_mode
        )
        success_ratio_value = K.epsilon() + success_ratio_function(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor
        )

        csi_value = (pod_value ** -1 + success_ratio_value ** -1 - 1) ** -1

        if use_as_loss_function:
            return 1. - csi_value

        return csi_value

    if function_name is not None:
        csi_function.__name__ = function_name

    return csi_function


def iou(half_window_size_px, mask_matrix, use_as_loss_function=False,
        function_name=None, test_mode=False):
    """Creates function to compute intersection over union.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param use_as_loss_function: Boolean flag.  If True (False), will use CSI as
        loss function (metric).
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: iou_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )
    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

    def iou_function(target_tensor, prediction_tensor):
        """Computes intersection over union.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: iou: Intersection over union.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        masked_target_tensor = eroded_mask_matrix * filtered_target_tensor
        masked_prediction_tensor = eroded_mask_matrix * prediction_tensor

        masked_target_tensor = masked_target_tensor[..., 0]
        masked_prediction_tensor = masked_prediction_tensor[..., 0]

        intersection_tensor = K.sum(
            masked_target_tensor * masked_prediction_tensor, axis=(1, 2)
        )
        union_tensor = (
            K.sum(masked_target_tensor, axis=(1, 2)) +
            K.sum(masked_prediction_tensor, axis=(1, 2)) -
            intersection_tensor
        )

        iou_value = K.mean(
            intersection_tensor / (union_tensor + K.epsilon())
        )

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        iou_function.__name__ = function_name

    return iou_function


def all_class_iou(half_window_size_px, mask_matrix, use_as_loss_function=False,
                  function_name=None, test_mode=False):
    """Creates function to compute all-class intersection over union (IOU).

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param use_as_loss_function: Boolean flag.  If True (False), will use CSI as
        loss function (metric).
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: iou_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )
    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

    def all_class_iou_function(target_tensor, prediction_tensor):
        """Computes all-class IOU.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: iou_value: All-class IOU.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        masked_target_tensor = eroded_mask_matrix * filtered_target_tensor
        masked_prediction_tensor = eroded_mask_matrix * prediction_tensor
        positive_intersection_tensor = K.sum(
            masked_target_tensor[..., 0] * masked_prediction_tensor[..., 0],
            axis=(1, 2)
        )
        positive_union_tensor = (
            K.sum(masked_target_tensor[..., 0], axis=(1, 2)) +
            K.sum(masked_prediction_tensor[..., 0], axis=(1, 2)) -
            positive_intersection_tensor
        )

        masked_target_tensor = (
            eroded_mask_matrix * (1. - filtered_target_tensor)
        )
        masked_prediction_tensor = eroded_mask_matrix * (1. - prediction_tensor)
        negative_intersection_tensor = K.sum(
            masked_target_tensor[..., 0] * masked_prediction_tensor[..., 0],
            axis=(1, 2)
        )
        negative_union_tensor = (
            K.sum(masked_target_tensor[..., 0], axis=(1, 2)) +
            K.sum(masked_prediction_tensor[..., 0], axis=(1, 2)) -
            negative_intersection_tensor
        )

        positive_iou = K.mean(
            positive_intersection_tensor / (positive_union_tensor + K.epsilon())
        )
        negative_iou = K.mean(
            negative_intersection_tensor / (negative_union_tensor + K.epsilon())
        )
        iou_value = (positive_iou + negative_iou) / 2

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        all_class_iou_function.__name__ = function_name

    return all_class_iou_function


def dice_coeff(half_window_size_px, mask_matrix, use_as_loss_function=False,
               function_name=None, test_mode=False):
    """Creates function to compute Dice coefficient.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param use_as_loss_function: Boolean flag.  If True (False), will use CSI as
        loss function (metric).
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: dice_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )
    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

    def dice_function(target_tensor, prediction_tensor):
        """Computes Dice coefficient.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: dice_coeff: Dice coefficient.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        masked_target_tensor = eroded_mask_matrix * filtered_target_tensor
        masked_prediction_tensor = eroded_mask_matrix * prediction_tensor
        positive_intersection_tensor = K.sum(
            masked_target_tensor[..., 0] * masked_prediction_tensor[..., 0],
            axis=(1, 2)
        )

        masked_target_tensor = (
            eroded_mask_matrix * (1. - filtered_target_tensor)
        )
        masked_prediction_tensor = eroded_mask_matrix * (1. - prediction_tensor)
        negative_intersection_tensor = K.sum(
            masked_target_tensor[..., 0] * masked_prediction_tensor[..., 0],
            axis=(1, 2)
        )

        eroded_mask_tensor = eroded_mask_matrix * K.ones_like(prediction_tensor)
        num_pixels_tensor = K.sum(eroded_mask_tensor, axis=(1, 2, 3))

        dice_value = K.mean(
            (positive_intersection_tensor + negative_intersection_tensor) /
            num_pixels_tensor
        )

        if use_as_loss_function:
            return 1. - dice_value

        return dice_value

    if function_name is not None:
        dice_function.__name__ = function_name

    return dice_function


def brier_score(half_window_size_px, mask_matrix, function_name=None,
                test_mode=False):
    """Creates function to compute Brier score.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: brier_function: Function (defined below).
    """

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )
    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

    def brier_function(target_tensor, prediction_tensor):
        """Computes Brier score.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: brier_score: Brier score.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        masked_target_tensor = eroded_mask_matrix * filtered_target_tensor
        masked_prediction_tensor = eroded_mask_matrix * prediction_tensor

        squared_error_tensor = K.sum(
            (masked_target_tensor - masked_prediction_tensor) ** 2,
            axis=(1, 2, 3)
        )

        eroded_mask_tensor = eroded_mask_matrix * K.ones_like(prediction_tensor)
        num_pixels_tensor = K.sum(eroded_mask_tensor, axis=(1, 2, 3))

        return K.mean(squared_error_tensor / num_pixels_tensor)

    if function_name is not None:
        brier_function.__name__ = function_name

    return brier_function


def cross_entropy(half_window_size_px, mask_matrix, function_name=None,
                  test_mode=False):
    """Creates function to compute cross-entropy.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: xentropy_function: Function (defined below).
    """

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )
    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

    def xentropy_function(target_tensor, prediction_tensor):
        """Computes cross-entropy.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: xentropy: Cross-entropy.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px
        )

        masked_target_tensor = eroded_mask_matrix * filtered_target_tensor
        masked_prediction_tensor = eroded_mask_matrix * prediction_tensor

        xentropy_tensor = K.sum(
            masked_target_tensor * _log2(masked_prediction_tensor) +
            (1. - masked_target_tensor) * _log2(1. - masked_prediction_tensor),
            axis=(1, 2, 3)
        )

        eroded_mask_tensor = eroded_mask_matrix * K.ones_like(prediction_tensor)
        num_pixels_tensor = K.sum(eroded_mask_tensor, axis=(1, 2, 3))

        return -K.mean(xentropy_tensor / num_pixels_tensor)

    if function_name is not None:
        xentropy_function.__name__ = function_name

    return xentropy_function


def crps(half_window_size_px, mask_matrix, function_name=None, test_mode=False):
    """Creates function to compute continuous ranked probability score.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: xentropy_function: Function (defined below).
    """

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )

    weight_matrix_for_targets = general_utils.create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )

    # TODO(thunderhoser): This is a HACK.  Need num estimates to be input arg.
    weight_matrix_for_predictions = general_utils.create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=50
    )

    def crps_function(target_tensor, prediction_tensor):
        """Computes CRPS.

        Adapted from Katherine Haynes:
        https://github.com/thunderhoser/cira_uq4ml/blob/main/crps_loss.ipynb

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: crps_value: CRPS value.
        """

        smoothed_target_tensor = K.conv2d(
            x=target_tensor, kernel=weight_matrix_for_targets,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        smoothed_prediction_tensor = K.conv2d(
            x=prediction_tensor, kernel=weight_matrix_for_predictions,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        smoothed_target_tensor = smoothed_target_tensor * eroded_mask_matrix
        smoothed_prediction_tensor = (
            smoothed_prediction_tensor * eroded_mask_matrix
        )

        mean_prediction_error_tensor = K.mean(
            K.abs(
                smoothed_prediction_tensor -
                K.expand_dims(smoothed_target_tensor, axis=-1)
            ),
            axis=-1
        )

        # prediction_diff_tensor = K.abs(
        #     K.expand_dims(smoothed_prediction_tensor, axis=-1) -
        #     K.expand_dims(smoothed_prediction_tensor, axis=-2)
        # )
        # mean_prediction_diff_tensor = K.mean(
        #     prediction_diff_tensor, axis=(-2, -1)
        # )

        mean_prediction_diff_tensor = K.map_fn(
            fn=lambda p: K.mean(
                K.abs(K.expand_dims(p, axis=-1) - K.expand_dims(p, axis=-2)),
                axis=(-2, -1)
            ),
            elems=smoothed_prediction_tensor
        )

        return K.mean(
            mean_prediction_error_tensor - 0.5 * mean_prediction_diff_tensor
        )

    if function_name is not None:
        crps_function.__name__ = function_name

    return crps_function


def fss_plus_pixelwise_crps(half_window_size_px, mask_matrix,
                            function_name=None, test_mode=False):
    """Creates function to compute FSS plus pixelwise CRPS.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param mask_matrix: See doc for `pod`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: xentropy_function: Function (defined below).
    """

    eroded_mask_matrix = _check_input_args(
        half_window_size_px=half_window_size_px, mask_matrix=mask_matrix,
        function_name=function_name, test_mode=test_mode
    )

    weight_matrix = general_utils.create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )

    def fss_plus_pixelwise_crps_function(target_tensor, prediction_tensor):
        """Computes FSS plus pixelwise CRPS.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss_value: FSS plus pixelwise CRPS.
        """

        smoothed_target_tensor = K.conv2d(
            x=target_tensor, kernel=weight_matrix,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        smoothed_mean_prediction_tensor = K.conv2d(
            x=K.mean(prediction_tensor, axis=-1), kernel=weight_matrix,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        target_tensor = target_tensor * eroded_mask_matrix
        prediction_tensor = prediction_tensor * eroded_mask_matrix
        smoothed_target_tensor = smoothed_target_tensor * eroded_mask_matrix
        smoothed_mean_prediction_tensor = (
            smoothed_mean_prediction_tensor * eroded_mask_matrix
        )

        actual_mse = K.mean(
            (smoothed_target_tensor - smoothed_mean_prediction_tensor) ** 2
        )
        reference_mse = K.mean(
            smoothed_target_tensor ** 2 + smoothed_mean_prediction_tensor ** 2
        )
        reference_mse = K.maximum(reference_mse, K.epsilon())
        fractions_score = actual_mse / reference_mse

        mean_prediction_error_tensor = K.mean(
            K.abs(prediction_tensor - K.expand_dims(target_tensor, axis=-1)),
            axis=-1
        )

        # prediction_diff_tensor = K.abs(
        #     K.expand_dims(prediction_tensor, axis=-1) -
        #     K.expand_dims(prediction_tensor, axis=-2)
        # )
        # mean_prediction_diff_tensor = K.mean(
        #     prediction_diff_tensor, axis=(-2, -1)
        # )

        mean_prediction_diff_tensor = K.map_fn(
            fn=lambda p: K.mean(
                K.abs(K.expand_dims(p, axis=-1) - K.expand_dims(p, axis=-2)),
                axis=(-2, -1)
            ),
            elems=prediction_tensor
        )

        crps_value = K.mean(
            mean_prediction_error_tensor - 0.5 * mean_prediction_diff_tensor
        )

        return fractions_score + crps_value

    if function_name is not None:
        fss_plus_pixelwise_crps_function.__name__ = function_name

    return fss_plus_pixelwise_crps_function
