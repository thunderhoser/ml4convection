"""Custom metrics for Keras models."""

import tensorflow
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6


def _apply_max_filter(input_tensor, half_window_size_px, test_mode):
    """Applies maximum-filter to tensor.

    :param input_tensor: Keras tensor.
    :param half_window_size_px: Number of pixels in half of filter window (on
        either side of center).  If this argument is K, the window size will be
        (1 + 2 * K) by (1 + 2 * K).
    :param test_mode: Leave this alone.
    :return: output_tensor: Filtered version of `input_tensor`.
    """

    window_size_px = 2 * half_window_size_px + 1

    return K.pool2d(
        x=input_tensor, pool_mode='max',
        pool_size=(window_size_px, window_size_px), strides=(1, 1),
        padding='same' if test_mode else 'valid', data_format='channels_last'
    )


def _match_predicted_convection(
        target_tensor, prediction_tensor, half_window_size_px, test_mode):
    """Matches each point with predicted convection to actual convection.

    :param target_tensor: Tensor of target (actual) values.
    :param prediction_tensor: Tensor of predicted values.
    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param test_mode: Leave this alone.
    :return: num_prediction_oriented_true_positives: Number of prediction-
        oriented true positives.
    :return: num_false_positives: Number of false positives.
    """

    filtered_target_tensor = _apply_max_filter(
        input_tensor=target_tensor, half_window_size_px=half_window_size_px,
        test_mode=test_mode
    )

    num_prediction_oriented_true_positives = K.sum(
        filtered_target_tensor * prediction_tensor
    )
    num_false_positives = K.sum(
        (1 - filtered_target_tensor) * prediction_tensor
    )

    return num_prediction_oriented_true_positives, num_false_positives


def _match_actual_convection(
        target_tensor, prediction_tensor, half_window_size_px, test_mode):
    """Matches each point with actual convection to prediction convection.

    :param target_tensor: Tensor of target (actual) values.
    :param prediction_tensor: Tensor of predicted values.
    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param test_mode: Leave this alone.
    :return: num_actual_oriented_true_positives: Number of actual-oriented true
        positives.
    :return: num_false_negatives: Number of false negatives.
    """

    filtered_prediction_tensor = _apply_max_filter(
        input_tensor=prediction_tensor, half_window_size_px=half_window_size_px,
        test_mode=test_mode
    )

    num_actual_oriented_true_positives = K.sum(
        target_tensor * filtered_prediction_tensor
    )
    num_false_negatives = K.sum(
        target_tensor * (1 - filtered_prediction_tensor)
    )

    return num_actual_oriented_true_positives, num_false_negatives


def pod(half_window_size_px, function_name=None, test_mode=False):
    """Creates function to compute probability of detection.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: pod_function: Function (defined below).
    """

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def pod_function(target_tensor, prediction_tensor):
        """Computes probability of detection.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: pod: Probability of detection.
        """

        filtered_prediction_tensor = _apply_max_filter(
            input_tensor=prediction_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )

        num_actual_oriented_true_positives = K.sum(
            target_tensor * filtered_prediction_tensor
        )
        num_false_negatives = K.sum(
            target_tensor * (1 - filtered_prediction_tensor)
        )

        denominator = (
            num_actual_oriented_true_positives + num_false_negatives +
            K.epsilon()
        )
        return num_actual_oriented_true_positives / denominator

    if function_name is not None:
        pod_function.__name__ = function_name

    return pod_function


def success_ratio(half_window_size_px, function_name=None, test_mode=False):
    """Creates function to compute success ratio.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: success_ratio_function: Function (defined below).
    """

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def success_ratio_function(target_tensor, prediction_tensor):
        """Computes success ratio.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: success_ratio: Success ratio.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor, half_window_size_px=half_window_size_px,
            test_mode=test_mode
        )

        num_prediction_oriented_true_positives = K.sum(
            filtered_target_tensor * prediction_tensor
        )
        num_false_positives = K.sum(
            (1 - filtered_target_tensor) * prediction_tensor
        )

        denominator = (
            num_prediction_oriented_true_positives + num_false_positives +
            K.epsilon()
        )
        return num_prediction_oriented_true_positives / denominator

    if function_name is not None:
        success_ratio_function.__name__ = function_name

    return success_ratio_function


def frequency_bias(half_window_size_px, function_name=None, test_mode=False):
    """Creates function to compute frequency bias.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: frequency_bias_function: Function (defined below).
    """

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def frequency_bias_function(target_tensor, prediction_tensor):
        """Computes frequency bias.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: frequency_bias: Frequency bias.
        """

        pod_function = pod(
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )
        pod_value = pod_function(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor
        )

        success_ratio_function = success_ratio(
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )
        success_ratio_value = success_ratio_function(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor
        )

        return pod_value / (success_ratio_value + K.epsilon())

    if function_name is not None:
        frequency_bias_function.__name__ = function_name

    return frequency_bias_function


def csi(half_window_size_px, function_name=None, test_mode=False):
    """Creates function to compute critical success index.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: csi_function: Function (defined below).
    """

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def csi_function(target_tensor, prediction_tensor):
        """Computes critical success index.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: csi: Critical success index.
        """

        pod_function = pod(
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )
        pod_value = K.epsilon() + pod_function(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor
        )

        success_ratio_function = success_ratio(
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )
        success_ratio_value = K.epsilon() + success_ratio_function(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor
        )

        return (pod_value ** -1 + success_ratio_value ** -1 - 1) ** -1

    if function_name is not None:
        csi_function.__name__ = function_name

    return csi_function


def dice_coeff(half_window_size_px, function_name=None, test_mode=False):
    """Creates function to compute Dice coefficient.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: dice_function: Function (defined below).
    """

    # TODO(thunderhoser): Need _check_input_args... maybe.

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def dice_function(target_tensor, prediction_tensor):
        """Computes Dice coefficient.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: dice_coeff: Dice coefficient.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )
        filtered_prediction_tensor = _apply_max_filter(
            input_tensor=prediction_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )

        intersection_tensor = K.sum(
            filtered_target_tensor * filtered_prediction_tensor, axis=(1, 2, 3)
        )
        num_pixels_tensor = K.sum(
            K.ones_like(filtered_target_tensor), axis=(1, 2, 3)
        )

        return K.mean(2 * intersection_tensor / num_pixels_tensor)

    if function_name is not None:
        dice_function.__name__ = function_name

    return dice_function


def iou(half_window_size_px, function_name=None, test_mode=False):
    """Creates function to compute intersection over union.

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: iou_function: Function (defined below).
    """

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def iou_function(target_tensor, prediction_tensor):
        """Computes intersection over union.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: iou: Intersection over union.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )
        filtered_prediction_tensor = _apply_max_filter(
            input_tensor=prediction_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )

        intersection_tensor = K.sum(
            filtered_target_tensor * filtered_prediction_tensor, axis=(1, 2, 3)
        )
        union_tensor = (
            K.sum(filtered_target_tensor, axis=(1, 2, 3)) +
            K.sum(filtered_prediction_tensor, axis=(1, 2, 3)) -
            intersection_tensor
        )

        return K.mean(
            intersection_tensor / (union_tensor + K.epsilon())
        )

    if function_name is not None:
        iou_function.__name__ = function_name

    return iou_function


def tversky_coeff(
        half_window_size_px, false_positive_weight, false_negative_weight,
        function_name=None, test_mode=False):
    """Creates function to compute Tversky coefficient (weighted CSI).

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param false_positive_weight: Weight for false positives.
    :param false_negative_weight: Weight for false negatives.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: tversky_function: Function (defined below).
    """

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_greater(false_positive_weight, 0.)
    error_checking.assert_is_greater(false_negative_weight, 0.)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def tversky_function(target_tensor, prediction_tensor):
        """Computes Tversky coefficient.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: tversky_coeff: Tversky coefficient.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )
        filtered_prediction_tensor = _apply_max_filter(
            input_tensor=prediction_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )

        num_true_positives = false_positive_weight * K.sum(
            filtered_prediction_tensor * filtered_target_tensor
        )
        num_false_positives = false_positive_weight * K.sum(
            filtered_prediction_tensor * (1. - filtered_target_tensor)
        )
        num_false_negatives = false_negative_weight * K.sum(
            filtered_target_tensor * (1. - filtered_prediction_tensor)
        )

        denominator = (
            num_false_positives + num_false_negatives + num_true_positives +
            K.epsilon()
        )

        return num_true_positives / denominator

    if function_name is not None:
        tversky_function.__name__ = function_name

    return tversky_function


def focal_loss(
        half_window_size_px, training_event_freq, focusing_factor,
        function_name=None, test_mode=False):
    """Creates function to compute focal loss.

    Paper reference: https://arxiv.org/pdf/1708.02002.pdf
    Code reference: https://github.com/umbertogriffo/focal-loss-keras/blob/
                    master/src/loss_function/losses.py

    :param half_window_size_px: See doc for `_apply_max_filter`.
    :param training_event_freq: Event frequency (positive-class frequency) in
        training data.
    :param focusing_factor: Focusing factor (gamma in referenced paper).
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Function (defined below).
    """

    error_checking.assert_is_integer(half_window_size_px)
    error_checking.assert_is_geq(half_window_size_px, 0)
    error_checking.assert_is_greater(training_event_freq, 0.)
    error_checking.assert_is_less_than(training_event_freq, 1.)
    error_checking.assert_is_geq(focusing_factor, 1.)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    def loss(target_tensor, prediction_tensor):
        """Computes loss.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Focal loss.
        """

        filtered_target_tensor = _apply_max_filter(
            input_tensor=target_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )
        filtered_prediction_tensor = _apply_max_filter(
            input_tensor=prediction_tensor,
            half_window_size_px=half_window_size_px, test_mode=test_mode
        )

        filtered_prediction_tensor = K.clip(
            filtered_prediction_tensor, K.epsilon(), 1. - K.epsilon()
        )
        probability_tensor = tensorflow.where(
            K.equal(filtered_target_tensor, 1),
            filtered_prediction_tensor, 1. - filtered_prediction_tensor
        )

        alpha_tensor = training_event_freq * K.ones_like(filtered_target_tensor)
        alpha_tensor = tensorflow.where(
            K.equal(filtered_target_tensor, 1), alpha_tensor, 1. - alpha_tensor
        )

        cross_entropy_tensor = -K.log(probability_tensor)
        weight_tensor = alpha_tensor * K.pow(
            (1. - probability_tensor), focusing_factor
        )
        return K.mean(K.sum(weight_tensor * cross_entropy_tensor, axis=0))

    if function_name is not None:
        loss.__name__ = function_name

    return loss
