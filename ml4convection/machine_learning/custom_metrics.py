"""Custom metrics for Keras models."""

from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking


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
