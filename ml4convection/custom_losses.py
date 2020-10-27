"""Custom loss functions for Keras models."""

import os
import sys
import copy
import numpy
from tensorflow.keras import backend as K

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


def weighted_xentropy(class_weights):
    """Weighted cross-entropy.

    :param class_weights: length-2 numpy with class weights for loss function.
        Elements will be interpreted as
        (negative_class_weight, positive_class_weight).
    :return: loss: Loss function (defined below).
    """

    def loss(target_tensor, prediction_tensor):
        """Computes loss (weighted cross-entropy).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Weighted cross-entropy.
        """

        weight_tensor = (
            target_tensor * class_weights[1] +
            (1. - target_tensor) * class_weights[0]
        )

        xentropy_tensor = (
            target_tensor * _log2(prediction_tensor) +
            (1. - target_tensor) * _log2(1. - prediction_tensor)
        )

        return -K.mean(weight_tensor * xentropy_tensor)

    return loss


def fractions_skill_score(
        half_window_size_px, use_as_loss_function, mask_matrix,
        function_name=None, test_mode=False):
    """Fractions skill score (FSS).

    M = number of rows in grid
    N = number of columns in grid

    :param half_window_size_px: Number of pixels (grid cells) in half of
        smoothing window (on either side of center).  If this argument is K, the
        window size will be (1 + 2 * K) by (1 + 2 * K).
    :param use_as_loss_function: Boolean flag.  FSS is positively oriented
        (higher is better), but if using it as loss function, we want it to be
        negatively oriented.  Thus, if `use_as_loss_function == True`, will
        return 1 - FSS.  If `use_as_loss_function == False`, will return just
        FSS.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Grid cells marked
        "False" are masked out and not used to compute the loss.
    :param function_name: Function name (string).
    :param test_mode: Leave this alone.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)
    error_checking.assert_is_boolean(use_as_loss_function)
    error_checking.assert_is_boolean(test_mode)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    # TODO(thunderhoser): Allow multiple channels.

    weight_matrix = general_utils.create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )

    if test_mode:
        eroded_mask_matrix = copy.deepcopy(mask_matrix)
    else:
        eroded_mask_matrix = general_utils.erode_binary_matrix(
            binary_matrix=copy.deepcopy(mask_matrix),
            buffer_distance_px=half_window_size_px
        )

    eroded_mask_matrix = numpy.expand_dims(
        eroded_mask_matrix.astype(float), axis=(0, -1)
    )

    def loss(target_tensor, prediction_tensor):
        """Computes loss (fractions skill score).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Fractions skill score.
        """

        smoothed_target_tensor = K.conv2d(
            x=target_tensor, kernel=weight_matrix,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        smoothed_prediction_tensor = K.conv2d(
            x=prediction_tensor, kernel=weight_matrix,
            padding='same', strides=(1, 1), data_format='channels_last'
        )

        eroded_mask_tensor = K.variable(eroded_mask_matrix)
        smoothed_target_tensor = smoothed_target_tensor * eroded_mask_tensor
        smoothed_prediction_tensor = (
            smoothed_prediction_tensor * eroded_mask_tensor
        )

        actual_mse = K.mean(
            (smoothed_target_tensor - smoothed_prediction_tensor) ** 2
        )
        reference_mse = K.mean(
            smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2
        )

        if use_as_loss_function:
            return actual_mse / reference_mse

        return 1. - actual_mse / reference_mse

    if function_name is not None:
        loss.__name__ = function_name

    return loss
