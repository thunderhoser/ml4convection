"""Custom loss functions for Keras models."""

import os
import sys
import copy
import numpy
import tensorflow
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import general_utils
import fourier_metrics


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


def quantile_loss(quantile_level, mask_matrix):
    """Quantile loss function.

    :param quantile_level: Quantile level.
    :param mask_matrix: See doc for `fractions_skill_score`.
    :return: loss: Loss function (defined below).
    """

    mask_matrix = numpy.expand_dims(
        mask_matrix.astype(float), axis=(0, -1)
    )

    def loss(target_tensor, prediction_tensor):
        """Computes quantile loss.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Quantile loss.
        """

        return K.mean(
            mask_matrix * K.maximum(
                quantile_level * (target_tensor - prediction_tensor),
                (quantile_level - 1) * (target_tensor - prediction_tensor)
            )
        )

    return loss


def quantile_based_fss(
        quantile_level, half_window_size_px, use_as_loss_function, mask_matrix,
        function_name=None):
    """Quantile-based fractions skill score (FSS).

    :param quantile_level: Quantile level.
    :param half_window_size_px: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
    :param mask_matrix: Same.
    :param function_name: Same.
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)
    error_checking.assert_is_boolean(use_as_loss_function)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    weight_matrix = general_utils.create_mean_filter(
        half_num_rows=half_window_size_px,
        half_num_columns=half_window_size_px, num_channels=1
    )

    eroded_mask_matrix = general_utils.erode_binary_matrix(
        binary_matrix=copy.deepcopy(mask_matrix),
        buffer_distance_px=half_window_size_px
    )
    eroded_mask_matrix = numpy.expand_dims(
        eroded_mask_matrix.astype(float), axis=(0, -1)
    )

    def loss(target_tensor, prediction_tensor):
        """Computes loss (quantile-based FSS).

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

        smoothed_target_tensor = smoothed_target_tensor * eroded_mask_matrix
        smoothed_prediction_tensor = (
            smoothed_prediction_tensor * eroded_mask_matrix
        )

        multiplier_tensor = tensorflow.where(
            smoothed_target_tensor > smoothed_prediction_tensor,
            x=quantile_level, y=1. - quantile_level
        )
        actual_mse = K.mean(
            multiplier_tensor *
            (smoothed_target_tensor - smoothed_prediction_tensor) ** 2
        )
        reference_mse = K.mean(
            smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2
        )
        reference_mse = K.maximum(reference_mse, K.epsilon())

        if use_as_loss_function:
            return actual_mse / reference_mse

        return 1. - actual_mse / reference_mse

    if function_name is not None:
        loss.__name__ = function_name

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
    # eroded_mask_tensor = K.variable(eroded_mask_matrix)

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

        smoothed_target_tensor = smoothed_target_tensor * eroded_mask_matrix
        smoothed_prediction_tensor = (
            smoothed_prediction_tensor * eroded_mask_matrix
        )

        actual_mse = K.mean(
            (smoothed_target_tensor - smoothed_prediction_tensor) ** 2
        )
        reference_mse = K.mean(
            smoothed_target_tensor ** 2 + smoothed_prediction_tensor ** 2
        )
        reference_mse = K.maximum(reference_mse, K.epsilon())

        if use_as_loss_function:
            return actual_mse / reference_mse

        return 1. - actual_mse / reference_mse

    if function_name is not None:
        loss.__name__ = function_name

    return loss


def cross_entropy(mask_matrix, function_name=None):
    """Cross-entropy.

    M = number of rows in grid
    N = number of columns in grid

    :param mask_matrix: M-by-N numpy array of Boolean flags.  Grid cells marked
        "False" are masked out and not used to compute the loss.
    :param function_name: Function name (string).
    :return: loss: Loss function (defined below).
    """

    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    mask_matrix_4d = copy.deepcopy(mask_matrix)
    mask_matrix_4d = numpy.expand_dims(
        mask_matrix_4d.astype(float), axis=(0, -1)
    )

    def loss(target_tensor, prediction_tensor):
        """Computes loss (cross-entropy).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: loss: Fractions skill score.
        """

        filtered_target_tensor = target_tensor * mask_matrix_4d
        filtered_prediction_tensor = prediction_tensor * mask_matrix_4d

        xentropy_tensor = (
            filtered_target_tensor * _log2(filtered_prediction_tensor) +
            (1. - filtered_target_tensor) *
            _log2(1. - filtered_prediction_tensor)
        )

        return -K.mean(xentropy_tensor)

    if function_name is not None:
        loss.__name__ = function_name

    return loss


def heidke_score(mask_matrix, use_as_loss_function, function_name=None):
    """Creates function to compute Heidke score at given scale.

    :param mask_matrix: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
    :param function_name: Same.
    :return: heidke_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)
    mask_matrix = numpy.expand_dims(mask_matrix, axis=0).astype(numpy.float32)

    def heidke_function(target_tensor, prediction_tensor):
        """Computes Heidke score at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: heidke_value: Heidke score (scalar).
        """

        heidke_value = fourier_metrics.get_heidke_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - heidke_value

        return heidke_value

    if function_name is not None:
        heidke_function.__name__ = function_name

    return heidke_function


def peirce_score(mask_matrix, use_as_loss_function, function_name=None):
    """Creates function to compute Peirce score at given scale.

    :param mask_matrix: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
    :param function_name: Same.
    :return: peirce_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)
    mask_matrix = numpy.expand_dims(mask_matrix, axis=0).astype(numpy.float32)

    def peirce_function(target_tensor, prediction_tensor):
        """Computes Peirce score at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: peirce_value: Peirce score (scalar).
        """

        peirce_value = fourier_metrics.get_peirce_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - peirce_value

        return peirce_value

    if function_name is not None:
        peirce_function.__name__ = function_name

    return peirce_function


def gerrity_score(mask_matrix, use_as_loss_function, function_name=None):
    """Creates function to compute Gerrity score at given scale.

    :param mask_matrix: See doc for `fractions_skill_score`.
    :param use_as_loss_function: Same.
    :param function_name: Same.
    :return: gerrity_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)
    mask_matrix = numpy.expand_dims(mask_matrix, axis=0).astype(numpy.float32)

    def gerrity_function(target_tensor, prediction_tensor):
        """Computes Gerrity score at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: gerrity_value: Gerrity score (scalar).
        """

        gerrity_value = fourier_metrics.get_gerrity_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - gerrity_value

        return gerrity_value

    if function_name is not None:
        gerrity_function.__name__ = function_name

    return gerrity_function
