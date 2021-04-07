"""Scale-separation-based metrics (with Fourier decomp) for Keras models."""

import os
import sys
import numpy
import tensorflow
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import fourier_utils

SPATIAL_COEFFS_KEY = 'spatial_coeff_matrix'
FREQUENCY_COEFFS_KEY = 'frequency_coeff_matrix'
MASK_KEY = 'mask_matrix'
ORIG_NUM_ROWS_KEY = 'orig_num_rows'
ORIG_NUM_COLUMNS_KEY = 'orig_num_columns'

FSS_NAME = 'fss'
BRIER_SCORE_NAME = 'brier'
CSI_NAME = 'csi'
FREQUENCY_BIAS_NAME = 'bias'
IOU_NAME = 'iou'
DICE_COEFF_NAME = 'dice'
REAL_FREQ_MSE_NAME = 'fmser'
IMAGINARY_FREQ_MSE_NAME = 'fmsei'
FREQ_MSE_NAME = 'fmse'

VALID_SCORE_NAMES = [
    FSS_NAME, BRIER_SCORE_NAME, CSI_NAME, FREQUENCY_BIAS_NAME, IOU_NAME,
    DICE_COEFF_NAME, REAL_FREQ_MSE_NAME, IMAGINARY_FREQ_MSE_NAME, FREQ_MSE_NAME
]


def _check_input_args(
        spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        function_name):
    """Error-checks input arguments for any metric.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param spatial_coeff_matrix: numpy array (3M x 3N) of coefficients for
        window function, to be applied in spatial domain.
    :param frequency_coeff_matrix: numpy array (3M x 3N) of coefficients for
        filter function, to be applied in frequency domain.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Grid points marked
        "False" are masked out and not used to compute the metric.
    :param function_name: Function name (string).  May be None.
    :return: argument_dict: Dictionary with the following keys.
    argument_dict['spatial_coeff_matrix']: Same as input but with dimensions of
        1 x 3M x 3N.
    argument_dict['frequency_coeff_matrix']: Same as input but with dimensions
        of 1 x 3M x 3N.
    argument_dict['mask_matrix']: Same as input but with dimensions of
        1 x 3M x 3N.
    argument_dict['orig_num_rows']: M in the above discussion.
    argument_dict['orig_num_columns']: N in the above discussion.
    """

    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)

    orig_num_rows = mask_matrix.shape[0]
    orig_num_columns = mask_matrix.shape[1]
    expected_dim = numpy.array(
        [3 * orig_num_rows, 3 * orig_num_columns], dtype=int
    )

    error_checking.assert_is_numpy_array_without_nan(spatial_coeff_matrix)
    error_checking.assert_is_numpy_array(
        spatial_coeff_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(frequency_coeff_matrix)
    error_checking.assert_is_numpy_array(
        frequency_coeff_matrix, exact_dimensions=expected_dim
    )

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    spatial_coeff_matrix = numpy.expand_dims(spatial_coeff_matrix, axis=0)
    frequency_coeff_matrix = numpy.expand_dims(frequency_coeff_matrix, axis=0)
    mask_matrix = fourier_utils.taper_spatial_data(mask_matrix.astype(float))
    mask_matrix = numpy.expand_dims(mask_matrix, axis=0).astype(numpy.float32)

    return {
        SPATIAL_COEFFS_KEY: spatial_coeff_matrix,
        FREQUENCY_COEFFS_KEY: frequency_coeff_matrix,
        MASK_KEY: mask_matrix,
        ORIG_NUM_ROWS_KEY: orig_num_rows,
        ORIG_NUM_COLUMNS_KEY: orig_num_columns
    }


def _check_score_name(score_name):
    """Error-checks name of evaluation score.

    :param score_name: Name of evaluation score.
    :raises: ValueError: if `score_name not in VALID_SCORE_NAMES`.
    """

    error_checking.assert_is_string(score_name)
    if score_name in VALID_SCORE_NAMES:
        return

    error_string = (
        'Valid scores (listed below) do not include "{0:s}":\n{1:s}'
    ).format(score_name, str(VALID_SCORE_NAMES))

    raise ValueError(error_string)


def _filter_fields(
        target_tensor, prediction_tensor, spatial_coeff_matrix,
        frequency_coeff_matrix, orig_num_rows, orig_num_columns):
    """Filters predicted and target fields via Fourier transform, then inverse.

    :param target_tensor: Tensor of target (actual) values.
    :param prediction_tensor: Tensor of predicted values.
    :param spatial_coeff_matrix: See output doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param orig_num_rows: Same.
    :param orig_num_columns: Same.
    :return: target_tensor: Filtered version of input.
    :return: prediction_tensor: Filtered version of input.
    :return: target_coeff_tensor: Tensor of filtered Fourier weights, in the
        same shape as `target_tensor`.
    :return: predicted_coeff_tensor: Tensor of filtered Fourier weights, in the
        same shape as `prediction_tensor`.
    """

    padding_arg = (
        (orig_num_rows, orig_num_rows),
        (orig_num_columns, orig_num_columns)
    )

    target_tensor = K.spatial_2d_padding(
        target_tensor, padding=padding_arg, data_format='channels_last'
    )
    target_tensor = spatial_coeff_matrix * target_tensor[..., 0]

    target_coeff_tensor = frequency_coeff_matrix * tensorflow.signal.fft2d(
        K.cast(target_tensor, dtype=tensorflow.complex64)
    )

    target_tensor = tensorflow.math.real(
        tensorflow.signal.ifft2d(target_coeff_tensor)
    )
    target_tensor = K.maximum(target_tensor, 0.)
    target_tensor = K.minimum(target_tensor, 1.)

    prediction_tensor = K.spatial_2d_padding(
        prediction_tensor, padding=padding_arg, data_format='channels_last'
    )
    prediction_tensor = spatial_coeff_matrix * prediction_tensor[..., 0]

    predicted_coeff_tensor = (
        frequency_coeff_matrix *
        tensorflow.signal.fft2d(
            K.cast(prediction_tensor, dtype=tensorflow.complex64)
        )
    )

    prediction_tensor = tensorflow.math.real(
        tensorflow.signal.ifft2d(predicted_coeff_tensor)
    )
    prediction_tensor = K.maximum(prediction_tensor, 0.)
    prediction_tensor = K.minimum(prediction_tensor, 1.)

    return (
        target_tensor, prediction_tensor,
        target_coeff_tensor, predicted_coeff_tensor
    )


def _get_brier_score(target_tensor, prediction_tensor, mask_matrix):
    """Computes Brier score.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param target_tensor: Tensor of target (actual) values.
    :param prediction_tensor: Tensor of predicted values.
    :param mask_matrix: M-by-N numpy array of Boolean flags.  Grid points marked
        "False" are masked out and not used to compute the metric.
    :return: brier_score: Brier score (scalar).
    """

    squared_error_tensor = K.sum(
        (target_tensor - prediction_tensor) ** 2,
        axis=(1, 2)
    )

    mask_tensor = mask_matrix * K.ones_like(prediction_tensor)
    num_pixels_tensor = K.sum(mask_tensor, axis=(1, 2))

    return K.mean(squared_error_tensor / num_pixels_tensor)


def _get_csi(target_tensor, prediction_tensor, mask_matrix):
    """Computes critical success index.

    :param target_tensor: See doc for `_get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: csi: Critical success index (scalar).
    """

    num_true_positives = K.sum(
        mask_matrix * target_tensor * prediction_tensor
    )
    num_false_positives = K.sum(
        mask_matrix * (1 - target_tensor) * prediction_tensor
    )
    num_false_negatives = K.sum(
        mask_matrix * target_tensor * (1 - prediction_tensor)
    )

    denominator = (
        num_true_positives + num_false_positives + num_false_negatives +
        K.epsilon()
    )

    return num_true_positives / denominator


def _get_frequency_bias(target_tensor, prediction_tensor, mask_matrix):
    """Computes frequency bias.

    :param target_tensor: See doc for `_get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: frequency_bias: Frequency bias (scalar).
    """

    numerator = K.sum(mask_matrix * prediction_tensor)
    denominator = K.sum(mask_matrix * target_tensor) + K.epsilon()

    return numerator / denominator


def _get_pixelwise_fss(target_tensor, prediction_tensor, mask_matrix):
    """Computes pixelwise fractions skill score (FSS).

    :param target_tensor: See doc for `_get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: pixelwise_fss: Pixelwise FSS (scalar).
    """

    masked_target_tensor = target_tensor * mask_matrix
    masked_prediction_tensor = prediction_tensor * mask_matrix

    actual_mse = K.mean(
        (masked_target_tensor - masked_prediction_tensor) ** 2
    )
    reference_mse = K.mean(
        masked_target_tensor ** 2 + masked_prediction_tensor ** 2
    )

    return 1. - actual_mse / reference_mse


def _get_iou(target_tensor, prediction_tensor, mask_matrix):
    """Computes intersection over union.

    :param target_tensor: See doc for `_get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: iou: Intersection over union (scalar).
    """

    masked_target_tensor = target_tensor * mask_matrix
    masked_prediction_tensor = prediction_tensor * mask_matrix

    intersection_tensor = K.sum(
        masked_target_tensor * masked_prediction_tensor, axis=(1, 2)
    )
    union_tensor = (
        K.sum(masked_target_tensor, axis=(1, 2)) +
        K.sum(masked_prediction_tensor, axis=(1, 2)) -
        intersection_tensor
    )

    return K.mean(
        intersection_tensor / (union_tensor + K.epsilon())
    )


def _get_dice_coeff(target_tensor, prediction_tensor, mask_matrix):
    """Computes Dice coefficient.

    :param target_tensor: See doc for `_get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: dice_coeff: Dice coefficient (scalar).
    """

    masked_target_tensor = mask_matrix * target_tensor
    masked_prediction_tensor = mask_matrix * prediction_tensor
    positive_intersection_tensor = K.sum(
        masked_target_tensor * masked_prediction_tensor, axis=(1, 2)
    )

    masked_target_tensor = mask_matrix * (1. - target_tensor)
    masked_prediction_tensor = mask_matrix * (1. - prediction_tensor)
    negative_intersection_tensor = K.sum(
        masked_target_tensor * masked_prediction_tensor, axis=(1, 2)
    )

    mask_tensor = mask_matrix * K.ones_like(prediction_tensor)
    num_pixels_tensor = K.sum(mask_tensor, axis=(1, 2))

    return K.mean(
        (positive_intersection_tensor + negative_intersection_tensor) /
        num_pixels_tensor
    )


def _get_frequency_domain_mse(
        target_coeff_tensor, predicted_coeff_tensor, include_real,
        include_imaginary):
    """Computes mean squared error (MSE) in frequency domain.

    :param target_coeff_tensor: See output doc for `_filter_fields`.
    :param predicted_coeff_tensor: Same.
    :param include_real: See input doc for `frequency_domain_mse`.
    :param include_imaginary: Same.
    """

    error_checking.assert_is_boolean(include_real)
    error_checking.assert_is_boolean(include_imaginary)
    error_checking.assert_is_greater(
        int(include_real) + int(include_imaginary), 0
    )

    if not include_imaginary:
        target_coeff_tensor = tensorflow.math.real(target_coeff_tensor)
        predicted_coeff_tensor = tensorflow.math.real(
            predicted_coeff_tensor
        )

    if not include_real:
        target_coeff_tensor = tensorflow.math.imag(target_coeff_tensor)
        predicted_coeff_tensor = tensorflow.math.imag(
            predicted_coeff_tensor
        )

    return K.mean(K.abs(target_coeff_tensor - predicted_coeff_tensor) ** 2)


def brier_score(
        spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        use_as_loss_function=True, function_name=None):
    """Creates function to compute Brier score at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 - CSI, to
        use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: brier_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    mask_matrix = argument_dict[MASK_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def brier_function(target_tensor, prediction_tensor):
        """Computes Brier score at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: brier_score: Brier score (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        return _get_brier_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

    if function_name is not None:
        brier_function.__name__ = function_name

    return brier_function


def csi(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        use_as_loss_function, function_name=None):
    """Creates fctn to compute critical success index (CSI) at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 - CSI, to
        use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: csi_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    mask_matrix = argument_dict[MASK_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def csi_function(target_tensor, prediction_tensor):
        """Computes CSI at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: csi: Critical success index (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        csi_value = _get_csi(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - csi_value

        return csi_value

    if function_name is not None:
        csi_function.__name__ = function_name

    return csi_function


def frequency_bias(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
                   function_name=None):
    """Creates function to compute frequency bias at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param function_name: See doc for `_check_input_args`.
    :return: bias_function: Function (defined below).
    """

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    mask_matrix = argument_dict[MASK_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def bias_function(target_tensor, prediction_tensor):
        """Computes frequency bias at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: frequency_bias: Frequency bias (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        return _get_frequency_bias(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

    if function_name is not None:
        bias_function.__name__ = function_name

    return bias_function


def pixelwise_fss(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
                  use_as_loss_function, function_name=None):
    """Creates function to compute pixelwise FSS at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 - CSI, to
        use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: pixelwise_fss_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    mask_matrix = argument_dict[MASK_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def pixelwise_fss_function(target_tensor, prediction_tensor):
        """Computes pixelwise fractions skill score (FSS) at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: pixelwise_fss: Pixelwise FSS (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        fss_value = _get_pixelwise_fss(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - fss_value

        return fss_value

    if function_name is not None:
        pixelwise_fss_function.__name__ = function_name

    return pixelwise_fss_function


def iou(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        use_as_loss_function, function_name=None):
    """Creates function to compute intersctn over union (IOU) at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 - CSI, to
        use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: iou_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    mask_matrix = argument_dict[MASK_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def iou_function(target_tensor, prediction_tensor):
        """Computes intersection over union (IOU) at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: iou: Intersection over union (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        iou_value = _get_iou(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        iou_function.__name__ = function_name

    return iou_function


def dice_coeff(
        spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        use_as_loss_function, function_name=None):
    """Creates function to compute Dice coefficient at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 - CSI, to
        use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: dice_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    mask_matrix = argument_dict[MASK_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def dice_function(target_tensor, prediction_tensor):
        """Computes Dice coefficient at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: dice_coeff: Dice coefficient (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        dice_value = _get_dice_coeff(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - dice_value

        return dice_value

    if function_name is not None:
        dice_function.__name__ = function_name

    return dice_function


def metrics(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
            metric_names, function_name=None):
    """Creates function to compute several metrics at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param metric_names: 1-D list of metric names (strings).  Each metric name
        should start with the name of a score (accepted by `_check_score_name`),
        followed by an underscore.
    :param function_name: Name of this function (may be None).
    :return: metrics_function: Function (defined below).
    """

    error_checking.assert_is_string_list(metric_names)
    score_names = [n.split('_')[0] for n in metric_names]
    for this_score_name in score_names:
        _check_score_name(this_score_name)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    mask_matrix = argument_dict[MASK_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def metrics_function(target_tensor, prediction_tensor):
        """Computes several metrics at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: metrics_dict: Dictionary, where each key is a metric name and
            each value is a metric value (scalar).
        """

        (
            target_tensor, prediction_tensor,
            target_coeff_tensor, predicted_coeff_tensor
        ) = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )

        metrics_dict = dict()

        for i in range(len(metric_names)):
            if score_names[i] == BRIER_SCORE_NAME:
                metrics_dict[metric_names[i]] = _get_brier_score(
                    target_tensor=target_tensor,
                    prediction_tensor=prediction_tensor,
                    mask_matrix=mask_matrix
                )
            elif score_names[i] == CSI_NAME:
                metrics_dict[metric_names[i]] = _get_csi(
                    target_tensor=target_tensor,
                    prediction_tensor=prediction_tensor,
                    mask_matrix=mask_matrix
                )
            elif score_names[i] == FREQUENCY_BIAS_NAME:
                metrics_dict[metric_names[i]] = _get_frequency_bias(
                    target_tensor=target_tensor,
                    prediction_tensor=prediction_tensor,
                    mask_matrix=mask_matrix
                )
            elif score_names[i] == FSS_NAME:
                metrics_dict[metric_names[i]] = _get_pixelwise_fss(
                    target_tensor=target_tensor,
                    prediction_tensor=prediction_tensor,
                    mask_matrix=mask_matrix
                )
            elif score_names[i] == IOU_NAME:
                metrics_dict[metric_names[i]] = _get_iou(
                    target_tensor=target_tensor,
                    prediction_tensor=prediction_tensor,
                    mask_matrix=mask_matrix
                )
            elif score_names[i] == DICE_COEFF_NAME:
                metrics_dict[metric_names[i]] = _get_dice_coeff(
                    target_tensor=target_tensor,
                    prediction_tensor=prediction_tensor,
                    mask_matrix=mask_matrix
                )
            elif score_names[i] == REAL_FREQ_MSE_NAME:
                metrics_dict[metric_names[i]] = _get_frequency_domain_mse(
                    target_coeff_tensor=target_coeff_tensor,
                    predicted_coeff_tensor=predicted_coeff_tensor,
                    include_real=True, include_imaginary=False
                )
            elif score_names[i] == IMAGINARY_FREQ_MSE_NAME:
                metrics_dict[metric_names[i]] = _get_frequency_domain_mse(
                    target_coeff_tensor=target_coeff_tensor,
                    predicted_coeff_tensor=predicted_coeff_tensor,
                    include_real=False, include_imaginary=True
                )
            else:
                metrics_dict[metric_names[i]] = _get_frequency_domain_mse(
                    target_coeff_tensor=target_coeff_tensor,
                    predicted_coeff_tensor=predicted_coeff_tensor,
                    include_real=True, include_imaginary=True
                )

        return metrics_dict

    if function_name is not None:
        metrics_function.__name__ = function_name

    return metrics_function


def frequency_domain_mse_real(
        spatial_coeff_matrix, frequency_coeff_matrix, function_name=None):
    """Creates function to compute MSE on real part of Fourier weights.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param function_name: Same.
    :return: mse_function: Function (defined below).
    """

    orig_num_rows = int(numpy.round(
        float(spatial_coeff_matrix.shape[0]) / 3
    ))
    orig_num_columns = int(numpy.round(
        float(spatial_coeff_matrix.shape[1]) / 3
    ))
    mask_matrix = numpy.full((orig_num_rows, orig_num_columns), 1, dtype=bool)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def mse_function(target_tensor, prediction_tensor):
        """Computes MSE on real part of Fourier weights.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: mse: Mean squared error (scalar).
        """

        target_coeff_tensor, predicted_coeff_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[2:]

        return _get_frequency_domain_mse(
            target_coeff_tensor=target_coeff_tensor,
            predicted_coeff_tensor=predicted_coeff_tensor,
            include_real=True, include_imaginary=False
        )

    if function_name is not None:
        mse_function.__name__ = function_name

    return mse_function


def frequency_domain_mse_imag(
        spatial_coeff_matrix, frequency_coeff_matrix, function_name=None):
    """Creates function to compute MSE on imaginary part of Fourier weights.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param function_name: Same.
    :return: mse_function: Function (defined below).
    """

    orig_num_rows = int(numpy.round(
        float(spatial_coeff_matrix.shape[0]) / 3
    ))
    orig_num_columns = int(numpy.round(
        float(spatial_coeff_matrix.shape[1]) / 3
    ))
    mask_matrix = numpy.full((orig_num_rows, orig_num_columns), 1, dtype=bool)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def mse_function(target_tensor, prediction_tensor):
        """Computes MSE on imaginary part of Fourier weights.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: mse: Mean squared error (scalar).
        """

        target_coeff_tensor, predicted_coeff_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[2:]

        return _get_frequency_domain_mse(
            target_coeff_tensor=target_coeff_tensor,
            predicted_coeff_tensor=predicted_coeff_tensor,
            include_real=False, include_imaginary=True
        )

    if function_name is not None:
        mse_function.__name__ = function_name

    return mse_function


def frequency_domain_mse(
        spatial_coeff_matrix, frequency_coeff_matrix, function_name=None):
    """Creates function to compute MSE on Fourier weights (real & imagnry part).

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param function_name: Same.
    :return: mse_function: Function (defined below).
    """

    orig_num_rows = int(numpy.round(
        float(spatial_coeff_matrix.shape[0]) / 3
    ))
    orig_num_columns = int(numpy.round(
        float(spatial_coeff_matrix.shape[1]) / 3
    ))
    mask_matrix = numpy.full((orig_num_rows, orig_num_columns), 1, dtype=bool)

    argument_dict = _check_input_args(
        spatial_coeff_matrix=spatial_coeff_matrix,
        frequency_coeff_matrix=frequency_coeff_matrix,
        mask_matrix=mask_matrix, function_name=function_name
    )

    spatial_coeff_matrix = argument_dict[SPATIAL_COEFFS_KEY]
    frequency_coeff_matrix = argument_dict[FREQUENCY_COEFFS_KEY]
    orig_num_rows = argument_dict[ORIG_NUM_ROWS_KEY]
    orig_num_columns = argument_dict[ORIG_NUM_COLUMNS_KEY]

    def mse_function(target_tensor, prediction_tensor):
        """Computes MSE on Fourier weights (real & imaginary part).

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: mse: Mean squared error (scalar).
        """

        target_coeff_tensor, predicted_coeff_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[2:]

        return _get_frequency_domain_mse(
            target_coeff_tensor=target_coeff_tensor,
            predicted_coeff_tensor=predicted_coeff_tensor,
            include_real=True, include_imaginary=True
        )

    if function_name is not None:
        mse_function.__name__ = function_name

    return mse_function
