"""Scale-separation-based metrics (with Fourier decomp) for Keras models."""

import numpy
import tensorflow
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import error_checking
from ml4convection.utils import fourier_utils

SPATIAL_COEFFS_KEY = 'spatial_coeff_matrix'
FREQUENCY_COEFFS_KEY = 'frequency_coeff_matrix'
MASK_KEY = 'mask_matrix'
ORIG_NUM_ROWS_KEY = 'orig_num_rows'
ORIG_NUM_COLUMNS_KEY = 'orig_num_columns'

FSS_NAME = 'fss'
BRIER_SCORE_NAME = 'brier'
CROSS_ENTROPY_NAME = 'xentropy'
CSI_NAME = 'csi'
HEIDKE_SCORE_NAME = 'heidke'
GERRITY_SCORE_NAME = 'gerrity'
PEIRCE_SCORE_NAME = 'peirce'
FREQUENCY_BIAS_NAME = 'bias'
IOU_NAME = 'iou'
ALL_CLASS_IOU_NAME = 'all-class-iou'
DICE_COEFF_NAME = 'dice'
REAL_FREQ_MSE_NAME = 'fmser'
IMAGINARY_FREQ_MSE_NAME = 'fmsei'
FREQ_MSE_NAME = 'fmse'

VALID_SCORE_NAMES = [
    FSS_NAME, BRIER_SCORE_NAME, CROSS_ENTROPY_NAME, CSI_NAME,
    HEIDKE_SCORE_NAME, GERRITY_SCORE_NAME, PEIRCE_SCORE_NAME,
    FREQUENCY_BIAS_NAME, IOU_NAME, ALL_CLASS_IOU_NAME, DICE_COEFF_NAME,
    REAL_FREQ_MSE_NAME, IMAGINARY_FREQ_MSE_NAME, FREQ_MSE_NAME
]


def _log2(input_tensor):
    """Computes logarithm in base 2.

    :param input_tensor: Keras tensor.
    :return: logarithm_tensor: Keras tensor with the same shape as
        `input_tensor`.
    """

    return K.log(K.maximum(input_tensor, 1e-6)) / K.log(2.)


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


def get_brier_score(target_tensor, prediction_tensor, mask_matrix):
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


def get_cross_entropy(target_tensor, prediction_tensor, mask_matrix):
    """Computes cross-entropy.

    :param target_tensor: See doc for `get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: xentropy: Cross-entropy (scalar).
    """

    xentropy_tensor = K.sum(
        target_tensor * _log2(prediction_tensor) +
        (1. - target_tensor) * _log2(1. - prediction_tensor),
        axis=(1, 2)
    )

    mask_tensor = mask_matrix * K.ones_like(prediction_tensor)
    num_pixels_tensor = K.sum(mask_tensor, axis=(1, 2))

    return -K.mean(xentropy_tensor / num_pixels_tensor)


def get_csi(target_tensor, prediction_tensor, mask_matrix):
    """Computes critical success index.

    :param target_tensor: See doc for `get_brier_score`.
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


def get_peirce_score(target_tensor, prediction_tensor, mask_matrix):
    """Computes Peirce score.

    :param target_tensor: See doc for `get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: peirce_score: Peirce score (scalar).
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
    num_true_negatives = K.sum(
        mask_matrix * (1 - target_tensor) * (1 - prediction_tensor)
    )

    pod_value = (
        num_true_positives /
        (num_true_positives + num_false_negatives + K.epsilon())
    )
    pofd_value = (
        num_false_positives /
        (num_false_positives + num_true_negatives + K.epsilon())
    )

    return pod_value - pofd_value


def get_heidke_score(target_tensor, prediction_tensor, mask_matrix):
    """Computes Heidke score.

    :param target_tensor: See doc for `get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: heidke_score: Heidke score (scalar).
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
    num_true_negatives = K.sum(
        mask_matrix * (1 - target_tensor) * (1 - prediction_tensor)
    )

    random_num_correct = (
        (num_true_positives + num_false_positives) *
        (num_true_positives + num_false_negatives) +
        (num_false_negatives + num_true_negatives) *
        (num_false_positives + num_true_negatives)
    )
    num_examples = (
        num_true_positives + num_false_positives +
        num_false_negatives + num_true_negatives
    )
    random_num_correct = random_num_correct / num_examples

    numerator = num_true_positives + num_true_negatives - random_num_correct
    denominator = num_examples - random_num_correct + K.epsilon()

    return numerator / denominator


def get_gerrity_score(target_tensor, prediction_tensor, mask_matrix):
    """Computes Gerrity score.

    :param target_tensor: See doc for `get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: gerrity_score: Gerrity score (scalar).
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
    num_true_negatives = K.sum(
        mask_matrix * (1 - target_tensor) * (1 - prediction_tensor)
    )

    event_ratio = (
        (num_false_positives + num_true_negatives) /
        (num_true_positives + num_false_negatives + K.epsilon())
    )
    num_examples = (
        num_true_positives + num_false_positives +
        num_false_negatives + num_true_negatives
    )

    numerator = (
        num_true_positives * event_ratio
        + num_true_negatives * (1. / event_ratio)
        - num_false_positives - num_false_negatives
    )

    return numerator / num_examples


def get_frequency_bias(target_tensor, prediction_tensor, mask_matrix):
    """Computes frequency bias.

    :param target_tensor: See doc for `get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: frequency_bias: Frequency bias (scalar).
    """

    numerator = K.sum(mask_matrix * prediction_tensor)
    denominator = K.sum(mask_matrix * target_tensor) + K.epsilon()

    return numerator / denominator


def get_pixelwise_fss(target_tensor, prediction_tensor, mask_matrix):
    """Computes pixelwise fractions skill score (FSS).

    :param target_tensor: See doc for `get_brier_score`.
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


def get_iou(target_tensor, prediction_tensor, mask_matrix):
    """Computes intersection over union.

    :param target_tensor: See doc for `get_brier_score`.
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


def get_all_class_iou(target_tensor, prediction_tensor, mask_matrix):
    """Computes all-class intersection over union (IOU).

    :param target_tensor: See doc for `get_brier_score`.
    :param prediction_tensor: Same.
    :param mask_matrix: Same.
    :return: all_class_iou: All-class IOU.
    """

    masked_target_tensor = mask_matrix * target_tensor
    masked_prediction_tensor = mask_matrix * prediction_tensor
    positive_intersection_tensor = K.sum(
        masked_target_tensor * masked_prediction_tensor, axis=(1, 2)
    )
    positive_union_tensor = (
        K.sum(masked_target_tensor, axis=(1, 2)) +
        K.sum(masked_prediction_tensor, axis=(1, 2)) -
        positive_intersection_tensor
    )

    masked_target_tensor = mask_matrix * (1. - target_tensor)
    masked_prediction_tensor = mask_matrix * (1. - prediction_tensor)
    negative_intersection_tensor = K.sum(
        masked_target_tensor * masked_prediction_tensor, axis=(1, 2)
    )
    negative_union_tensor = (
        K.sum(masked_target_tensor, axis=(1, 2)) +
        K.sum(masked_prediction_tensor, axis=(1, 2)) -
        negative_intersection_tensor
    )

    positive_iou = K.mean(
        positive_intersection_tensor / (positive_union_tensor + K.epsilon())
    )
    negative_iou = K.mean(
        negative_intersection_tensor / (negative_union_tensor + K.epsilon())
    )
    return (positive_iou + negative_iou) / 2


def get_dice_coeff(target_tensor, prediction_tensor, mask_matrix):
    """Computes Dice coefficient.

    :param target_tensor: See doc for `get_brier_score`.
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


def brier_score(
        spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        use_as_loss_function=True, function_name=None):
    """Creates function to compute Brier score at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
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

        return get_brier_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

    if function_name is not None:
        brier_function.__name__ = function_name

    return brier_function


def cross_entropy(
        spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        use_as_loss_function=True, function_name=None):
    """Creates function to compute cross-entropy at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
    :param function_name: See doc for `_check_input_args`.
    :return: xentropy_function: Function (defined below).
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

    def xentropy_function(target_tensor, prediction_tensor):
        """Computes cross-entropy at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: xentropy: Cross-entropy (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        return get_cross_entropy(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

    if function_name is not None:
        xentropy_function.__name__ = function_name

    return xentropy_function


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

        csi_value = get_csi(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - csi_value

        return csi_value

    if function_name is not None:
        csi_function.__name__ = function_name

    return csi_function


def peirce_score(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
                 use_as_loss_function, function_name=None):
    """Creates function to compute Peirce score at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 minus
        Peirce score, to use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: peirce_function: Function (defined below).
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

    def peirce_function(target_tensor, prediction_tensor):
        """Computes Peirce score at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: peirce_value: Peirce score (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        peirce_value = get_peirce_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - peirce_value

        return peirce_value

    if function_name is not None:
        peirce_function.__name__ = function_name

    return peirce_function


def heidke_score(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
                 use_as_loss_function, function_name=None):
    """Creates function to compute Heidke score at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 minus
        Heidke score, to use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: heidke_function: Function (defined below).
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

    def heidke_function(target_tensor, prediction_tensor):
        """Computes Heidke score at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: heidke_value: Heidke score (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        heidke_value = get_heidke_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - heidke_value

        return heidke_value

    if function_name is not None:
        heidke_function.__name__ = function_name


    return heidke_function


def gerrity_score(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
                  use_as_loss_function, function_name=None):
    """Creates function to compute Gerrity score at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 minus
        Gerrity score, to use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: gerrity_function: Function (defined below).
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

    def gerrity_function(target_tensor, prediction_tensor):
        """Computes Gerrity score at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: gerrity_value: Gerrity score (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        gerrity_value = get_gerrity_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - gerrity_value

        return gerrity_value

    if function_name is not None:
        gerrity_function.__name__ = function_name

    return gerrity_function


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

        return get_frequency_bias(
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
    :param use_as_loss_function: Boolean flag.  If True, will return 1 - FSS, to
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

        fss_value = get_pixelwise_fss(
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
    :param use_as_loss_function: Boolean flag.  If True, will return 1 - IOU, to
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

        iou_value = get_iou(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        iou_function.__name__ = function_name

    return iou_function


def all_class_iou(spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
                  use_as_loss_function, function_name=None):
    """Creates function to compute all-class IOU at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 - IOU, to
        use as negatively oriented loss function.
    :param function_name: See doc for `_check_input_args`.
    :return: all_class_iou_function: Function (defined below).
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

    def all_class_iou_function(target_tensor, prediction_tensor):
        """Computes all-class intersection over union (IOU) at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: iou_value: All-class IOU (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )[:2]

        iou_value = get_all_class_iou(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        all_class_iou_function.__name__ = function_name

    return all_class_iou_function


def dice_coeff(
        spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        use_as_loss_function, function_name=None):
    """Creates function to compute Dice coefficient at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Boolean flag.  If True, will return 1 minus
        Dice coeff, to use as negatively oriented loss function.
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

        dice_value = get_dice_coeff(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - dice_value

        return dice_value

    if function_name is not None:
        dice_function.__name__ = function_name

    return dice_function


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
