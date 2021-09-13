"""Scale-separation-based metrics (with wavelet decomp) for Keras models."""

import numpy
import tensorflow
from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from ml4convection.machine_learning import fourier_metrics
from wavetf import WaveTFFactory

MIN_RESOLUTION_DEG = 0.01
MAX_RESOLUTION_DEG = 10.
GRID_SPACING_DEG = 0.0125

START_PADDING_PX = 26
END_PADDING_PX = 25
NUM_ROWS_BEFORE_PADDING = 205
NUM_ROWS_AFTER_PADDING = 256

FSS_NAME = 'fss'
BRIER_SCORE_NAME = 'brier'
CSI_NAME = 'csi'
FREQUENCY_BIAS_NAME = 'bias'
IOU_NAME = 'iou'
ALL_CLASS_IOU_NAME = 'all-class-iou'
DICE_COEFF_NAME = 'dice'
# REAL_FREQ_MSE_NAME = 'fmser'
# IMAGINARY_FREQ_MSE_NAME = 'fmsei'
# FREQ_MSE_NAME = 'fmse'

VALID_SCORE_NAMES = [
    FSS_NAME, BRIER_SCORE_NAME, CSI_NAME, FREQUENCY_BIAS_NAME,
    IOU_NAME, ALL_CLASS_IOU_NAME, DICE_COEFF_NAME
]


def _check_input_args(min_resolution_deg, max_resolution_deg, mask_matrix,
                      function_name):
    """Error-checks input arguments for any metric.

    m = number of rows in original grid = 205
    n = number of columns in original grid = 205
    M = number of rows in padded grid = 256
    N = number of columns in padded grid = 256
    K = number of levels in wavelet decomposition = log_2(M) = 8

    :param min_resolution_deg: Minimum resolution (degrees lat/long) allowed
        through band-pass filter.
    :param max_resolution_deg: Max resolution (degrees lat/long) allowed through
        band-pass filter.
    :param mask_matrix: m-by-n numpy array of Boolean flags.  Grid points marked
        "False" are masked out and not used to compute the metric.
    :param function_name: Function name (string).  May be None.
    :return: mask_matrix: Same as input but with dimensions of 1 x M x N.
    :return: keep_mean_flags: length-K numpy array of Boolean flags, indicating
        at which levels the mean signal will be kept.
    :return: keep_detail_flags: Same but for details.
    """

    # TODO(thunderhoser): Needs unit test.

    error_checking.assert_is_geq(min_resolution_deg, 0.)
    error_checking.assert_is_greater(max_resolution_deg, min_resolution_deg)

    if min_resolution_deg <= MIN_RESOLUTION_DEG:
        min_resolution_deg = 0.
    if max_resolution_deg >= MAX_RESOLUTION_DEG:
        max_resolution_deg = numpy.inf

    error_checking.assert_is_boolean_numpy_array(mask_matrix)
    expected_dim = numpy.full(2, NUM_ROWS_BEFORE_PADDING, dtype=int)
    error_checking.assert_is_numpy_array(
        mask_matrix, exact_dimensions=expected_dim
    )

    num_levels = int(numpy.round(
        numpy.log2(NUM_ROWS_AFTER_PADDING)
    ))

    level_indices = numpy.linspace(0, num_levels - 1, num=num_levels, dtype=int)
    detail_resolution_by_level_deg = GRID_SPACING_DEG * (2 ** level_indices)
    mean_resolution_by_level_deg = GRID_SPACING_DEG * (2 ** (level_indices + 1))

    keep_mean_flags = numpy.full(num_levels, 1, dtype=bool)
    keep_detail_flags = numpy.full(num_levels, 1, dtype=bool)

    this_index = numpy.searchsorted(
        a=mean_resolution_by_level_deg, v=max_resolution_deg, side='right'
    )
    if this_index < num_levels:
        keep_mean_flags[this_index:] = False

    this_index = -1 + numpy.searchsorted(
        a=detail_resolution_by_level_deg, v=min_resolution_deg, side='left'
    )
    if this_index > 0:
        keep_detail_flags[:(this_index + 1)] = False

    padding_arg = (
        (START_PADDING_PX, END_PADDING_PX),
        (START_PADDING_PX, END_PADDING_PX)
    )
    mask_matrix = numpy.pad(
        mask_matrix, pad_width=padding_arg, mode='constant', constant_values=0.
    )
    mask_matrix = numpy.expand_dims(mask_matrix, axis=0).astype(numpy.float32)

    if function_name is not None:
        error_checking.assert_is_string(function_name)

    return mask_matrix, keep_mean_flags, keep_detail_flags


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


def _do_forward_transform(input_tensor, num_levels):
    """Does forward multi-level wavelet transform.

    K = number of levels in wavelet transform

    :param input_tensor: Input tensor, to which wavelet transform will be
        applied.
    :param num_levels: K in the above discussion.
    :return: coeff_tensor_by_level: length-K list of tensors, each containing
        coefficients in format returned by WaveTF library.
    """

    dwt_object = WaveTFFactory().build('haar', dim=2)
    coeff_tensor_by_level = [None] * num_levels

    for k in range(num_levels):
        this_num_rows = int(numpy.round(
            float(NUM_ROWS_AFTER_PADDING) / (2 ** k)
        ))

        if k == 0:
            coeff_tensor_by_level[k] = dwt_object.call(
                input_tensor, cn=1, ox=this_num_rows, oy=this_num_rows
            )
        else:
            coeff_tensor_by_level[k] = dwt_object.call(
                coeff_tensor_by_level[k - 1][..., :1],
                cn=1, ox=this_num_rows, oy=this_num_rows
            )

    return coeff_tensor_by_level


def _filter_wavelet_coeffs(coeff_tensor_by_level, keep_mean_flags,
                           keep_detail_flags):
    """Filters wavelet coeffs (zeroes out coeffs at undesired wavelengths).

    K = number of levels in wavelet transform

    :param coeff_tensor_by_level: See documentation for `_do_forward_transform`.
    :param keep_mean_flags: length-K numpy array of Boolean flags, indicating
        whether or not to keep mean at each level.
    :param keep_detail_flags: length-K numpy array of Boolean flags, indicating
        whether or not to keep details at each level.
    :return: coeff_tensor_by_level: Same as input but maybe with more zeros.
    """

    num_levels = len(coeff_tensor_by_level)
    inverse_dwt_object = WaveTFFactory().build('haar', dim=2, inverse=True)

    these_indices = numpy.where(keep_mean_flags)[0]
    if len(these_indices) == 0:
        max_index = num_levels
    else:
        max_index = these_indices[0]

    these_indices = numpy.where(keep_detail_flags)[0]
    if len(these_indices) == 0:
        min_index = -1
    else:
        min_index = these_indices[-1]

    if max_index < num_levels:
        coeff_tensor_by_level[max_index] = tensorflow.concat([
            tensorflow.zeros_like(coeff_tensor_by_level[max_index][..., :1]),
            coeff_tensor_by_level[max_index][..., 1:]
        ], axis=-1)

        k = max_index + 0

        while k > 0 and k > min_index:
            this_num_rows = int(numpy.round(
                float(NUM_ROWS_AFTER_PADDING) / (2 ** (k + 1))
            ))

            this_coeff_tensor = inverse_dwt_object.call(
                coeff_tensor_by_level[k],
                cn=4, nx=this_num_rows, ny=this_num_rows
            )

            coeff_tensor_by_level[k - 1] = tensorflow.concat([
                this_coeff_tensor,
                coeff_tensor_by_level[k - 1][..., 1:]
            ], axis=-1)

            k -= 1

    if min_index > 0:
        coeff_tensor_by_level[min_index] = tensorflow.concat([
            coeff_tensor_by_level[min_index][..., :1],
            tensorflow.zeros_like(coeff_tensor_by_level[min_index][..., 1:])
        ], axis=-1)

    k = min_index + 0

    while k > 0:
        this_num_rows = int(numpy.round(
            float(NUM_ROWS_AFTER_PADDING) / (2 ** (k + 1))
        ))

        this_coeff_tensor = inverse_dwt_object.call(
            coeff_tensor_by_level[k],
            cn=4, nx=this_num_rows, ny=this_num_rows
        )

        coeff_tensor_by_level[k - 1] = tensorflow.concat([
            this_coeff_tensor,
            coeff_tensor_by_level[k - 1][..., 1:]
        ], axis=-1)

        coeff_tensor_by_level[k - 1] = tensorflow.concat([
            coeff_tensor_by_level[k - 1][..., :1],
            tensorflow.zeros_like(coeff_tensor_by_level[k - 1][..., 1:])
        ], axis=-1)

        k -= 1

    return coeff_tensor_by_level


def _filter_fields(
        target_tensor, prediction_tensor, keep_mean_flags, keep_detail_flags):
    """Filters predicted and target fields via wavelet transform, then inverse.

    K = number of levels in wavelet transform

    :param target_tensor: Tensor of target (actual) values.
    :param prediction_tensor: Tensor of predicted values.
    :param keep_mean_flags: length-K numpy array of Boolean flags, indicating
        whether or not to keep mean at each level.
    :param keep_detail_flags: length-K numpy array of Boolean flags, indicating
        whether or not to keep details at each level.
    :return: target_tensor: Filtered version of input.
    :return: prediction_tensor: Filtered version of input.
    """

    padding_arg = (
        (START_PADDING_PX, END_PADDING_PX),
        (START_PADDING_PX, END_PADDING_PX)
    )

    target_tensor = K.spatial_2d_padding(
        target_tensor, padding=padding_arg, data_format='channels_last'
    )
    coeff_tensor_by_level = _do_forward_transform(
        input_tensor=target_tensor, num_levels=len(keep_mean_flags)
    )
    coeff_tensor_by_level = _filter_wavelet_coeffs(
        coeff_tensor_by_level=coeff_tensor_by_level,
        keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
    )

    inverse_dwt_object = WaveTFFactory().build('haar', dim=2, inverse=True)
    this_num_rows = int(numpy.round(
        float(NUM_ROWS_AFTER_PADDING) / 2
    ))

    target_tensor = inverse_dwt_object.call(
        coeff_tensor_by_level[0], cn=4, nx=this_num_rows, ny=this_num_rows
    )
    target_tensor = K.maximum(target_tensor, 0.)
    target_tensor = K.minimum(target_tensor, 1.)

    prediction_tensor = K.spatial_2d_padding(
        prediction_tensor, padding=padding_arg, data_format='channels_last'
    )
    coeff_tensor_by_level = _do_forward_transform(
        input_tensor=prediction_tensor, num_levels=len(keep_mean_flags)
    )
    coeff_tensor_by_level = _filter_wavelet_coeffs(
        coeff_tensor_by_level=coeff_tensor_by_level,
        keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
    )

    prediction_tensor = inverse_dwt_object.call(
        coeff_tensor_by_level[0], cn=4, nx=this_num_rows, ny=this_num_rows
    )
    prediction_tensor = K.maximum(prediction_tensor, 0.)
    prediction_tensor = K.minimum(prediction_tensor, 1.)

    return target_tensor[..., 0], prediction_tensor[..., 0]


def brier_score(min_resolution_deg, max_resolution_deg, mask_matrix,
                use_as_loss_function=True, function_name=None):
    """Creates function to compute Brier score at a given scale.

    :param min_resolution_deg: See documentation for `_check_input_args`.
    :param max_resolution_deg: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
    :param function_name: Function name (string).
    :return: brier_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    mask_matrix, keep_mean_flags, keep_detail_flags = _check_input_args(
        min_resolution_deg=min_resolution_deg,
        max_resolution_deg=max_resolution_deg,
        mask_matrix=mask_matrix, function_name=function_name
    )

    def brier_function(target_tensor, prediction_tensor):
        """Computes Brier score at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: brier_score: Brier score (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
        )

        return fourier_metrics.get_brier_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

    if function_name is not None:
        brier_function.__name__ = function_name

    return brier_function


def csi(min_resolution_deg, max_resolution_deg, mask_matrix,
        use_as_loss_function, function_name=None):
    """Creates function to compute critical success index (CSI) at given scale.

    :param min_resolution_deg: See documentation for `_check_input_args`.
    :param max_resolution_deg: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
    :param function_name: Function name (string).
    :return: csi_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    mask_matrix, keep_mean_flags, keep_detail_flags = _check_input_args(
        min_resolution_deg=min_resolution_deg,
        max_resolution_deg=max_resolution_deg,
        mask_matrix=mask_matrix, function_name=function_name
    )

    def csi_function(target_tensor, prediction_tensor):
        """Computes CSI at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: csi: CSI (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
        )

        csi_value = fourier_metrics.get_csi(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - csi_value

        return csi_value

    if function_name is not None:
        csi_function.__name__ = function_name

    return csi_function


def frequency_bias(min_resolution_deg, max_resolution_deg, mask_matrix,
                   use_as_loss_function=False, function_name=None):
    """Creates function to compute frequency bias at a given scale.

    :param min_resolution_deg: See documentation for `_check_input_args`.
    :param max_resolution_deg: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
    :param function_name: Function name (string).
    :return: bias_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    mask_matrix, keep_mean_flags, keep_detail_flags = _check_input_args(
        min_resolution_deg=min_resolution_deg,
        max_resolution_deg=max_resolution_deg,
        mask_matrix=mask_matrix, function_name=function_name
    )

    def bias_function(target_tensor, prediction_tensor):
        """Computes frequency bias at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: frequency_bias: Frequency bias (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
        )

        return fourier_metrics.get_frequency_bias(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

    if function_name is not None:
        bias_function.__name__ = function_name

    return bias_function


def pixelwise_fss(min_resolution_deg, max_resolution_deg, mask_matrix,
                  use_as_loss_function, function_name=None):
    """Creates function to compute pixelwise FSS at a given scale.

    FSS = fractions skill score

    :param min_resolution_deg: See documentation for `_check_input_args`.
    :param max_resolution_deg: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
    :param function_name: Function name (string).
    :return: pixelwise_fss_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    mask_matrix, keep_mean_flags, keep_detail_flags = _check_input_args(
        min_resolution_deg=min_resolution_deg,
        max_resolution_deg=max_resolution_deg,
        mask_matrix=mask_matrix, function_name=function_name
    )

    def pixelwise_fss_function(target_tensor, prediction_tensor):
        """Computes pixelwise FSS at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: pixelwise_fss: Pixelwise FSS (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
        )

        fss_value = fourier_metrics.get_pixelwise_fss(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - fss_value

        return fss_value

    if function_name is not None:
        pixelwise_fss_function.__name__ = function_name

    return pixelwise_fss_function


def iou(min_resolution_deg, max_resolution_deg, mask_matrix,
        use_as_loss_function, function_name=None):
    """Creates function to compute intersctn over union (IOU) at a given scale.

    :param min_resolution_deg: See documentation for `_check_input_args`.
    :param max_resolution_deg: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
    :param function_name: Function name (string).
    :return: iou_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    mask_matrix, keep_mean_flags, keep_detail_flags = _check_input_args(
        min_resolution_deg=min_resolution_deg,
        max_resolution_deg=max_resolution_deg,
        mask_matrix=mask_matrix, function_name=function_name
    )

    def iou_function(target_tensor, prediction_tensor):
        """Computes IOU at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: iou: IOU (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
        )

        iou_value = fourier_metrics.get_iou(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        iou_function.__name__ = function_name

    return iou_function


def all_class_iou(min_resolution_deg, max_resolution_deg, mask_matrix,
                  use_as_loss_function, function_name=None):
    """Creates function to compute all-class IOU at a given scale.

    :param min_resolution_deg: See documentation for `_check_input_args`.
    :param max_resolution_deg: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
    :param function_name: Function name (string).
    :return: all_class_iou_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    mask_matrix, keep_mean_flags, keep_detail_flags = _check_input_args(
        min_resolution_deg=min_resolution_deg,
        max_resolution_deg=max_resolution_deg,
        mask_matrix=mask_matrix, function_name=function_name
    )

    def all_class_iou_function(target_tensor, prediction_tensor):
        """Computes all-class IOU at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: all_class_iou: All-class IOU (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
        )

        iou_value = fourier_metrics.get_all_class_iou(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - iou_value

        return iou_value

    if function_name is not None:
        all_class_iou_function.__name__ = function_name

    return all_class_iou_function


def dice_coeff(min_resolution_deg, max_resolution_deg, mask_matrix,
               use_as_loss_function, function_name=None):
    """Creates function to compute Dice coefficient at a given scale.

    :param min_resolution_deg: See documentation for `_check_input_args`.
    :param max_resolution_deg: Same.
    :param mask_matrix: Same.
    :param use_as_loss_function: Leave this alone.
    :param function_name: Function name (string).
    :return: dice_function: Function (defined below).
    """

    error_checking.assert_is_boolean(use_as_loss_function)

    mask_matrix, keep_mean_flags, keep_detail_flags = _check_input_args(
        min_resolution_deg=min_resolution_deg,
        max_resolution_deg=max_resolution_deg,
        mask_matrix=mask_matrix, function_name=function_name
    )

    def dice_function(target_tensor, prediction_tensor):
        """Computes Dice coefficient at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: dice_coeff: Dice coefficient (scalar).
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            keep_mean_flags=keep_mean_flags, keep_detail_flags=keep_detail_flags
        )

        dice_coeff_value = fourier_metrics.get_dice_coeff(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

        if use_as_loss_function:
            return 1. - dice_coeff_value

        return dice_coeff_value

    if function_name is not None:
        dice_function.__name__ = function_name

    return dice_function
