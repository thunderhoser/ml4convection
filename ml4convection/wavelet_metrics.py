"""Scale-separation-based metrics (with wavelet decomp) for Keras models."""

import os
import sys
import numpy
import tensorflow
from keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
from _wavetf import WaveTFFactory

START_PADDING_PX = 26
END_PADDING_PX = 25

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
    print(input_tensor.shape)
    print(tensorflow.shape(input_tensor))
    print('\n\n\n*********\n\n\n')

    for k in range(num_levels):
        if k == 0:
            coeff_tensor_by_level[k] = dwt_object.call(input_tensor)
        else:
            coeff_tensor_by_level[k] = dwt_object.call(
                coeff_tensor_by_level[k - 1][..., :1]
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
            coeff_tensor_by_level[k - 1] = tensorflow.concat([
                inverse_dwt_object.call(coeff_tensor_by_level[k]),
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
        coeff_tensor_by_level[k - 1] = tensorflow.concat([
            inverse_dwt_object.call(coeff_tensor_by_level[k]),
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
    target_tensor = inverse_dwt_object.call(coeff_tensor_by_level[0])
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

    prediction_tensor = inverse_dwt_object.call(coeff_tensor_by_level[0])
    prediction_tensor = K.maximum(prediction_tensor, 0.)
    prediction_tensor = K.minimum(prediction_tensor, 1.)

    return target_tensor[..., 0], prediction_tensor[..., 0]


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


def brier_score(keep_mean_flags, keep_detail_flags, mask_matrix,
                use_as_loss_function=True, function_name=None):
    """Creates function to compute Brier score at a given scale.

    :param keep_mean_flags: FOO.
    :param keep_detail_flags: FOO.
    :param mask_matrix: FOO.
    :param use_as_loss_function: Leave this alone.
    :param function_name: FOO.
    :return: brier_function: Function (defined below).
    """

    # TODO(thunderhoser): Fix doc.

    padding_arg = (
        (START_PADDING_PX, END_PADDING_PX),
        (START_PADDING_PX, END_PADDING_PX)
    )
    mask_matrix = numpy.pad(
        mask_matrix, pad_width=padding_arg, mode='constant', constant_values=0.
    )
    mask_matrix = numpy.expand_dims(mask_matrix, axis=0).astype(numpy.float32)

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

        return _get_brier_score(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            mask_matrix=mask_matrix
        )

    if function_name is not None:
        brier_function.__name__ = function_name

    return brier_function
