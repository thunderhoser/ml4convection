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

    return target_tensor, prediction_tensor


def mean_squared_error(
        spatial_coeff_matrix, frequency_coeff_matrix, mask_matrix,
        function_name=None):
    """Creates function to compute mean squared error at a given scale.

    :param spatial_coeff_matrix: See doc for `_check_input_args`.
    :param frequency_coeff_matrix: Same.
    :param mask_matrix: Same.
    :param function_name: Same.
    :return: mse_function: Function (defined below).
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

    def mse_function(target_tensor, prediction_tensor):
        """Computes mean squared error (MSE) at a given scale.

        :param target_tensor: Tensor of target (actual) values.
        :param prediction_tensor: Tensor of predicted values.
        :return: mse_value: MSE.
        """

        target_tensor, prediction_tensor = _filter_fields(
            target_tensor=target_tensor, prediction_tensor=prediction_tensor,
            spatial_coeff_matrix=spatial_coeff_matrix,
            frequency_coeff_matrix=frequency_coeff_matrix,
            orig_num_rows=orig_num_rows, orig_num_columns=orig_num_columns
        )

        squared_error_tensor = K.sum(
            (target_tensor - prediction_tensor) ** 2,
            axis=(1, 2)
        )

        mask_tensor = mask_matrix * K.ones_like(prediction_tensor)
        num_pixels_tensor = K.sum(mask_tensor, axis=(1, 2))

        return K.mean(squared_error_tensor / num_pixels_tensor)

    if function_name is not None:
        mse_function.__name__ = function_name

    return mse_function
