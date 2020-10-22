"""Stand-alone Keras operations."""

import os
import sys
import numpy
import tensorflow.python.keras.backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def do_2d_pooling(feature_matrix, do_max_pooling, window_size_px=2,
                  stride_length_px=None, pad_edges=False):
    """Pools 2-D feature maps.

    E = number of examples
    M = number of rows before pooling
    N = number of columns before pooling
    C = number of channels
    m = number of rows after pooling
    n = number of columns after pooling

    :param feature_matrix: E-by-M-by-N-by-C numpy array of feature values.
    :param do_max_pooling: Boolean flag.  If True, will do max-pooling.  If
        False, will do average-pooling.
    :param window_size_px: Window size (pixels).  The pooling window will be
        K rows x K columns, where K = `window_size_px`.
    :param stride_length_px: Stride length (pixels).  The pooling window will
        move k rows or columns with each stride, where k = `stride_length_px`.
    :param pad_edges: See doc for `do_2d_convolution`.
    :return: feature_matrix: E-by-m-by-n-by-C numpy array of new feature values.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=4)
    error_checking.assert_is_boolean(do_max_pooling)
    error_checking.assert_is_boolean(pad_edges)
    error_checking.assert_is_integer(window_size_px)
    error_checking.assert_is_geq(window_size_px, 2)

    if stride_length_px is None:
        stride_length_px = window_size_px

    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 1)

    feature_tensor = K.pool2d(
        x=K.variable(feature_matrix),
        pool_mode='max' if do_max_pooling else 'avg',
        pool_size=(window_size_px, window_size_px),
        strides=(stride_length_px, stride_length_px),
        padding='same' if pad_edges else 'valid', data_format='channels_last'
    )

    return feature_tensor.numpy()


def do_1d_pooling(feature_matrix, do_max_pooling, window_size_px=2):
    """Pools 1-D feature maps.

    E = number of examples
    P = number of pixels before pooling
    C = number of channels
    p = number of pixels after pooling

    :param feature_matrix: E-by-P-by-C numpy array of feature values.
    :param do_max_pooling: See doc for `do_2d_pooling`.
    :param window_size_px: Same.
    :return: feature_matrix: E-by-p-by-C numpy array of new feature values.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=3)
    error_checking.assert_is_integer(window_size_px)
    error_checking.assert_is_geq(window_size_px, 2)

    feature_matrix = numpy.expand_dims(feature_matrix, axis=-2)
    feature_matrix = numpy.repeat(
        feature_matrix, repeats=window_size_px, axis=-2
    )

    feature_matrix = do_2d_pooling(
        feature_matrix=feature_matrix, do_max_pooling=do_max_pooling,
        window_size_px=window_size_px
    )

    return feature_matrix[..., 0, :]


def do_2d_convolution(
        feature_matrix, kernel_matrix, pad_edges=False, stride_length_px=1):
    """Convolves 2-D feature maps.

    E = number of examples
    M = number of input rows
    N = number of input columns
    C = number of input channels
    m = number of rows in kernel
    n = number of columns in kernel
    c = number of output channels

    :param feature_matrix: E-by-M-by-N-by-C numpy array of feature values.
    :param kernel_matrix: m-by-n-by-C-by-c numpy array of kernel weights (filter
        weights).
    :param pad_edges: Boolean flag.  If True, edges will be padded so that the
        output feature matrix has the same dimensions as the input matrix.  If
        False, edges will not be padded, so the output feature matrix will be
        smaller.
    :param stride_length_px: Stride length in pixels.  The kernel (filter) will
        move this far at each step.
    :return: feature_matrix: E-by-?-by-?-by-c numpy array of feature values.
        The spatial dimensions will be determined by filter size and
        edge-padding.
    """

    error_checking.assert_is_numpy_array_without_nan(feature_matrix)
    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=4)
    error_checking.assert_is_numpy_array_without_nan(kernel_matrix)
    error_checking.assert_is_numpy_array(kernel_matrix, num_dimensions=4)
    error_checking.assert_is_boolean(pad_edges)
    error_checking.assert_is_integer(stride_length_px)
    error_checking.assert_is_geq(stride_length_px, 1)

    feature_tensor = K.conv2d(
        x=K.variable(feature_matrix),
        kernel=K.variable(kernel_matrix),
        strides=(stride_length_px, stride_length_px),
        padding='same' if pad_edges else 'valid',
        data_format='channels_last'
    )

    return feature_tensor.numpy()
