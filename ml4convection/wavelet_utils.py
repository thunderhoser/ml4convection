"""Helper methods for wavelet transforms in 2-D space."""

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
from _wavetf import WaveTFFactory


def taper_spatial_data(spatial_data_matrix):
    """Tapers spatial data by putting zeros along the edge.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid

    :param spatial_data_matrix: E-by-M-by-N numpy array of real numbers.
    :return: spatial_data_matrix: Same but after tapering.
    :return: numpy_pad_width: Argument `pad_width` used for `numpy.pad`.
    """

    error_checking.assert_is_numpy_array_without_nan(spatial_data_matrix)
    error_checking.assert_is_numpy_array(spatial_data_matrix, num_dimensions=3)

    num_rowcols = max(spatial_data_matrix.shape[1:])
    num_transform_levels = int(numpy.ceil(
        numpy.log2(num_rowcols)
    ))
    num_rowcols_needed = int(numpy.round(
        2 ** num_transform_levels
    ))

    num_rows = spatial_data_matrix.shape[1]
    num_columns = spatial_data_matrix.shape[2]
    num_padding_rows = num_rowcols_needed - num_rows
    num_padding_columns = num_rowcols_needed - num_columns

    if numpy.mod(num_padding_rows, 2) == 0:
        num_start_rows = int(numpy.round(
            float(num_padding_rows) / 2
        ))
        num_end_rows = num_start_rows + 0
    else:
        num_start_rows = int(numpy.floor(
            float(num_padding_rows) / 2
        ))
        num_end_rows = num_start_rows + 1

    if numpy.mod(num_padding_columns, 2) == 0:
        num_start_columns = int(numpy.round(
            float(num_padding_columns) / 2
        ))
        num_end_columns = num_start_columns + 0
    else:
        num_start_columns = int(numpy.floor(
            float(num_padding_columns) / 2
        ))
        num_end_columns = num_start_columns + 1

    padding_arg = (
        (0, 0),
        (num_start_rows, num_end_rows),
        (num_start_columns, num_end_columns)
    )

    spatial_data_matrix = numpy.pad(
        spatial_data_matrix, pad_width=padding_arg, mode='constant',
        constant_values=0.
    )

    return spatial_data_matrix, padding_arg


def untaper_spatial_data(spatial_data_matrix, numpy_pad_width):
    """Removes zeros along the edge of spatial data.

    This method is the inverse of `taper_spatial_data`.

    :param spatial_data_matrix: See output doc for `taper_spatial_data`.
    :param numpy_pad_width: Same.
    :return: spatial_data_matrix: See input doc for `taper_spatial_data`.
    """

    error_checking.assert_is_numpy_array_without_nan(spatial_data_matrix)
    error_checking.assert_is_numpy_array(spatial_data_matrix, num_dimensions=3)

    expected_dim = numpy.array([
        spatial_data_matrix.shape[0],
        spatial_data_matrix.shape[1], spatial_data_matrix.shape[1]
    ], dtype=int)

    error_checking.assert_is_numpy_array(
        spatial_data_matrix, exact_dimensions=expected_dim
    )

    assert len(numpy_pad_width) == 3
    assert len(numpy_pad_width[1]) == 2
    assert len(numpy_pad_width[2]) == 2

    for x in numpy_pad_width[1]:
        error_checking.assert_is_integer(x)
        error_checking.assert_is_geq(x, 0)
    for x in numpy_pad_width[2]:
        error_checking.assert_is_integer(x)
        error_checking.assert_is_geq(x, 0)

    return spatial_data_matrix[
        :,
        numpy_pad_width[1][0]:-numpy_pad_width[1][-1],
        numpy_pad_width[2][0]:-numpy_pad_width[2][-1]
    ]


def do_forward_transform(spatial_data_matrix):
    """Does forward multi-level wavelet transform.

    E = number of examples
    N = number of rows in grid = number of columns in grid
    K = number of levels in wavelet transform = log_2(N)

    :param spatial_data_matrix: E-by-N-by-N numpy array of real numbers.
    :return: coeff_tensor_by_level: length-K list of tensors, each containing
        coefficients in format returned by WaveTF library.
    """

    error_checking.assert_is_numpy_array_without_nan(spatial_data_matrix)
    error_checking.assert_is_numpy_array(spatial_data_matrix, num_dimensions=3)

    expected_dim = numpy.array([
        spatial_data_matrix.shape[0],
        spatial_data_matrix.shape[1], spatial_data_matrix.shape[1]
    ], dtype=int)

    error_checking.assert_is_numpy_array(
        spatial_data_matrix, exact_dimensions=expected_dim
    )
    num_levels = int(numpy.round(
        numpy.log2(spatial_data_matrix.shape[1])
    ))

    spatial_data_tensor = tensorflow.constant(
        spatial_data_matrix, dtype=tensorflow.float64
    )
    spatial_data_tensor = tensorflow.expand_dims(spatial_data_tensor, axis=-1)

    dwt_object = WaveTFFactory().build('haar', dim=2)
    coeff_tensor_by_level = [None] * num_levels

    for k in range(num_levels):
        if k == 0:
            coeff_tensor_by_level[k] = dwt_object.call(spatial_data_tensor)
        else:
            coeff_tensor_by_level[k] = dwt_object.call(
                coeff_tensor_by_level[k - 1][..., :1]
            )

    return coeff_tensor_by_level


def coeff_tensors_to_numpy(coeff_tensor_by_level):
    """Converts wavelet coeffs from tensor format to numpy format.

    E = number of examples
    N = number of rows in grid = number of columns in grid
    K = number of levels in wavelet transform

    :param coeff_tensor_by_level: length-K list of tensors, each containing
        coefficients in format returned by WaveTF library.
    :return: mean_coeff_matrix: E-by-N-by-N numpy array of mean coefficients.
        Resolution decreases (wavelength increases) with row index and column
        index.
    :return: horizontal_coeff_matrix: E-by-N-by-N numpy array of
        horizontal-detail coefficients.  Resolution decreases (wavelength
        increases) with row index and column index.
    :return: vertical_coeff_matrix: Same but for vertical detail.
    :return: diagonal_coeff_matrix: Same but for diagonal detail.
    """

    num_examples = K.eval(coeff_tensor_by_level[0]).shape[0]
    num_grid_rows = 2 * K.eval(coeff_tensor_by_level[0]).shape[1]

    dimensions = (num_examples, num_grid_rows, num_grid_rows)
    mean_coeff_matrix = numpy.full(dimensions, numpy.nan)
    horizontal_coeff_matrix = numpy.full(dimensions, numpy.nan)
    vertical_coeff_matrix = numpy.full(dimensions, numpy.nan)
    diagonal_coeff_matrix = numpy.full(dimensions, numpy.nan)

    num_levels = len(coeff_tensor_by_level)
    i = 0

    for k in range(num_levels):
        this_coeff_matrix = K.eval(coeff_tensor_by_level[k])
        num_rows = this_coeff_matrix.shape[1]

        mean_coeff_matrix[:, i:(i + num_rows), i:(i + num_rows)] = (
            this_coeff_matrix[..., 0]
        )
        horizontal_coeff_matrix[:, i:(i + num_rows), i:(i + num_rows)] = (
            this_coeff_matrix[..., 2]
        )
        vertical_coeff_matrix[:, i:(i + num_rows), i:(i + num_rows)] = (
            this_coeff_matrix[..., 1]
        )
        diagonal_coeff_matrix[:, i:(i + num_rows), i:(i + num_rows)] = (
            this_coeff_matrix[..., 3]
        )

        i += num_rows

    return (
        mean_coeff_matrix, horizontal_coeff_matrix, vertical_coeff_matrix,
        diagonal_coeff_matrix
    )


def filter_coefficients(
        coeff_tensor_by_level, grid_spacing_metres, min_resolution_metres,
        max_resolution_metres, verbose=True):
    """Filters wavelet coeffs (zeroes out coeffs at undesired wavelengths).

    :param coeff_tensor_by_level: See documentation for `do_forward_transform`.
    :param grid_spacing_metres: Grid spacing (resolution).
    :param min_resolution_metres: Minimum resolution to preserve.
    :param max_resolution_metres: Max resolution to preserve.
    :param verbose: Boolean flag.
    :return: coeff_tensor_by_level: Same as input but maybe with more zeros.
    """

    error_checking.assert_is_greater(grid_spacing_metres, 0.)
    error_checking.assert_is_geq(min_resolution_metres, 0.)
    error_checking.assert_is_greater(
        max_resolution_metres, min_resolution_metres
    )
    error_checking.assert_is_boolean(verbose)

    inverse_dwt_object = WaveTFFactory().build('haar', dim=2, inverse=True)
    num_levels = len(coeff_tensor_by_level)

    level_indices = numpy.linspace(0, num_levels - 1, num=num_levels, dtype=int)
    detail_res_by_level_metres = grid_spacing_metres * (2 ** level_indices)
    mean_res_by_level_metres = grid_spacing_metres * (2 ** (level_indices + 1))

    max_index = numpy.searchsorted(
        a=mean_res_by_level_metres, v=max_resolution_metres, side='right'
    )
    min_index = -1 + numpy.searchsorted(
        a=detail_res_by_level_metres, v=min_resolution_metres, side='left'
    )

    if max_index < num_levels:
        k = num_levels - 1

        while k >= max_index:
            if verbose:
                print((
                    'Zeroing out low-frequency coefficients at level '
                    '{0:d} of {1:d} (resolutions = {2:.4f} and {3:.4f} deg)...'
                ).format(
                    k + 1, num_levels,
                    mean_res_by_level_metres[k],
                    detail_res_by_level_metres[k]
                ))

            coeff_tensor_by_level[k] = tensorflow.concat([
                tensorflow.zeros_like(coeff_tensor_by_level[k][..., :1]),
                coeff_tensor_by_level[k][..., 1:]
            ], axis=-1)

            k -= 1

        k = max_index + 0

        while k > 0 and k > min_index:
            if verbose:
                print((
                    'Reconstructing low-frequency coefficients at level '
                    '{0:d} of {1:d} (resolutions = {2:.4f} and {3:.4f} deg)...'
                ).format(
                    k, num_levels,
                    mean_res_by_level_metres[k - 1],
                    detail_res_by_level_metres[k - 1]
                ))

            coeff_tensor_by_level[k - 1] = tensorflow.concat([
                inverse_dwt_object.call(coeff_tensor_by_level[k]),
                coeff_tensor_by_level[k - 1][..., 1:]
            ], axis=-1)

            k -= 1

    if min_index > 0:
        if verbose:
            print((
                'Zeroing out high-frequency coefficients at level '
                '{0:d} of {1:d} (resolutions = {2:.4f} and {3:.4f} deg)...'
            ).format(
                min_index + 1, num_levels,
                mean_res_by_level_metres[min_index],
                detail_res_by_level_metres[min_index]
            ))

        coeff_tensor_by_level[min_index] = tensorflow.concat([
            coeff_tensor_by_level[min_index][..., :1],
            tensorflow.zeros_like(coeff_tensor_by_level[min_index][..., 1:])
        ], axis=-1)

    k = min_index + 0

    while k > 0:
        if verbose:
            print((
                'Reconstructing low-frequency coefficients at level '
                '{0:d} of {1:d} (resolutions = {2:.4f} and {3:.4f} deg)...'
            ).format(
                k, num_levels,
                mean_res_by_level_metres[k - 1],
                detail_res_by_level_metres[k - 1]
            ))

        coeff_tensor_by_level[k - 1] = tensorflow.concat([
            inverse_dwt_object.call(coeff_tensor_by_level[k]),
            coeff_tensor_by_level[k - 1][..., 1:]
        ], axis=-1)

        if verbose:
            print((
                'Zeroing out high-frequency coefficients at level '
                '{0:d} of {1:d} (resolutions = {2:.4f} and {3:.4f} deg)...'
            ).format(
                k, num_levels,
                mean_res_by_level_metres[k - 1],
                detail_res_by_level_metres[k - 1]
            ))

        coeff_tensor_by_level[k - 1] = tensorflow.concat([
            coeff_tensor_by_level[k - 1][..., :1],
            tensorflow.zeros_like(coeff_tensor_by_level[k - 1][..., 1:])
        ], axis=-1)

        k -= 1

    return coeff_tensor_by_level
