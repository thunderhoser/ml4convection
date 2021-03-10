"""Helper methods for Fourier transforms in 2-D space."""

import numpy
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6


def _get_spatial_resolutions(num_grid_rows, num_grid_columns,
                             grid_spacing_metres):
    """Computes spatial resolution for each Fourier coefficient.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    Matrices returned by this method correspond to matrices of Fourier
    coefficients returned by `numpy.fft.fft2`.  The x-coordinate increases with
    column index, and the y-coordinate increases with row index.

    :param num_grid_rows: M in the above discussion.
    :param num_grid_columns: N in the above discussion.
    :param grid_spacing_metres: Grid spacing (for which I use "resolution" as a
        synonym).
    :return: x_resolution_matrix_metres: M-by-N numpy array of resolutions in
        x-direction.
    :return: y_resolution_matrix_metres: Same but for y-direction.
    """

    # Check input args.
    error_checking.assert_is_integer(num_grid_rows)
    error_checking.assert_is_geq(num_grid_rows, 3)
    error_checking.assert_is_integer(num_grid_columns)
    error_checking.assert_is_geq(num_grid_columns, 3)
    error_checking.assert_is_greater(grid_spacing_metres, 0.)

    num_half_rows_float = float(num_grid_rows - 1) / 2
    num_half_rows = int(numpy.round(num_half_rows_float))
    assert numpy.isclose(num_half_rows, num_half_rows_float, atol=TOLERANCE)

    num_half_columns_float = float(num_grid_columns - 1) / 2
    num_half_columns = int(numpy.round(num_half_columns_float))
    assert numpy.isclose(
        num_half_columns, num_half_columns_float, atol=TOLERANCE
    )

    # Find resolutions in x-direction.
    unique_x_wavenumbers = numpy.linspace(
        0, num_half_columns, num=num_half_columns + 1, dtype=int
    )
    x_wavenumbers = numpy.concatenate((
        unique_x_wavenumbers, unique_x_wavenumbers[1:][::-1]
    ))
    x_wavenumber_matrix = numpy.expand_dims(x_wavenumbers, axis=0)
    x_wavenumber_matrix = numpy.repeat(
        x_wavenumber_matrix, axis=0, repeats=num_grid_rows
    )

    x_grid_length_metres = grid_spacing_metres * (num_grid_columns - 1)
    x_resolution_matrix_metres = (
        0.5 * x_grid_length_metres / x_wavenumber_matrix
    )

    # Find resolutions in y-direction.
    unique_y_wavenumbers = numpy.linspace(
        0, num_half_rows, num=num_half_rows + 1, dtype=int
    )
    y_wavenumbers = numpy.concatenate((
        unique_y_wavenumbers, unique_y_wavenumbers[1:][::-1]
    ))
    y_wavenumber_matrix = numpy.expand_dims(y_wavenumbers, axis=1)
    y_wavenumber_matrix = numpy.repeat(
        y_wavenumber_matrix, axis=1, repeats=num_grid_columns
    )

    y_grid_length_metres = grid_spacing_metres * (num_grid_rows - 1)
    y_resolution_matrix_metres = (
        0.5 * y_grid_length_metres / y_wavenumber_matrix
    )

    return x_resolution_matrix_metres, y_resolution_matrix_metres


def apply_rectangular_filter(
        coefficient_matrix, grid_spacing_metres, min_resolution_metres,
        max_resolution_metres):
    """Applies rectangular band-pass filter to Fourier coefficients.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param coefficient_matrix: M-by-N numpy array of coefficients in format
        returned by `numpy.fft.fft2`.
    :param grid_spacing_metres: Grid spacing (resolution).
    :param min_resolution_metres: Minimum resolution to preserve.
    :param max_resolution_metres: Max resolution to preserve.
    :return: coefficient_matrix: Same as input but maybe with some coefficients
        zeroed out.
    """

    # Check input args.
    error_checking.assert_is_geq(min_resolution_metres, 0.)
    error_checking.assert_is_greater(
        max_resolution_metres, min_resolution_metres
    )

    error_checking.assert_is_numpy_array(coefficient_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array_without_nan(
        numpy.real(coefficient_matrix)
    )
    error_checking.assert_is_numpy_array_without_nan(
        numpy.imag(coefficient_matrix)
    )

    # Do actual stuff.
    x_resolution_matrix_metres, y_resolution_matrix_metres = (
        _get_spatial_resolutions(
            num_grid_rows=coefficient_matrix.shape[0],
            num_grid_columns=coefficient_matrix.shape[1],
            grid_spacing_metres=grid_spacing_metres
        )
    )

    resolution_matrix_metres = numpy.sqrt(
        x_resolution_matrix_metres ** 2 + y_resolution_matrix_metres ** 2
    )

    coefficient_matrix[resolution_matrix_metres > max_resolution_metres] = 0.
    coefficient_matrix[resolution_matrix_metres < min_resolution_metres] = 0.
    return coefficient_matrix


def apply_butterworth_filter(
        coefficient_matrix, filter_order, grid_spacing_metres,
        min_resolution_metres, max_resolution_metres):
    """Applies Butterworth band-pass filter to Fourier coefficients.

    :param coefficient_matrix: See doc for `apply_rectangular_filter`.
    :param filter_order: Order of Butterworth filter (same as input arg `N` for
        `scipy.signal.butter`).
    :param grid_spacing_metres: See doc for `apply_rectangular_filter`.
    :param min_resolution_metres: Same.
    :param max_resolution_metres: Same.
    :return: coefficient_matrix: Same as input but after filtering.
    """

    # Check input args.
    error_checking.assert_is_geq(filter_order, 1.)
    error_checking.assert_is_geq(min_resolution_metres, 0.)
    error_checking.assert_is_greater(
        max_resolution_metres, min_resolution_metres
    )

    error_checking.assert_is_numpy_array(coefficient_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array_without_nan(
        numpy.real(coefficient_matrix)
    )
    error_checking.assert_is_numpy_array_without_nan(
        numpy.imag(coefficient_matrix)
    )

    # Determine horizontal, vertical, and total wavenumber for each Fourier
    # coefficient.
    x_resolution_matrix_metres, y_resolution_matrix_metres = (
        _get_spatial_resolutions(
            num_grid_rows=coefficient_matrix.shape[0],
            num_grid_columns=coefficient_matrix.shape[1],
            grid_spacing_metres=grid_spacing_metres
        )
    )

    x_wavenumber_matrix_metres01 = (2 * x_resolution_matrix_metres) ** -1
    y_wavenumber_matrix_metres01 = (2 * y_resolution_matrix_metres) ** -1
    wavenumber_matrix_metres01 = numpy.sqrt(
        x_wavenumber_matrix_metres01 ** 2 + y_wavenumber_matrix_metres01 ** 2
    )

    # High-pass part.
    if not numpy.isinf(max_resolution_metres):
        min_wavenumber_metres01 = (2 * max_resolution_metres) ** -1
        ratio_matrix = wavenumber_matrix_metres01 / min_wavenumber_metres01
        gain_matrix = 1 - (1 + ratio_matrix ** (2 * filter_order)) ** -1
        coefficient_matrix = coefficient_matrix * gain_matrix

    # Low-pass part.
    if min_resolution_metres > grid_spacing_metres:
        max_wavenumber_metres01 = (2 * min_resolution_metres) ** -1
        ratio_matrix = wavenumber_matrix_metres01 / max_wavenumber_metres01
        gain_matrix = (1 + ratio_matrix ** (2 * filter_order)) ** -1
        coefficient_matrix = coefficient_matrix * gain_matrix

    return coefficient_matrix


def taper_spatial_data(spatial_data_matrix):
    """Tapers spatial data by putting zeros along the edge.

    M = number of rows in grid
    N = number of columns in grid

    :param spatial_data_matrix: M-by-N numpy array of real numbers.
    :return: spatial_data_matrix: Same but after tapering.
    """

    error_checking.assert_is_numpy_array_without_nan(spatial_data_matrix)
    error_checking.assert_is_numpy_array(spatial_data_matrix, num_dimensions=2)

    num_rows = spatial_data_matrix.shape[0]
    num_columns = spatial_data_matrix.shape[1]

    padding_arg = (
        (num_rows, num_rows),
        (num_columns, num_columns)
    )

    spatial_data_matrix = numpy.pad(
        spatial_data_matrix, pad_width=padding_arg, mode='constant',
        constant_values=0.
    )

    return spatial_data_matrix


def untaper_spatial_data(spatial_data_matrix):
    """Removes zeros along the edge of spatial data.

    This method is the inverse of `taper_spatial_data`.

    :param spatial_data_matrix: See output doc for `taper_spatial_data`.
    :return: spatial_data_matrix: See input doc for `taper_spatial_data`.
    """

    error_checking.assert_is_numpy_array_without_nan(spatial_data_matrix)
    error_checking.assert_is_numpy_array(spatial_data_matrix, num_dimensions=2)

    num_rows = spatial_data_matrix.shape[0]
    num_columns = spatial_data_matrix.shape[1]

    num_third_rows_float = float(num_rows) / 3
    num_third_rows = int(numpy.round(num_third_rows_float))
    assert numpy.isclose(num_third_rows, num_third_rows_float, atol=TOLERANCE)

    num_third_columns_float = float(num_columns) / 3
    num_third_columns = int(numpy.round(num_third_columns_float))
    assert numpy.isclose(
        num_third_columns, num_third_columns_float, atol=TOLERANCE
    )

    return spatial_data_matrix[
        num_third_rows:-num_third_rows,
        num_third_columns:-num_third_columns
    ]


def apply_blackman_window(spatial_data_matrix):
    """Applies Blackman window to 2-D spatial data.

    M = number of rows in grid
    N = number of columns in grid

    :param spatial_data_matrix: M-by-N numpy array of real numbers.
    :return: spatial_data_matrix: Same but after smoothing via Blackman window.
    """

    # TODO(thunderhoser): Implement non-radial version (separation of coords?)?

    error_checking.assert_is_numpy_array_without_nan(spatial_data_matrix)
    error_checking.assert_is_numpy_array(spatial_data_matrix, num_dimensions=2)

    num_rows = spatial_data_matrix.shape[0]
    num_columns = spatial_data_matrix.shape[1]
    num_half_rows = float(num_rows - 1) / 2
    num_half_columns = float(num_columns - 1) / 2

    row_indices = numpy.linspace(0, num_rows - 1, num=num_rows, dtype=float)
    column_indices = numpy.linspace(
        0, num_columns - 1, num=num_columns, dtype=float
    )

    y_distances = numpy.absolute(row_indices - num_half_rows)
    x_distances = numpy.absolute(column_indices - num_half_columns)
    x_distance_matrix, y_distance_matrix = numpy.meshgrid(
        x_distances, y_distances
    )

    distance_matrix = numpy.sqrt(
        x_distance_matrix ** 2 + y_distance_matrix ** 2
    )
    max_distance = numpy.maximum(
        numpy.max(x_distance_matrix),
        numpy.max(y_distance_matrix)
    )
    fractional_distance_matrix = distance_matrix / max_distance
    fractional_distance_matrix = numpy.minimum(fractional_distance_matrix, 1.)

    weight_matrix = (
        0.42 -
        0.5 * numpy.cos(numpy.pi * (1 + fractional_distance_matrix)) +
        0.08 * numpy.cos(2 * numpy.pi * (1 + fractional_distance_matrix))
    )

    return spatial_data_matrix * weight_matrix
