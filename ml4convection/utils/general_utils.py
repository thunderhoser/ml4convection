"""General helper methods."""

import numpy
import skimage.measure
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import time_conversion

DATE_FORMAT = '%Y%m%d'
DAYS_TO_SECONDS = 86400


def get_previous_date(date_string):
    """Returns previous date.

    :param date_string: Date (format "yyyymmdd").
    :return: prev_date_string: Previous date (format "yyyymmdd").
    """

    unix_time_sec = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)
    return time_conversion.unix_sec_to_string(
        unix_time_sec - DAYS_TO_SECONDS, DATE_FORMAT
    )


def get_next_date(date_string):
    """Returns next date.

    :param date_string: Date (format "yyyymmdd").
    :return: next_date_string: Next date (format "yyyymmdd").
    """

    unix_time_sec = time_conversion.string_to_unix_sec(date_string, DATE_FORMAT)
    return time_conversion.unix_sec_to_string(
        unix_time_sec + DAYS_TO_SECONDS, DATE_FORMAT
    )


def create_mean_filter(half_num_rows, half_num_columns, num_channels):
    """Creates convolutional filter that computes mean.

    M = number of rows in filter
    N = number of columns in filter
    C = number of channels

    :param half_num_rows: Number of rows on either side of center.  This is
        (M - 1) / 2.
    :param half_num_columns: Number of columns on either side of center.  This
        is (N - 1) / 2.
    :param num_channels: Number of channels.
    :return: weight_matrix: M-by-N-by-C-by-C numpy array of filter weights.
    """

    error_checking.assert_is_integer(half_num_rows)
    error_checking.assert_is_geq(half_num_rows, 0)
    error_checking.assert_is_integer(half_num_columns)
    error_checking.assert_is_geq(half_num_columns, 0)
    error_checking.assert_is_integer(num_channels)
    error_checking.assert_is_greater(num_channels, 0)

    num_rows = 2 * half_num_rows + 1
    num_columns = 2 * half_num_columns + 1
    weight = 1. / (num_rows * num_columns)

    return numpy.full(
        (num_rows, num_columns, num_channels, num_channels), weight
    )


def check_2d_binary_matrix(binary_matrix):
    """Error-checks 2-D binary matrix.

    :param binary_matrix: 2-D numpy array, containing either Boolean flags or
        integers in 0...1.
    :return: is_boolean: Boolean flag, indicating whether or not matrix is
        Boolean.
    """

    error_checking.assert_is_numpy_array(binary_matrix, num_dimensions=2)

    try:
        error_checking.assert_is_boolean_numpy_array(binary_matrix)
        return True
    except TypeError:
        error_checking.assert_is_integer_numpy_array(binary_matrix)
        error_checking.assert_is_geq_numpy_array(binary_matrix, 0)
        error_checking.assert_is_leq_numpy_array(binary_matrix, 1)
        return False


def get_structure_matrix(buffer_distance_px):
    """Creates structure matrix for dilation or erosion.

    :param buffer_distance_px: Buffer distance (number of pixels).
    :return: structure_matrix: 2-D numpy array of Boolean flags.
    """

    # TODO(thunderhoser): This method is implicitly tested in
    # evaluation_test.py.

    error_checking.assert_is_geq(buffer_distance_px, 0.)

    half_grid_size_px = int(numpy.ceil(buffer_distance_px))
    pixel_offsets = numpy.linspace(
        -half_grid_size_px, half_grid_size_px, num=2 * half_grid_size_px + 1,
        dtype=float
    )

    column_offset_matrix, row_offset_matrix = numpy.meshgrid(
        pixel_offsets, pixel_offsets
    )
    distance_matrix_px = numpy.sqrt(
        row_offset_matrix ** 2 + column_offset_matrix ** 2
    )
    return distance_matrix_px <= buffer_distance_px


def dilate_binary_matrix(binary_matrix, buffer_distance_px):
    """Dilates binary matrix.

    :param binary_matrix: See doc for `check_2d_binary_matrix`.
    :param buffer_distance_px: Buffer distance (pixels).
    :return: dilated_binary_matrix: Dilated version of input.
    """

    # TODO(thunderhoser): This method is implicitly tested in
    # evaluation_test.py.

    check_2d_binary_matrix(binary_matrix)
    error_checking.assert_is_geq(buffer_distance_px, 0.)

    structure_matrix = get_structure_matrix(buffer_distance_px)
    dilated_binary_matrix = binary_dilation(
        binary_matrix.astype(int), structure=structure_matrix, iterations=1,
        border_value=0
    )
    return dilated_binary_matrix.astype(binary_matrix.dtype)


def erode_binary_matrix(binary_matrix, buffer_distance_px):
    """Erodes binary matrix.

    :param binary_matrix: See doc for `check_2d_binary_matrix`.
    :param buffer_distance_px: Buffer distance (pixels).
    :return: eroded_binary_matrix: Eroded version of input.
    """

    # TODO(thunderhoser): This method is implicitly tested in
    # evaluation_test.py.

    check_2d_binary_matrix(binary_matrix)
    error_checking.assert_is_geq(buffer_distance_px, 0.)

    structure_matrix = get_structure_matrix(buffer_distance_px)
    eroded_binary_matrix = binary_erosion(
        binary_matrix.astype(int), structure=structure_matrix, iterations=1,
        border_value=1
    )
    return eroded_binary_matrix.astype(binary_matrix.dtype)


def fill_nans(data_matrix):
    """Fills NaN's with nearest neighbours.

    This method is adapted from the method `fill`, which you can find here:
    https://stackoverflow.com/posts/9262129/revisions

    :param data_matrix: numpy array of real-valued data.
    :return: data_matrix: Same but without NaN's.
    """

    error_checking.assert_is_real_numpy_array(data_matrix)

    indices = distance_transform_edt(
        numpy.isnan(data_matrix), return_distances=False, return_indices=True
    )
    return data_matrix[tuple(indices)]


def fill_nans_by_interp(data_matrix):
    """Fills NaN's in a multi-dimensional grid via linear interpolation.

    Code adapted from Stefan van der Welt:

    https://stackoverflow.com/questions/21690608/
    numpy-inpaint-nans-interpolate-and-extrapolate

    :param data_matrix: 2-D numpy array of floats.
    :return: data_matrix: Same as input but without NaN's.
    """

    error_checking.assert_is_real_numpy_array(data_matrix)
    assert len(data_matrix.shape) > 1

    good_flag_matrix = 1 - numpy.isnan(data_matrix).astype(int)
    good_index_matrix = numpy.array(numpy.nonzero(good_flag_matrix)).T
    good_values = data_matrix[good_flag_matrix == 1]

    interp_object = LinearNDInterpolator(good_index_matrix, good_values)
    all_index_tuples = list(numpy.ndindex(data_matrix.shape))
    interp_values = interp_object(all_index_tuples)

    return numpy.reshape(interp_values, data_matrix.shape)


def mask_to_outline_2d(mask_matrix, x_coords, y_coords):
    """Converts 2-D binary mask to outline of unmasked pixels.

    M = number of rows in grid
    N = number of columns in grid
    C = number of contours in outline

    :param mask_matrix: M-by-N numpy array of Boolean or integer values.  If
        integer values, they must all be in range 0...1.
    :param x_coords: length-N numpy array of x-coordinates.
    :param y_coords: length-M numpy array of y-coordinates.
    :return: x_coords_by_contour: length-C list, where each element is a numpy
        array of x-coordinates.
    :return: y_coords_by_contour: Same but for y-coordinates.
    """

    error_checking.assert_is_numpy_array(mask_matrix, num_dimensions=2)
    mask_matrix = mask_matrix.astype(int)
    error_checking.assert_is_geq_numpy_array(mask_matrix, 0)
    error_checking.assert_is_leq_numpy_array(mask_matrix, 1)

    num_rows = mask_matrix.shape[0]
    num_columns = mask_matrix.shape[1]
    row_indices = numpy.linspace(0, num_rows - 1, num=num_rows, dtype=float)
    column_indices = numpy.linspace(
        0, num_columns - 1, num=num_columns, dtype=float
    )

    row_to_y_interp_object = interp1d(
        row_indices, y_coords, kind='linear', bounds_error=False,
        fill_value='extrapolate', assume_sorted=True
    )
    column_to_x_interp_object = interp1d(
        column_indices, x_coords, kind='linear', bounds_error=False,
        fill_value='extrapolate', assume_sorted=True
    )

    error_checking.assert_is_numpy_array(
        x_coords, exact_dimensions=numpy.array([num_columns], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(numpy.diff(x_coords), 0.)

    error_checking.assert_is_numpy_array(
        y_coords, exact_dimensions=numpy.array([num_rows], dtype=int)
    )
    error_checking.assert_is_greater_numpy_array(numpy.diff(y_coords), 0.)

    coord_matrix_by_contour = skimage.measure.find_contours(mask_matrix, 0.99)

    num_contours = len(coord_matrix_by_contour)
    x_coords_by_contour = [numpy.array([])] * num_contours
    y_coords_by_contour = [numpy.array([])] * num_contours

    for k in range(num_contours):
        x_coords_by_contour[k] = column_to_x_interp_object(
            coord_matrix_by_contour[k][:, 1]
        )
        y_coords_by_contour[k] = row_to_y_interp_object(
            coord_matrix_by_contour[k][:, 0]
        )

    return x_coords_by_contour, y_coords_by_contour
