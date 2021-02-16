"""Plotting methods for saliency."""

import numpy
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking

DEFAULT_COLOUR_MAP_OBJECT = pyplot.get_cmap('binary')
DEFAULT_CONTOUR_WIDTH = 2.


def plot_2d_grid_latlng(
        saliency_matrix, axes_object, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg,
        min_abs_contour_value, max_abs_contour_value, half_num_contours,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        line_width=DEFAULT_CONTOUR_WIDTH):
    """Plots saliency on 2-D grid with lat-long coordinates.

    M = number of rows in grid
    N = number of columns in grid

    :param saliency_matrix: M-by-N numpy array of saliency values.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param min_latitude_deg_n: Latitude (deg N) at southernmost row of grid
        points.
    :param min_longitude_deg_e: Longitude (deg E) at westernmost column of
        grid points.
    :param latitude_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param longitude_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :param min_abs_contour_value: Minimum absolute saliency to plot.
    :param max_abs_contour_value: Max absolute saliency to plot.
    :param half_num_contours: Number of contours on either side of zero.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param line_width: Width of contour lines.
    """

    if min_abs_contour_value < 0.001 or max_abs_contour_value < 0.01:
        min_abs_contour_value = 0.001
        max_abs_contour_value = 0.01

    error_checking.assert_is_greater(
        max_abs_contour_value, min_abs_contour_value
    )
    error_checking.assert_is_integer(half_num_contours)
    error_checking.assert_is_geq(half_num_contours, 5)
    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=2)

    latitudes_deg_n, longitudes_deg_e = grids.get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg_n,
        min_longitude_deg=min_longitude_deg_e,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg,
        num_rows=saliency_matrix.shape[0],
        num_columns=saliency_matrix.shape[1]
    )

    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=latitudes_deg_n,
            unique_longitudes_deg=longitudes_deg_e
        )
    )

    # Plot positive values.
    contour_levels = numpy.linspace(
        min_abs_contour_value, max_abs_contour_value, num=half_num_contours
    )

    axes_object.contour(
        longitude_matrix_deg_e, latitude_matrix_deg_n, saliency_matrix,
        contour_levels, cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles='solid', zorder=1e6
    )

    # Plot negative values.
    axes_object.contour(
        longitude_matrix_deg_e, latitude_matrix_deg_n, -1 * saliency_matrix,
        contour_levels, cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles='dashed', zorder=1e6
    )


def plot_2d_grid_xy(
        saliency_matrix, axes_object, min_abs_contour_value,
        max_abs_contour_value, half_num_contours,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        line_width=DEFAULT_CONTOUR_WIDTH):
    """Plots saliency on 2-D grid with x-y coordinates (no basemap).

    :param saliency_matrix: See doc for `plot_2d_grid_latlng`.
    :param axes_object: Same.
    :param min_abs_contour_value: Same.
    :param max_abs_contour_value: Same.
    :param half_num_contours: Same.
    :param colour_map_object: Same.
    :param line_width: Same.
    """

    if min_abs_contour_value < 0.001 or max_abs_contour_value < 0.01:
        min_abs_contour_value = 0.001
        max_abs_contour_value = 0.01

    error_checking.assert_is_greater(
        max_abs_contour_value, min_abs_contour_value
    )
    error_checking.assert_is_integer(half_num_contours)
    error_checking.assert_is_geq(half_num_contours, 5)
    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=2)

    num_grid_rows = saliency_matrix.shape[0]
    num_grid_columns = saliency_matrix.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords, y_coords)

    # Plot positive values.
    contour_levels = numpy.linspace(
        min_abs_contour_value, max_abs_contour_value, num=half_num_contours
    )

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, saliency_matrix,
        contour_levels, cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles='solid', zorder=1e6,
        transform=axes_object.transAxes
    )

    # Plot negative values.
    axes_object.contour(
        x_coord_matrix, y_coord_matrix, -1 * saliency_matrix,
        contour_levels, cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles='dashed', zorder=1e6,
        transform=axes_object.transAxes
    )


def plot_3d_grid_latlng(
        saliency_matrix, axes_object_matrix, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg,
        min_abs_contour_value, max_abs_contour_value, half_num_contours,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        line_width=DEFAULT_CONTOUR_WIDTH):
    """Plots saliency on 3-D grid with lat-long coordinates.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels

    :param saliency_matrix: M-by-N-by-C numpy array of saliency values.
    :param axes_object_matrix: 2-D numpy array of axes (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param min_latitude_deg_n: See documentation for `plot_2d_grid`.
    :param min_longitude_deg_e: Same.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    :param min_abs_contour_value: Same.
    :param max_abs_contour_value: Same.
    :param half_num_contours: Same.
    :param colour_map_object: Same.
    :param line_width: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(axes_object_matrix, num_dimensions=2)

    num_channels = saliency_matrix.shape[-1]

    for k in range(num_channels):
        i, j = numpy.unravel_index(k, axes_object_matrix.shape)

        plot_2d_grid_latlng(
            saliency_matrix=saliency_matrix[..., k],
            axes_object=axes_object_matrix[i, j],
            min_latitude_deg_n=min_latitude_deg_n,
            min_longitude_deg_e=min_longitude_deg_e,
            latitude_spacing_deg=latitude_spacing_deg,
            longitude_spacing_deg=longitude_spacing_deg,
            min_abs_contour_value=min_abs_contour_value,
            max_abs_contour_value=max_abs_contour_value,
            half_num_contours=half_num_contours,
            colour_map_object=colour_map_object, line_width=line_width
        )


def plot_3d_grid_xy(
        saliency_matrix, axes_object_matrix, min_abs_contour_value,
        max_abs_contour_value, half_num_contours,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        line_width=DEFAULT_CONTOUR_WIDTH):
    """Plots saliency on 3-D grid with x-y coordinates (no basemap).

    :param saliency_matrix: See doc for `plot_3d_grid_latlng`.
    :param axes_object_matrix: Same.
    :param min_abs_contour_value: Same.
    :param max_abs_contour_value: Same.
    :param half_num_contours: Same.
    :param colour_map_object: Same.
    :param line_width: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array(axes_object_matrix, num_dimensions=2)

    num_channels = saliency_matrix.shape[-1]

    for k in range(num_channels):
        i, j = numpy.unravel_index(k, axes_object_matrix.shape)

        plot_2d_grid_xy(
            saliency_matrix=saliency_matrix[..., k],
            axes_object=axes_object_matrix[i, j],
            min_abs_contour_value=min_abs_contour_value,
            max_abs_contour_value=max_abs_contour_value,
            half_num_contours=half_num_contours,
            colour_map_object=colour_map_object, line_width=line_width
        )


def plot_4d_grid_latlng(
        saliency_matrix, axes_object_matrix, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg,
        min_abs_contour_value, max_abs_contour_value, half_num_contours,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        line_width=DEFAULT_CONTOUR_WIDTH):
    """Plots saliency on 4-D grid with lat-long coordinates.

    M = number of rows in grid
    N = number of columns in grid
    T = number of lag times
    C = number of channels

    :param saliency_matrix: M-by-N-by-T-by-C numpy array of saliency values.
    :param axes_object_matrix: T-by-C numpy array of axes (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param min_latitude_deg_n: See documentation for `plot_2d_grid`.
    :param min_longitude_deg_e: Same.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    :param min_abs_contour_value: Same.
    :param max_abs_contour_value: Same.
    :param half_num_contours: Same.
    :param colour_map_object: Same.
    :param line_width: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=4)

    num_lag_times = saliency_matrix.shape[-2]
    num_channels = saliency_matrix.shape[-1]

    expected_dim = numpy.array([num_lag_times, num_channels], dtype=int)
    error_checking.assert_is_numpy_array(
        axes_object_matrix, exact_dimensions=expected_dim
    )

    for j in range(num_lag_times):
        for k in range(num_channels):
            plot_2d_grid_latlng(
                saliency_matrix=saliency_matrix[..., j, k],
                axes_object=axes_object_matrix[j, k],
                min_latitude_deg_n=min_latitude_deg_n,
                min_longitude_deg_e=min_longitude_deg_e,
                latitude_spacing_deg=latitude_spacing_deg,
                longitude_spacing_deg=longitude_spacing_deg,
                min_abs_contour_value=min_abs_contour_value,
                max_abs_contour_value=max_abs_contour_value,
                half_num_contours=half_num_contours,
                colour_map_object=colour_map_object, line_width=line_width
            )


def plot_4d_grid_xy(
        saliency_matrix, axes_object_matrix, min_abs_contour_value,
        max_abs_contour_value, half_num_contours,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        line_width=DEFAULT_CONTOUR_WIDTH):
    """Plots saliency on 4-D grid with x-y coordinates (no basemap).

    :param saliency_matrix: See doc for `plot_3d_grid_latlng`.
    :param axes_object_matrix: Same.
    :param min_abs_contour_value: Same.
    :param max_abs_contour_value: Same.
    :param half_num_contours: Same.
    :param colour_map_object: Same.
    :param line_width: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    error_checking.assert_is_numpy_array(saliency_matrix, num_dimensions=4)

    num_lag_times = saliency_matrix.shape[-2]
    num_channels = saliency_matrix.shape[-1]

    expected_dim = numpy.array([num_lag_times, num_channels], dtype=int)
    error_checking.assert_is_numpy_array(
        axes_object_matrix, exact_dimensions=expected_dim
    )

    for j in range(num_lag_times):
        for k in range(num_channels):
            plot_2d_grid_xy(
                saliency_matrix=saliency_matrix[..., j, k],
                axes_object=axes_object_matrix[j, k],
                min_abs_contour_value=min_abs_contour_value,
                max_abs_contour_value=max_abs_contour_value,
                half_num_contours=half_num_contours,
                colour_map_object=colour_map_object, line_width=line_width
            )
