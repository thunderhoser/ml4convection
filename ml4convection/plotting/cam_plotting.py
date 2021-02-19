"""Plotting methods for class-activation maps."""

import numpy
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking

DEFAULT_COLOUR_MAP_OBJECT = pyplot.get_cmap('binary')
DEFAULT_CONTOUR_WIDTH = 2
DEFAULT_CONTOUR_STYLE = 'solid'


def plot_2d_grid_latlng(
        class_activation_matrix, axes_object, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg,
        min_contour_value, max_contour_value, num_contours,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        line_width=DEFAULT_CONTOUR_WIDTH, line_style=DEFAULT_CONTOUR_STYLE):
    """Plots class activation on 2-D grid with lat-long coordinates.

    M = number of rows in grid
    N = number of columns in grid

    :param class_activation_matrix: M-by-N numpy array of class activations.
    :param axes_object: Instance of `matplotlib.axes._subplots.AxesSubplot`.
        Will plot on these axes.
    :param min_latitude_deg_n: Latitude (deg N) at southernmost row of grid
        points.
    :param min_longitude_deg_e: Longitude (deg E) at westernmost column of
        grid points.
    :param latitude_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param longitude_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :param min_contour_value: Minimum class activation to plot.
    :param max_contour_value: Max class activation to plot.
    :param num_contours: Number of contours.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param line_width: Width of contour lines.
    :param line_style: Style of contour lines.
    """

    error_checking.assert_is_greater(
        max_contour_value, min_contour_value
    )
    error_checking.assert_is_integer(num_contours)
    error_checking.assert_is_geq(num_contours, 5)
    error_checking.assert_is_numpy_array_without_nan(class_activation_matrix)
    error_checking.assert_is_numpy_array(
        class_activation_matrix, num_dimensions=2
    )

    latitudes_deg_n, longitudes_deg_e = grids.get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg_n,
        min_longitude_deg=min_longitude_deg_e,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg,
        num_rows=class_activation_matrix.shape[0],
        num_columns=class_activation_matrix.shape[1]
    )

    latitude_matrix_deg_n, longitude_matrix_deg_e = (
        grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=latitudes_deg_n,
            unique_longitudes_deg=longitudes_deg_e
        )
    )

    contour_levels = numpy.linspace(
        min_contour_value, max_contour_value, num=num_contours
    )

    axes_object.contour(
        longitude_matrix_deg_e, latitude_matrix_deg_n, class_activation_matrix,
        contour_levels, cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles=line_style, zorder=1e6
    )


def plot_2d_grid_xy(
        class_activation_matrix, axes_object, min_contour_value,
        max_contour_value, num_contours,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        line_width=DEFAULT_CONTOUR_WIDTH, line_style=DEFAULT_CONTOUR_STYLE):
    """Plots class activation on 2-D grid with x-y coordinates (no basemap).

    :param class_activation_matrix: See doc for `plot_2d_grid_latlng`.
    :param axes_object: Same.
    :param min_contour_value: Same.
    :param max_contour_value: Same.
    :param num_contours: Same.
    :param colour_map_object: Same.
    :param line_width: Same.
    :param line_style: Same.
    """

    error_checking.assert_is_greater(
        max_contour_value, min_contour_value
    )
    error_checking.assert_is_integer(num_contours)
    error_checking.assert_is_geq(num_contours, 5)
    error_checking.assert_is_numpy_array_without_nan(class_activation_matrix)
    error_checking.assert_is_numpy_array(
        class_activation_matrix, num_dimensions=2
    )

    num_grid_rows = class_activation_matrix.shape[0]
    num_grid_columns = class_activation_matrix.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    x_coord_matrix, y_coord_matrix = numpy.meshgrid(x_coords, y_coords)

    contour_levels = numpy.linspace(
        min_contour_value, max_contour_value, num=num_contours
    )

    axes_object.contour(
        x_coord_matrix, y_coord_matrix, class_activation_matrix,
        contour_levels, cmap=colour_map_object,
        vmin=numpy.min(contour_levels), vmax=numpy.max(contour_levels),
        linewidths=line_width, linestyles=line_style, zorder=1e6,
        transform=axes_object.transAxes
    )
