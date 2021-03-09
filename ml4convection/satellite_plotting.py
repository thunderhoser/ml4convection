"""Plotting methods for satellite data."""

import os
import sys
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import error_checking
import gg_plotting_utils

DEFAULT_MIN_TEMP_KELVINS = 190.
DEFAULT_MAX_TEMP_KELVINS = 310.
DEFAULT_CUTOFF_TEMP_KELVINS = 240.


def _get_colour_scheme(
        min_temp_kelvins=DEFAULT_MIN_TEMP_KELVINS,
        max_temp_kelvins=DEFAULT_MAX_TEMP_KELVINS,
        cutoff_temp_kelvins=DEFAULT_CUTOFF_TEMP_KELVINS):
    """Returns colour scheme for brightness temperature.

    :param min_temp_kelvins: Minimum temperature in colour scheme.
    :param max_temp_kelvins: Max temperature in colour scheme.
    :param cutoff_temp_kelvins: Cutoff between grey and non-grey colours.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    error_checking.assert_is_greater(max_temp_kelvins, cutoff_temp_kelvins)
    error_checking.assert_is_greater(cutoff_temp_kelvins, min_temp_kelvins)

    normalized_values = numpy.linspace(0, 1, num=1001, dtype=float)

    grey_colour_map_object = pyplot.get_cmap('Greys')
    grey_temps_kelvins = numpy.linspace(
        cutoff_temp_kelvins, max_temp_kelvins, num=1001, dtype=float
    )
    grey_rgb_matrix = grey_colour_map_object(normalized_values)[:, :-1]

    plasma_colour_map_object = pyplot.get_cmap('plasma')
    plasma_temps_kelvins = numpy.linspace(
        min_temp_kelvins, cutoff_temp_kelvins, num=1001, dtype=float
    )
    plasma_rgb_matrix = plasma_colour_map_object(normalized_values)[:, :-1]

    boundary_temps_kelvins = numpy.concatenate(
        (plasma_temps_kelvins, grey_temps_kelvins), axis=0
    )
    rgb_matrix = numpy.concatenate(
        (plasma_rgb_matrix, grey_rgb_matrix), axis=0
    )

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        boundary_temps_kelvins, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _add_colour_bar(
        brightness_temp_matrix_kelvins, axes_object, colour_map_object,
        colour_norm_object, orientation_string, font_size):
    """Adds colour bar to plot.

    :param brightness_temp_matrix_kelvins: See doc for `plot_2d_grid_latlng`.
    :param axes_object: Same.
    :param colour_map_object: See doc for `_get_colour_scheme`.
    :param colour_norm_object: Same.
    :param orientation_string: Orientation ("vertical" or "horizontal").
    :param font_size: Font size for labels on colour bar.
    :return: colour_bar_object: See doc for `plot_2d_grid_latlng`.
    """

    print('\n\n\n\n\n\n\nPADDING\n\n\n\n\n\n\n\n')

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=brightness_temp_matrix_kelvins,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=orientation_string,
        extend_min=True, extend_max=True, font_size=font_size, padding=0.1
    )

    num_tick_values = 1 + int(numpy.round(
        (DEFAULT_MAX_TEMP_KELVINS - DEFAULT_MIN_TEMP_KELVINS) / 10
    ))
    tick_values = numpy.linspace(
        DEFAULT_MIN_TEMP_KELVINS, DEFAULT_MAX_TEMP_KELVINS, num=num_tick_values,
        dtype=int
    )

    tick_strings = ['{0:d}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return colour_bar_object


def plot_2d_grid_latlng(
        brightness_temp_matrix_kelvins, axes_object, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg,
        cbar_orientation_string='vertical',
        font_size=gg_plotting_utils.FONT_SIZE):
    """Plots brightness temperatures on 2-D grid with lat-long coordinates.

    M = number of rows in grid
    N = number of columns in grid

    :param brightness_temp_matrix_kelvins: M-by-N numpy array of brightness
        temperatures.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param min_latitude_deg_n: Latitude (deg N) at southernmost row of grid
        points.
    :param min_longitude_deg_e: Longitude (deg E) at westernmost column of
        grid points.
    :param latitude_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param longitude_spacing_deg: Spacing (deg E) between adjacent grid columns.
    :param cbar_orientation_string: Colour-bar orientation.  May be
        "horizontal", "vertical", or None.
    :param font_size: Font size.
    :return: colour_bar_object: Colour-bar handle (instance of
        `matplotlib.pyplot.colorbar`).
    """

    if cbar_orientation_string is not None:
        error_checking.assert_is_string(cbar_orientation_string)

    edge_temp_matrix_kelvins, edge_latitudes_deg_n, edge_longitudes_deg_e = (
        grids.latlng_field_grid_points_to_edges(
            field_matrix=brightness_temp_matrix_kelvins,
            min_latitude_deg=min_latitude_deg_n,
            min_longitude_deg=min_longitude_deg_e,
            lat_spacing_deg=latitude_spacing_deg,
            lng_spacing_deg=longitude_spacing_deg
        )
    )

    edge_temp_matrix_kelvins = numpy.ma.masked_where(
        numpy.isnan(edge_temp_matrix_kelvins), edge_temp_matrix_kelvins
    )

    colour_map_object, colour_norm_object = _get_colour_scheme()

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        edge_longitudes_deg_e, edge_latitudes_deg_n, edge_temp_matrix_kelvins,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e11
    )

    if cbar_orientation_string is None:
        return None

    return _add_colour_bar(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object, colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=cbar_orientation_string, font_size=font_size
    )


def plot_2d_grid_xy(
        brightness_temp_matrix_kelvins, axes_object,
        cbar_orientation_string='vertical',
        font_size=gg_plotting_utils.FONT_SIZE):
    """Plots brightness temp on 2-D grid with x-y coordinates (no basemap).

    :param brightness_temp_matrix_kelvins: See doc for `plot_2d_grid_latlng`.
    :param axes_object: Same.
    :param cbar_orientation_string: Same.
    :param font_size: Same.
    :return: colour_bar_object: Same.
    """

    if cbar_orientation_string is not None:
        error_checking.assert_is_string(cbar_orientation_string)

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, num_dimensions=2
    )

    num_grid_rows = brightness_temp_matrix_kelvins.shape[0]
    num_grid_columns = brightness_temp_matrix_kelvins.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    edge_temp_matrix_kelvins, edge_x_coords, edge_y_coords = (
        grids.xy_field_grid_points_to_edges(
            field_matrix=brightness_temp_matrix_kelvins,
            x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
            x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing
        )
    )

    edge_temp_matrix_kelvins = numpy.ma.masked_where(
        numpy.isnan(edge_temp_matrix_kelvins), edge_temp_matrix_kelvins
    )

    colour_map_object, colour_norm_object = _get_colour_scheme()

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        edge_x_coords, edge_y_coords, edge_temp_matrix_kelvins,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e11, transform=axes_object.transAxes
    )

    if cbar_orientation_string is None:
        return None

    return _add_colour_bar(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object=axes_object, colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string=cbar_orientation_string, font_size=font_size
    )


def plot_3d_grid_latlng(
        brightness_temp_matrix_kelvins, axes_object_matrix, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg):
    """Plots brightness temperatures on 3-D grid with lat-long coordinates.

    M = number of rows in grid
    N = number of columns in grid
    C = number of channels (spectral bands)

    :param brightness_temp_matrix_kelvins: M-by-N-by-C numpy array of brightness
        temperatures.
    :param axes_object_matrix: 2-D numpy array of axes (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param min_latitude_deg_n: See doc for `plot_2d_grid_latlng`.
    :param min_longitude_deg_e: Same.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    """

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, num_dimensions=3
    )
    error_checking.assert_is_numpy_array(axes_object_matrix, num_dimensions=2)

    num_channels = brightness_temp_matrix_kelvins.shape[-1]

    for k in range(num_channels):
        i, j = numpy.unravel_index(k, axes_object_matrix.shape)

        plot_2d_grid_latlng(
            brightness_temp_matrix_kelvins=
            brightness_temp_matrix_kelvins[..., k],
            axes_object=axes_object_matrix[i, j],
            min_latitude_deg_n=min_latitude_deg_n,
            min_longitude_deg_e=min_longitude_deg_e,
            latitude_spacing_deg=latitude_spacing_deg,
            longitude_spacing_deg=longitude_spacing_deg,
            cbar_orientation_string=None
        )


def plot_3d_grid_xy(brightness_temp_matrix_kelvins, axes_object_matrix):
    """Plots brightness temp on 3-D grid with x-y coordinates (no basemap).

    :param brightness_temp_matrix_kelvins: See doc for `plot_3d_grid_latlng`.
    :param axes_object_matrix: Same.
    """

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, num_dimensions=3
    )
    error_checking.assert_is_numpy_array(axes_object_matrix, num_dimensions=2)

    num_channels = brightness_temp_matrix_kelvins.shape[-1]

    for k in range(num_channels):
        i, j = numpy.unravel_index(k, axes_object_matrix.shape)

        plot_2d_grid_xy(
            brightness_temp_matrix_kelvins=
            brightness_temp_matrix_kelvins[..., k],
            axes_object=axes_object_matrix[i, j],
            cbar_orientation_string=None
        )


def plot_4d_grid_latlng(
        brightness_temp_matrix_kelvins, axes_object_matrix, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg):
    """Plots brightness temperatures on 4-D grid with lat-long coordinates.

    M = number of rows in grid
    N = number of columns in grid
    T = number of times
    C = number of channels (spectral bands)

    :param brightness_temp_matrix_kelvins: M-by-N-by-T-by-C numpy array of
        brightness temperatures.
    :param axes_object_matrix: T-by-C numpy array of axes (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param min_latitude_deg_n: See doc for `plot_2d_grid_latlng`.
    :param min_longitude_deg_e: Same.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    """

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, num_dimensions=4
    )

    num_lag_times = brightness_temp_matrix_kelvins.shape[-2]
    num_channels = brightness_temp_matrix_kelvins.shape[-1]

    expected_dim = numpy.array([num_lag_times, num_channels], dtype=int)
    error_checking.assert_is_numpy_array(
        axes_object_matrix, exact_dimensions=expected_dim
    )

    for j in range(num_lag_times):
        for k in range(num_channels):
            plot_2d_grid_latlng(
                brightness_temp_matrix_kelvins=
                brightness_temp_matrix_kelvins[..., j, k],
                axes_object=axes_object_matrix[j, k],
                min_latitude_deg_n=min_latitude_deg_n,
                min_longitude_deg_e=min_longitude_deg_e,
                latitude_spacing_deg=latitude_spacing_deg,
                longitude_spacing_deg=longitude_spacing_deg,
                cbar_orientation_string=None
            )


def plot_4d_grid_xy(brightness_temp_matrix_kelvins, axes_object_matrix):
    """Plots brightness temp on 4-D grid with x-y coordinates (no basemap).

    :param brightness_temp_matrix_kelvins: See doc for `plot_4d_grid_latlng`.
    :param axes_object_matrix: Same.
    """

    error_checking.assert_is_numpy_array(
        brightness_temp_matrix_kelvins, num_dimensions=4
    )

    num_lag_times = brightness_temp_matrix_kelvins.shape[-2]
    num_channels = brightness_temp_matrix_kelvins.shape[-1]

    expected_dim = numpy.array([num_lag_times, num_channels], dtype=int)
    error_checking.assert_is_numpy_array(
        axes_object_matrix, exact_dimensions=expected_dim
    )

    for j in range(num_lag_times):
        for k in range(num_channels):
            plot_2d_grid_xy(
                brightness_temp_matrix_kelvins=
                brightness_temp_matrix_kelvins[..., j, k],
                axes_object=axes_object_matrix[j, k],
                cbar_orientation_string=None
            )
