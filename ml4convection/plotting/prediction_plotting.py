"""Plotting methods for predictions."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking

BACKGROUND_COLOUR = numpy.full(3, 1.)
NO_CONVECTION_COLOUR = numpy.full(3, 1.)
ACTUAL_CONVECTION_COLOUR = numpy.array([247, 129, 191], dtype=float) / 255
PREDICTED_CONVECTION_COLOUR = numpy.full(3, 153. / 255)

ACTUAL_CONVECTION_OPACITY = 1.
PREDICTED_CONVECTION_OPACITY = 0.5


def _get_colour_scheme(for_targets):
    """Returns colour scheme for either predicted or target (actual) values.

    :param for_targets: Boolean flag.  If True (False), will return colour
        scheme for actual (predicted) values.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    main_colour_list = [NO_CONVECTION_COLOUR]

    if for_targets:
        main_colour_list.append(ACTUAL_CONVECTION_COLOUR)
    else:
        main_colour_list.append(PREDICTED_CONVECTION_COLOUR)

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(BACKGROUND_COLOUR)
    colour_map_object.set_over(BACKGROUND_COLOUR)

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        numpy.array([0.5, 1.5]), colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def plot_with_basemap(
        target_matrix, prediction_matrix, axes_object, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg):
    """Plots predicted and target (actual) convection masks with basemap.

    M = number of rows in grid
    N = number of columns in grid

    :param target_matrix: M-by-N numpy array of integers (in range 0...1),
        indicating where convection actually occurred.
    :param prediction_matrix: M-by-N numpy array of integers (in range 0...1),
        indicating where convection was predicted.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param min_latitude_deg_n: Latitude (deg N) at southernmost row of grid
        points.
    :param min_longitude_deg_e: Longitude (deg E) at westernmost column of
        grid points.
    :param latitude_spacing_deg: Spacing (deg N) between adjacent grid rows.
    :param longitude_spacing_deg: Spacing (deg E) between adjacent grid columns.
    """

    error_checking.assert_is_integer_numpy_array(target_matrix)
    error_checking.assert_is_geq_numpy_array(target_matrix, 0)
    error_checking.assert_is_leq_numpy_array(target_matrix, 1)

    error_checking.assert_is_integer_numpy_array(prediction_matrix)
    error_checking.assert_is_geq_numpy_array(prediction_matrix, 0)
    error_checking.assert_is_leq_numpy_array(prediction_matrix, 1)

    edge_target_matrix, edge_latitudes_deg_n, edge_longitudes_deg_e = (
        grids.latlng_field_grid_points_to_edges(
            field_matrix=target_matrix.astype(float),
            min_latitude_deg=min_latitude_deg_n,
            min_longitude_deg=min_longitude_deg_e,
            lat_spacing_deg=latitude_spacing_deg,
            lng_spacing_deg=longitude_spacing_deg
        )
    )

    edge_prediction_matrix = grids.latlng_field_grid_points_to_edges(
        field_matrix=prediction_matrix.astype(float),
        min_latitude_deg=min_latitude_deg_n,
        min_longitude_deg=min_longitude_deg_e,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg
    )[0]

    edge_target_matrix = numpy.ma.masked_where(
        numpy.isnan(edge_target_matrix), edge_target_matrix
    )
    colour_map_object, colour_norm_object = _get_colour_scheme(for_targets=True)

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    pyplot.pcolormesh(
        edge_longitudes_deg_e, edge_latitudes_deg_n, edge_target_matrix,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', axes=axes_object, zorder=-1e11,
        alpha=ACTUAL_CONVECTION_OPACITY
    )

    edge_prediction_matrix = numpy.ma.masked_where(
        numpy.isnan(edge_prediction_matrix), edge_prediction_matrix
    )
    colour_map_object, colour_norm_object = _get_colour_scheme(
        for_targets=False
    )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    pyplot.pcolormesh(
        edge_longitudes_deg_e, edge_latitudes_deg_n, edge_prediction_matrix,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', axes=axes_object, zorder=-1e11,
        alpha=PREDICTED_CONVECTION_OPACITY
    )
