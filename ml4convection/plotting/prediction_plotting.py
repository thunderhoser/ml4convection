"""Plotting methods for predictions."""

import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import error_checking

BACKGROUND_COLOUR = numpy.full(3, 1.)
NO_MASK_COLOUR = numpy.full(3, 1.)
ACTUAL_MASK_COLOUR = numpy.array([247, 129, 191], dtype=float) / 255
PREDICTED_MASK_COLOUR = numpy.full(3, 152. / 255)

ACTUAL_MASK_OPACITY = 1.
PREDICTED_MASK_OPACITY = 0.5

DEFAULT_TARGET_MARKER_TYPE = 'o'
# DEFAULT_TARGET_MARKER_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DEFAULT_TARGET_MARKER_COLOUR = numpy.full(3, 0.)

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _get_deterministic_colour_scheme(for_targets):
    """Returns colour scheme for either predicted or actual convection mask.

    :param for_targets: Boolean flag.  If True (False), will return colour
        scheme for actual (predicted) values.
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    main_colour_list = [NO_MASK_COLOUR] * 2

    if for_targets:
        main_colour_list += [ACTUAL_MASK_COLOUR] * 2
    else:
        main_colour_list += [PREDICTED_MASK_COLOUR] * 2

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(BACKGROUND_COLOUR)
    colour_map_object.set_over(main_colour_list[-1])

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        numpy.array([0, 0.5, 1, 1.5]), colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def get_prob_colour_scheme(max_probability=1., make_lowest_prob_grey=False):
    """Returns colour scheme for probabilities.

    :param max_probability: Max probability in colour bar.
    :param make_lowest_prob_grey: Boolean flag.  If True (False), will make
        lowest probabilities grey (white).
    :return: colour_map_object: See doc for `_get_deterministic_colour_scheme`.
    :return: colour_norm_object: Same.
    """

    error_checking.assert_is_greater(max_probability, 0.)
    error_checking.assert_is_leq(max_probability, 1.)
    error_checking.assert_is_boolean(make_lowest_prob_grey)

    green_colour_map_object = pyplot.get_cmap(name='Greens')
    data_values = numpy.linspace(0.5, 1, num=9, dtype=float)
    main_colour_matrix = green_colour_map_object(data_values)[:, :-1]

    purple_colour_map_object = pyplot.get_cmap(name='Purples')
    data_values = numpy.linspace(0.5, 1, num=10, dtype=float)
    new_colour_matrix = purple_colour_map_object(data_values)[:, :-1]
    main_colour_matrix = numpy.concatenate(
        (main_colour_matrix, new_colour_matrix), axis=0
    )

    main_colour_list = [
        main_colour_matrix[i, :] for i in range(main_colour_matrix.shape[0])
    ]

    if make_lowest_prob_grey:
        main_colour_list = [numpy.full(3, 152. / 255)] + main_colour_list

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(BACKGROUND_COLOUR)
    colour_map_object.set_over(main_colour_list[-1])

    colour_bounds = max_probability * numpy.linspace(0.05, 1, num=20)

    if make_lowest_prob_grey:
        colour_bounds = numpy.concatenate((
            numpy.array([max_probability * 0.001]),
            colour_bounds
        ))

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def get_prob_colour_scheme_hail(
        max_probability=1., make_lowest_prob_grey=False,
        make_highest_prob_black=False):
    """Returns colour scheme for probabilities.

    :param max_probability: Max probability in colour bar.
    :param make_lowest_prob_grey: Boolean flag.  If True (False), will make
        lowest probabilities grey (white).
    :param make_highest_prob_black: Boolean flag.  If True (False), will make
        highest probabilities grey (black).
    :return: colour_map_object: See doc for `_get_deterministic_colour_scheme`.
    :return: colour_norm_object: Same.
    """

    error_checking.assert_is_greater(max_probability, 0.)
    error_checking.assert_is_leq(max_probability, 1.)
    error_checking.assert_is_boolean(make_lowest_prob_grey)
    error_checking.assert_is_boolean(make_highest_prob_black)

    main_colour_list = [
        # numpy.array([0, 90, 50]),
        numpy.array([35, 139, 69]),
        numpy.array([65, 171, 93]), numpy.array([116, 196, 118]),
        numpy.array([161, 217, 155]), numpy.array([8, 69, 148]),
        numpy.array([33, 113, 181]), numpy.array([66, 146, 198]),
        numpy.array([107, 174, 214]), numpy.array([158, 202, 225]),
        numpy.array([74, 20, 134]), numpy.array([106, 81, 163]),
        numpy.array([128, 125, 186]), numpy.array([158, 154, 200]),
        numpy.array([188, 189, 220]), numpy.array([153, 0, 13]),
        numpy.array([203, 24, 29]), numpy.array([239, 59, 44]),
        numpy.array([251, 106, 74]), numpy.array([252, 146, 114])
    ]

    for i in range(len(main_colour_list)):
        main_colour_list[i] = main_colour_list[i].astype(float) / 255

    if make_highest_prob_black:
        main_colour_list[i][-1] = numpy.full(3, 0.)

    if make_lowest_prob_grey:
        main_colour_list = [numpy.full(3, 152. / 255)] + main_colour_list

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(BACKGROUND_COLOUR)
    colour_map_object.set_over(main_colour_list[-1])

    colour_bounds = max_probability * numpy.linspace(0.05, 1, num=20)

    if make_lowest_prob_grey:
        colour_bounds = numpy.concatenate((
            numpy.array([max_probability * 0.001]),
            colour_bounds
        ))

    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def plot_deterministic(
        target_matrix, prediction_matrix, axes_object, min_latitude_deg_n,
        min_longitude_deg_e, latitude_spacing_deg, longitude_spacing_deg):
    """Plots gridded predictions and labels.

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

    colour_map_object, colour_norm_object = _get_deterministic_colour_scheme(
        for_targets=True
    )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        edge_longitudes_deg_e, edge_latitudes_deg_n, edge_target_matrix,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e11, alpha=ACTUAL_MASK_OPACITY
    )

    colour_map_object, colour_norm_object = _get_deterministic_colour_scheme(
        for_targets=False
    )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        edge_longitudes_deg_e, edge_latitudes_deg_n, edge_prediction_matrix,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e11, alpha=PREDICTED_MASK_OPACITY
    )


def plot_probabilistic(
        target_matrix, probability_matrix, figure_object, axes_object,
        min_latitude_deg_n, min_longitude_deg_e, latitude_spacing_deg,
        longitude_spacing_deg, colour_map_object, colour_norm_object,
        target_marker_size_grid_cells=0.5,
        target_marker_type=DEFAULT_TARGET_MARKER_TYPE,
        target_marker_colour=DEFAULT_TARGET_MARKER_COLOUR):
    """Plots gridded probabilities and labels.

    M = number of rows in grid
    N = number of columns in grid

    :param target_matrix: See doc for `plot_deterministic`.
    :param probability_matrix: M-by-N numpy array with forecast probabilities of
        convection.
    :param figure_object: Will plot in this figure (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: See doc for `plot_deterministic`.
    :param min_latitude_deg_n: Same.
    :param min_longitude_deg_e: Same.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    :param target_marker_size_grid_cells: Size of marker used to show where
        convection occurs.
    :param target_marker_type: Type of marker used to show where convection
        occurs.
    :param target_marker_colour: Colour of marker used to show where convection
        occurs.
    """

    error_checking.assert_is_geq_numpy_array(probability_matrix, 0.)
    error_checking.assert_is_leq_numpy_array(probability_matrix, 1.)
    error_checking.assert_is_numpy_array(probability_matrix, num_dimensions=2)

    error_checking.assert_is_integer_numpy_array(target_matrix)
    error_checking.assert_is_geq_numpy_array(target_matrix, 0)
    error_checking.assert_is_leq_numpy_array(target_matrix, 1)
    error_checking.assert_is_numpy_array(
        target_matrix,
        exact_dimensions=numpy.array(probability_matrix.shape, dtype=int)
    )

    latitudes_deg_n, longitudes_deg_e = grids.get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg_n,
        min_longitude_deg=min_longitude_deg_e,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg,
        num_rows=target_matrix.shape[0], num_columns=target_matrix.shape[1]
    )

    edge_probability_matrix, edge_latitudes_deg_n, edge_longitudes_deg_e = (
        grids.latlng_field_grid_points_to_edges(
            field_matrix=probability_matrix,
            min_latitude_deg=min_latitude_deg_n,
            min_longitude_deg=min_longitude_deg_e,
            lat_spacing_deg=latitude_spacing_deg,
            lng_spacing_deg=longitude_spacing_deg
        )
    )

    if hasattr(colour_norm_object, 'boundaries'):
        min_colour_value = colour_norm_object.boundaries[0]
        max_colour_value = colour_norm_object.boundaries[-1]
    else:
        min_colour_value = colour_norm_object.vmin
        max_colour_value = colour_norm_object.vmax

    axes_object.pcolormesh(
        edge_longitudes_deg_e, edge_latitudes_deg_n, edge_probability_matrix,
        cmap=colour_map_object, norm=colour_norm_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e11
    )

    row_indices, column_indices = numpy.where(target_matrix == 1)
    positive_latitudes_deg_n = latitudes_deg_n[row_indices]
    positive_longitudes_deg_e = longitudes_deg_e[column_indices]

    figure_width_px = figure_object.get_size_inches()[0] * figure_object.dpi
    target_marker_size_px = figure_width_px * (
        float(target_marker_size_grid_cells) / target_matrix.shape[1]
    )

    axes_object.plot(
        positive_longitudes_deg_e, positive_latitudes_deg_n,
        linestyle='None', marker=target_marker_type,
        markersize=target_marker_size_px, markeredgewidth=0,
        markerfacecolor=target_marker_colour,
        markeredgecolor=target_marker_colour
    )
