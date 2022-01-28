"""Makes figure to illustrate neighbourhood evaluation."""

import os
import argparse
from PIL import Image
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.patches
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils

DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'
MATCHING_DISTANCE_PX = 2.

TARGET_MATRIX = numpy.full((9, 11), 0, dtype=int)
TARGET_MATRIX[4, 3] = 1

PROBABILITY_MATRIX = numpy.array([
    [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00],
    [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.23, 0.21, 0.19, 0.17, 0.15],
    [0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.42, 0.39, 0.36, 0.33, 0.30],
    [0.45, 0.49, 0.53, 0.57, 0.61, 0.65, 0.61, 0.57, 0.53, 0.49, 0.45],
    [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60],
    [0.45, 0.49, 0.53, 0.57, 0.61, 0.65, 0.61, 0.57, 0.53, 0.49, 0.45],
    [0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.42, 0.39, 0.36, 0.33, 0.30],
    [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.23, 0.21, 0.19, 0.17, 0.15],
    [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00]
])

PROBABILITY_MATRIX = number_rounding.round_to_nearest(PROBABILITY_MATRIX, 0.1)

PREDICTION_ORIENTED_X_CENTERS = numpy.array([2, 9], dtype=float)
PREDICTION_ORIENTED_Y_CENTERS = numpy.array([3, 7], dtype=float)

CIRCLE_EDGE_COLOUR = numpy.full(3, 0.)
SMALL_CIRCLE_FACE_COLOUR = numpy.full(3, 0.)
LARGE_CIRCLE_FACE_COLOUR = matplotlib.colors.to_rgba(numpy.full(3, 1.), 0.)

TITLE_FONT_SIZE = 50
LABEL_FONT_SIZE = 75
GRID_LINE_WIDTH = 2.
GRID_LINE_COLOUR = numpy.full(3, 152. / 255)

FIGURE_WIDTH_INCHES = 15.
FIGURE_HEIGHT_INCHES = 15.
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
SQUARE_NEIGH_ARG_NAME = 'use_square_neigh'

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)
SQUARE_NEIGH_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use square (circular) neighbourhood.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SQUARE_NEIGH_ARG_NAME, type=int, required=False, default=0,
    help=SQUARE_NEIGH_HELP_STRING
)


def _get_prob_colour_scheme():
    """Returns colour scheme for probabilities.

    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm` or similar).
    :return: colour_norm_object: Normalizer for colour scheme (instance of
        `matplotlib.pyplot.Normalize` or similar).
    """

    main_colour_list = [
        numpy.array([35, 139, 69]), numpy.array([161, 217, 155]),
        numpy.array([8, 69, 148]), numpy.array([158, 202, 225]),
        numpy.array([74, 20, 134]), numpy.array([188, 189, 220]),
        numpy.array([153, 0, 13]), numpy.array([252, 146, 114])
    ]

    for i in range(len(main_colour_list)):
        main_colour_list[i] = main_colour_list[i].astype(float) / 255

    colour_map_object = matplotlib.colors.ListedColormap(main_colour_list)
    colour_map_object.set_under(numpy.full(3, 1.))
    colour_map_object.set_over(main_colour_list[-1])

    colour_bounds = numpy.linspace(0.1, 0.9, num=9)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        colour_bounds, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_one_panel(
        target_matrix, probability_matrix, use_square_neigh,
        observation_oriented, letter_label, title_string, output_file_name):
    """Plots one panel.

    M = number of rows in grid
    N = number of columns in grid

    :param target_matrix: M-by-N numpy array of integers in [0, 1].
    :param probability_matrix: M-by-N numpy array of forecast event
        probabilities.
    :param use_square_neigh: Boolean flag.  If True (False), will use square
        (circular) neighbourhood.
    :param observation_oriented: Boolean flag.  If True (False), matching
        observed to predicted (predicted to observed) event.
    :param letter_label: Letter label for panel.
    :param title_string: Figure title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    num_grid_rows = target_matrix.shape[0]
    num_grid_columns = target_matrix.shape[1]
    center_latitudes_deg_n = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float
    )
    center_longitudes_deg_e = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    colour_map_object, colour_norm_object = _get_prob_colour_scheme()

    radar_plotting.plot_latlng_grid(
        field_matrix=probability_matrix, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(center_latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(center_longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(center_latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(center_longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    edge_latitudes_deg_n, edge_longitudes_deg_e = (
        grids.get_latlng_grid_cell_edges(
            min_latitude_deg=center_latitudes_deg_n[0],
            min_longitude_deg=center_longitudes_deg_e[0],
            lat_spacing_deg=numpy.diff(center_latitudes_deg_n[:2])[0],
            lng_spacing_deg=numpy.diff(center_longitudes_deg_e[:2])[0],
            num_rows=num_grid_rows, num_columns=num_grid_columns
        )
    )

    y_center, x_center = numpy.where(target_matrix == 1)
    center_coords = (x_center - 0.5, y_center - 0.5)

    face_colour = matplotlib.colors.to_rgba(c=numpy.full(3, 1.), alpha=0)
    shape_object = matplotlib.patches.Rectangle(
        xy=center_coords, width=1, height=1, lw=0,
        ec=numpy.full(3, 0.), fc=face_colour, hatch='*'
    )
    axes_object.add_patch(shape_object)

    if observation_oriented:
        y_center, x_center = numpy.where(target_matrix == 1)

        if use_square_neigh:
            center_coords = (
                x_center - MATCHING_DISTANCE_PX - 0.5,
                y_center - MATCHING_DISTANCE_PX - 0.5
            )

            shape_object = matplotlib.patches.Rectangle(
                xy=center_coords, width=2 * MATCHING_DISTANCE_PX + 1,
                height=2 * MATCHING_DISTANCE_PX + 1,
                lw=4, ec=CIRCLE_EDGE_COLOUR, fc=LARGE_CIRCLE_FACE_COLOUR,
                zorder=1e12
            )
        else:
            shape_object = pyplot.Circle(
                xy=(x_center, y_center), radius=MATCHING_DISTANCE_PX, lw=2,
                ec=CIRCLE_EDGE_COLOUR, fc=LARGE_CIRCLE_FACE_COLOUR
            )

        axes_object.add_patch(shape_object)
    else:
        for x_center, y_center in zip(
                PREDICTION_ORIENTED_X_CENTERS, PREDICTION_ORIENTED_Y_CENTERS
        ):
            if use_square_neigh:
                center_coords = (
                    x_center - MATCHING_DISTANCE_PX - 0.5,
                    y_center - MATCHING_DISTANCE_PX - 0.5
                )

                shape_object = matplotlib.patches.Rectangle(
                    xy=center_coords, width=2 * MATCHING_DISTANCE_PX + 1,
                    height=2 * MATCHING_DISTANCE_PX + 1,
                    lw=4, ec=CIRCLE_EDGE_COLOUR, fc=LARGE_CIRCLE_FACE_COLOUR,
                    zorder=1e12
                )
            else:
                shape_object = pyplot.Circle(
                    xy=(x_center, y_center), radius=MATCHING_DISTANCE_PX, lw=2,
                    ec=CIRCLE_EDGE_COLOUR, fc=LARGE_CIRCLE_FACE_COLOUR
                )

            axes_object.add_patch(shape_object)

            shape_object = pyplot.Circle(
                xy=(x_center, y_center), radius=0.2, lw=2,
                ec=CIRCLE_EDGE_COLOUR, fc=SMALL_CIRCLE_FACE_COLOUR
            )
            axes_object.add_patch(shape_object)

    axes_object.set_xticks(edge_longitudes_deg_e)
    axes_object.set_yticks(edge_latitudes_deg_n)
    axes_object.grid(
        b=True, which='major', axis='both', linestyle='--',
        linewidth=GRID_LINE_WIDTH, color=GRID_LINE_COLOUR
    )

    axes_object.xaxis.set_ticklabels([])
    axes_object.yaxis.set_ticklabels([])
    axes_object.xaxis.set_ticks_position('none')
    axes_object.yaxis.set_ticks_position('none')

    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.resize_image(
        input_file_name=output_file_name, output_file_name=output_file_name,
        output_size_pixels=PANEL_SIZE_PX
    )


def _add_colour_bar(figure_file_name, colour_bar_file_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_bar_file_name: Path to output file.  Image with colour bar
        will be saved here.
    """

    colour_map_object, colour_norm_object = _get_prob_colour_scheme()

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=numpy.array([0.]),
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=1.25
    )
    colour_bar_object.set_label('Probability')

    tick_values = numpy.linspace(0.15, 0.85, num=8)
    tick_strings = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8']
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_figure_object.savefig(
        colour_bar_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, colour_bar_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(colour_bar_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def _run(output_dir_name, use_square_neigh):
    """Makes figure to illustrate neighbourhood evaluation.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    :param use_square_neigh: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    panel_file_names = [
        '{0:s}/observation_oriented.jpg'.format(output_dir_name)
    ]
    _plot_one_panel(
        target_matrix=TARGET_MATRIX, probability_matrix=PROBABILITY_MATRIX,
        use_square_neigh=use_square_neigh,
        observation_oriented=True, letter_label='a',
        title_string='Observation-oriented',
        output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/prediction_oriented.jpg'.format(output_dir_name)
    )
    _plot_one_panel(
        target_matrix=TARGET_MATRIX, probability_matrix=PROBABILITY_MATRIX,
        use_square_neigh=use_square_neigh,
        observation_oriented=False, letter_label='b',
        title_string='Prediction-oriented',
        output_file_name=panel_file_names[-1]
    )

    _add_colour_bar(
        figure_file_name=panel_file_names[-1],
        colour_bar_file_name=
        '{0:s}/prediction_oriented_cbar.jpg'.format(output_dir_name)
    )

    concat_figure_file_name = '{0:s}/neigh_eval_schematic.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=1, num_panel_columns=2
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
        use_square_neigh=bool(getattr(INPUT_ARG_OBJECT, SQUARE_NEIGH_ARG_NAME))
    )
