"""Makes figure to illustrate neighbourhood evaluation."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.patches
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils

DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

ACTUAL_ENUM = 1
FORECAST_ENUM = 2
MATCHING_DISTANCE_PX = 4.

GOOD_MASK_MATRIX = numpy.full((8, 10), 0, dtype=int)
GOOD_MASK_MATRIX[4, 3] = ACTUAL_ENUM
GOOD_MASK_MATRIX[5, 4] = FORECAST_ENUM

BAD_MASK_MATRIX = numpy.full((8, 10), 0, dtype=int)
BAD_MASK_MATRIX[4, 3] = ACTUAL_ENUM
BAD_MASK_MATRIX[6, 7] = FORECAST_ENUM

ACTUAL_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
FORECAST_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
BACKGROUND_COLOUR = numpy.full(3, 1.)

CIRCLE_EDGE_COLOUR = numpy.full(3, 0.)
SMALL_CIRCLE_FACE_COLOUR = numpy.full(3, 0.)
LARGE_CIRCLE_FACE_COLOUR = matplotlib.colors.to_rgba(numpy.full(3, 1.), 0.)

COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap(
    [ACTUAL_COLOUR, FORECAST_COLOUR, FORECAST_COLOUR]
)
COLOUR_MAP_OBJECT.set_under(BACKGROUND_COLOUR)
COLOUR_MAP_OBJECT.set_over(FORECAST_COLOUR)

COLOUR_BOUNDS = numpy.array([
    ACTUAL_ENUM - 0.5, FORECAST_ENUM - 0.5, FORECAST_ENUM
])
COLOUR_NORM_OBJECT = matplotlib.colors.BoundaryNorm(
    boundaries=COLOUR_BOUNDS, ncolors=COLOUR_MAP_OBJECT.N
)

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


def _plot_one_panel(
        mask_matrix, use_square_neigh, actual_to_forecast, letter_label,
        title_string, output_file_name):
    """Plots one panel.

    M = number of rows in grid
    N = number of columns in grid

    :param mask_matrix: M-by-N numpy array of integers.
    :param use_square_neigh: Boolean flag.  If True (False), will use square
        (circular) neighbourhood.
    :param actual_to_forecast: Boolean flag.  If True (False), matching actual
        to forecast (forecast to actual).
    :param letter_label: Letter label for panel.
    :param title_string: Figure title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    num_grid_rows = mask_matrix.shape[0]
    num_grid_columns = mask_matrix.shape[1]
    center_latitudes_deg_n = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float
    )
    center_longitudes_deg_e = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    radar_plotting.plot_latlng_grid(
        field_matrix=mask_matrix, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(center_latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(center_longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(center_latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(center_longitudes_deg_e[:2])[0],
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=COLOUR_NORM_OBJECT
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

    y_forecast, x_forecast = numpy.where(mask_matrix == FORECAST_ENUM)
    y_actual, x_actual = numpy.where(mask_matrix == ACTUAL_ENUM)

    if actual_to_forecast:
        if use_square_neigh:
            center_coords = (
                x_actual - MATCHING_DISTANCE_PX, y_actual - MATCHING_DISTANCE_PX
            )

            shape_object = matplotlib.patches.Rectangle(
                xy=center_coords, width=2 * MATCHING_DISTANCE_PX,
                height=2 * MATCHING_DISTANCE_PX,
                lw=2, ec=CIRCLE_EDGE_COLOUR, fc=LARGE_CIRCLE_FACE_COLOUR
            )
        else:
            shape_object = pyplot.Circle(
                xy=(x_actual, y_actual), radius=MATCHING_DISTANCE_PX, lw=2,
                ec=CIRCLE_EDGE_COLOUR, fc=LARGE_CIRCLE_FACE_COLOUR
            )
    else:
        if use_square_neigh:
            center_coords = (
                x_forecast - MATCHING_DISTANCE_PX,
                y_forecast - MATCHING_DISTANCE_PX
            )

            shape_object = matplotlib.patches.Rectangle(
                xy=center_coords, width=2 * MATCHING_DISTANCE_PX,
                height=2 * MATCHING_DISTANCE_PX,
                lw=2, ec=CIRCLE_EDGE_COLOUR, fc=LARGE_CIRCLE_FACE_COLOUR
            )
        else:
            shape_object = pyplot.Circle(
                xy=(x_forecast, y_forecast), radius=MATCHING_DISTANCE_PX, lw=2,
                ec=CIRCLE_EDGE_COLOUR, fc=LARGE_CIRCLE_FACE_COLOUR
            )

    axes_object.add_patch(shape_object)

    shape_object = pyplot.Circle(
        xy=(x_forecast, y_forecast), radius=0.1, lw=0, ec=CIRCLE_EDGE_COLOUR,
        fc=SMALL_CIRCLE_FACE_COLOUR
    )
    axes_object.add_patch(shape_object)

    shape_object = pyplot.Circle(
        xy=(x_actual, y_actual), radius=0.1, lw=0, ec=CIRCLE_EDGE_COLOUR,
        fc=SMALL_CIRCLE_FACE_COLOUR
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

    axes_object.set_title(title_string)
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
        '{0:s}/observation_oriented_true_positive.jpg'.format(output_dir_name)
    ]
    _plot_one_panel(
        mask_matrix=GOOD_MASK_MATRIX, use_square_neigh=use_square_neigh,
        actual_to_forecast=True, letter_label='a',
        title_string='{0:s}-oriented true positive'.format(
            'Observation' if use_square_neigh else 'Actual'
        ),
        output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/false_negative.jpg'.format(output_dir_name)
    )
    _plot_one_panel(
        mask_matrix=BAD_MASK_MATRIX, use_square_neigh=use_square_neigh,
        actual_to_forecast=True, letter_label='b',
        title_string='False negative',
        output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/prediction_oriented_true_positive.jpg'.format(output_dir_name)
    )
    _plot_one_panel(
        mask_matrix=GOOD_MASK_MATRIX, use_square_neigh=use_square_neigh,
        actual_to_forecast=False, letter_label='c',
        title_string='{0:s}-oriented true positive'.format(
            'Prediction' if use_square_neigh else 'Forecast'
        ),
        output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/false_positive.jpg'.format(output_dir_name)
    )
    _plot_one_panel(
        mask_matrix=BAD_MASK_MATRIX, use_square_neigh=use_square_neigh,
        actual_to_forecast=False, letter_label='d',
        title_string='False positive',
        output_file_name=panel_file_names[-1]
    )

    concat_figure_file_name = '{0:s}/neigh_eval_schematic.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=2
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
