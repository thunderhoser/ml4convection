"""Makes figure to illustrate neighbourhood evaluation."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import radar_plotting
import plotting_utils

DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

FIRST_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

SECOND_MASK_MATRIX = numpy.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=int)

MATCHING_DISTANCE_PX = 4.

TARGET_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
PREDICTION_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
BACKGROUND_COLOUR = numpy.full(3, 1.)

COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap(
    [BACKGROUND_COLOUR, TARGET_COLOUR, PREDICTION_COLOUR]
)
COLOUR_MAP_OBJECT.set_under(BACKGROUND_COLOUR)
COLOUR_MAP_OBJECT.set_over(BACKGROUND_COLOUR)

COLOUR_BOUNDS = numpy.array([0, 1, 2], dtype=float) - 1.
COLOUR_NORM_OBJECT = matplotlib.colors.BoundaryNorm(
    boundaries=COLOUR_BOUNDS, ncolors=COLOUR_MAP_OBJECT.N
)

FIGURE_WIDTH_INCHES = 15.
FIGURE_HEIGHT_INCHES = 15.
FIGURE_RESOLUTION_DPI = 300

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(output_dir_name):
    """Makes figure to illustrate neighbourhood evaluation.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_grid_rows = FIRST_MASK_MATRIX.shape[0]
    num_grid_columns = FIRST_MASK_MATRIX.shape[1]
    dummy_latitudes_deg_n = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=float
    )
    dummy_longitudes_deg_e = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    radar_plotting.plot_latlng_grid(
        field_matrix=FIRST_MASK_MATRIX, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(dummy_latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(dummy_longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(dummy_latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(dummy_longitudes_deg_e[:2])[0],
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=COLOUR_NORM_OBJECT
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=dummy_latitudes_deg_n,
        plot_longitudes_deg_e=dummy_longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=1., meridian_spacing_deg=1.
    )

    axes_object.set_xticks([], [])
    axes_object.set_yticks([], [])

    output_file_name = '{0:s}/actual_oriented_true_positive.jpg'.format(
        output_dir_name
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
