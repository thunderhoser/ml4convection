"""Plots mask for radar data."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import radar_plotting
from ml4convection.io import radar_io
from ml4convection.io import border_io
from ml4convection.plotting import plotting_utils

DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'
COLOUR_MAP_OBJECT = pyplot.get_cmap('winter')
COLOUR_NORM_OBJECT = pyplot.Normalize(vmin=0., vmax=1.)

TITLE_FONT_SIZE = 16

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INPUT_FILE_ARG_NAME = 'input_mask_file_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `radar_io.read_mask_file`.'
)
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Image will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(mask_file_name, output_file_name):
    """Plots mask for radar data.

    This is effectively the main method.

    :param mask_file_name: See documentation at top of file.
    :param output_file_name: Same.
    """

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Reading data from: "{0:s}"...'.format(mask_file_name))
    mask_dict = radar_io.read_mask_file(mask_file_name)

    latitudes_deg_n = mask_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = mask_dict[radar_io.LONGITUDES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    mask_matrix = mask_dict[radar_io.MASK_MATRIX_KEY].astype(float)
    mask_matrix[mask_matrix < 0.5] = numpy.nan

    radar_plotting.plot_latlng_grid(
        field_matrix=mask_matrix, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=COLOUR_NORM_OBJECT
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )

    title_string = (
        'Mask (grid columns with >= {0:d} obs at >= {1:.1f}% of heights up to '
        '{2:d} m ASL)'
    ).format(
        mask_dict[radar_io.MIN_OBSERVATIONS_KEY],
        100 * mask_dict[radar_io.MIN_HEIGHT_FRACTION_FOR_MASK_KEY],
        int(numpy.round(mask_dict[radar_io.MAX_MASK_HEIGHT_KEY]))
    )
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        mask_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
