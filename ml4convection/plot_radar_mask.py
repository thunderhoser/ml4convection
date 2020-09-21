"""Plots mask for radar data."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import plotting_utils
import radar_plotting
import radar_io

DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'
COLOUR_MAP_OBJECT = pyplot.get_cmap('winter')
COLOUR_NORM_OBJECT = pyplot.Normalize(vmin=0., vmax=1.)

TITLE_FONT_SIZE = 16

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

INPUT_FILE_ARG_NAME = 'input_mask_file_name'
PLOT_BASEMAP_ARG_NAME = 'plot_basemap'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `radar_io.read_mask_file`.'
)
PLOT_BASEMAP_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot image with (without) basemap.'
)
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Image will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_BASEMAP_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_BASEMAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(mask_file_name, plot_basemap, output_file_name):
    """Plots mask for radar data.

    This is effectively the main method.

    :param mask_file_name: See documentation at top of file.
    :param plot_basemap: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Reading data from: "{0:s}"...'.format(mask_file_name))
    mask_dict = radar_io.read_mask_file(mask_file_name)

    latitudes_deg_n = mask_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = mask_dict[radar_io.LONGITUDES_KEY]

    if plot_basemap:
        figure_object, axes_object, basemap_object = (
            plotting_utils.create_equidist_cylindrical_map(
                min_latitude_deg=numpy.min(latitudes_deg_n),
                max_latitude_deg=numpy.max(latitudes_deg_n),
                min_longitude_deg=numpy.min(longitudes_deg_e),
                max_longitude_deg=numpy.max(longitudes_deg_e),
                resolution_string='i'
            )
        )

        plotting_utils.plot_coastlines(
            basemap_object=basemap_object, axes_object=axes_object,
            line_colour=plotting_utils.DEFAULT_COUNTRY_COLOUR
        )
        plotting_utils.plot_countries(
            basemap_object=basemap_object, axes_object=axes_object
        )
        plotting_utils.plot_states_and_provinces(
            basemap_object=basemap_object, axes_object=axes_object
        )
        plotting_utils.plot_parallels(
            basemap_object=basemap_object, axes_object=axes_object,
            num_parallels=NUM_PARALLELS
        )
        plotting_utils.plot_meridians(
            basemap_object=basemap_object, axes_object=axes_object,
            num_meridians=NUM_MERIDIANS
        )
    else:
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
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

    if not plot_basemap:
        tick_latitudes_deg_n = numpy.unique(numpy.round(latitudes_deg_n))
        tick_latitudes_deg_n = tick_latitudes_deg_n[
            tick_latitudes_deg_n >= numpy.min(latitudes_deg_n)
        ]
        tick_latitudes_deg_n = tick_latitudes_deg_n[
            tick_latitudes_deg_n <= numpy.max(latitudes_deg_n)
        ]

        tick_longitudes_deg_e = numpy.unique(numpy.round(longitudes_deg_e))
        tick_longitudes_deg_e = tick_longitudes_deg_e[
            tick_longitudes_deg_e >= numpy.min(longitudes_deg_e)
        ]
        tick_longitudes_deg_e = tick_longitudes_deg_e[
            tick_longitudes_deg_e <= numpy.max(longitudes_deg_e)
        ]

        axes_object.set_xticks(tick_longitudes_deg_e)
        axes_object.set_yticks(tick_latitudes_deg_n)
        axes_object.grid(
            b=True, which='major', axis='both', linestyle='--', linewidth=2
        )

        axes_object.set_xlabel(r'Longitude ($^{\circ}$E)')
        axes_object.set_ylabel(r'Latitude ($^{\circ}$N)')

    title_string = (
        'Mask (grid columns with >= {0:d} obs at >= {1:.1f}% of heights up to '
        '{2:d} m ASL)'
    ).format(
        mask_dict[radar_io.MIN_OBSERVATIONS_KEY],
        100 * mask_dict[radar_io.MIN_HEIGHT_FRACTION_KEY],
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
        plot_basemap=bool(getattr(INPUT_ARG_OBJECT, PLOT_BASEMAP_ARG_NAME)),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
