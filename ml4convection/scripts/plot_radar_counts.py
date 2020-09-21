"""Plots observation counts for radar data (one figure per height)."""

import copy
import glob
import argparse
import numpy
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import radar_plotting
from ml4convection.io import radar_io
from ml4convection.scripts import count_radar_observations as count_obs

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

COUNT_DIR_ARG_NAME = 'input_count_dir_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
PLOT_BASEMAP_ARG_NAME = 'plot_basemap'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

COUNT_DIR_HELP_STRING = (
    'Name of directory with count files.  Names must be in format "counts*.nc".'
    '  Files will be read by `count_radar_observations.read_count_file`.'
)
COLOUR_MAP_HELP_STRING = (
    'Name of colour map (must be accepted by `matplotlib.pyplot.get_cmap`).'
)
PLOT_BASEMAP_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot image with (without) basemap.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + COUNT_DIR_ARG_NAME, type=str, required=True,
    help=COUNT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False,
    default='twilight_shifted', help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_BASEMAP_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_BASEMAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _increment_counts_one_file(count_file_name, count_matrix):
    """Increments counts, based on one count file.

    M = number of rows in grid
    N = number of columns in grid
    H = number of heights in grid

    :param count_file_name: Path to input file (will be read by
        `count_radar_observations.read_count_file`).
    :param count_matrix: M-by-N-by-H numpy array of observation counts.
        If None, will be created on the fly.
    :return: count_matrix: Same as input but with new (incremented) values.
    :return: metadata_dict: Dictionary returned by
        `count_radar_observations.read_count_file`, excluding counts.
    """

    print('Reading data from: "{0:s}"...'.format(count_file_name))
    count_dict = count_obs.read_count_file(count_file_name)

    new_count_matrix = count_dict[count_obs.OBSERVATION_COUNT_KEY]
    del count_dict[count_obs.OBSERVATION_COUNT_KEY]
    metadata_dict = copy.deepcopy(count_dict)

    if count_matrix is None:
        count_matrix = new_count_matrix + 0
    else:
        count_matrix += new_count_matrix

    return count_matrix, metadata_dict


def _plot_counts_one_height(
        count_matrix, metadata_dict, height_index, colour_map_object,
        colour_norm_object, plot_basemap, output_dir_name):
    """Plots observation counts at one height.

    M = number of rows in grid
    N = number of columns in grid

    :param count_matrix: M-by-N numpy array of observation counts.
    :param metadata_dict: Dictionary created by `_increment_counts_one_file`.
    :param height_index: Will plot the [k]th height, where k = `height_index`.
    :param colour_map_object: Colour map.
    :param colour_norm_object: Normalizer for colour map.
    :param plot_basemap: See documentation at top of file.
    :param output_dir_name: Same.
    """

    latitudes_deg_n = metadata_dict[count_obs.LATITUDES_KEY]
    longitudes_deg_e = metadata_dict[count_obs.LONGITUDES_KEY]

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

    matrix_to_plot = count_matrix.astype(float)
    matrix_to_plot[matrix_to_plot < 0.5] = numpy.nan

    radar_plotting.plot_latlng_grid(
        field_matrix=matrix_to_plot, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
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

    plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=True
    )

    height_m_asl = metadata_dict[count_obs.HEIGHTS_KEY][height_index]
    title_string = 'Observation counts at {0:d} m AGL'.format(
        int(numpy.round(height_m_asl))
    )
    axes_object.set_title(title_string)

    output_file_name = '{0:s}/observation_counts_{1:05d}-metres-agl.jpg'.format(
        output_dir_name, int(numpy.round(height_m_asl))
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(count_dir_name, colour_map_name, plot_basemap, output_dir_name):
    """Plots observation counts for radar data (one figure per height).

    This is effectively the main method.

    :param count_dir_name: See documentation at top of file.
    :param colour_map_name: Same.
    :param plot_basemap: Same.
    :param output_dir_name: Same.
    """

    colour_map_object = pyplot.get_cmap(colour_map_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    count_file_names = glob.glob('{0:s}/counts*.nc'.format(count_dir_name))
    count_file_names.sort()

    count_matrix = None
    metadata_dict = None

    for this_file_name in count_file_names:
        count_matrix, metadata_dict = _increment_counts_one_file(
            count_file_name=this_file_name, count_matrix=count_matrix
        )

    print(SEPARATOR_STRING)

    heights_m_asl = metadata_dict[radar_io.HEIGHTS_KEY]
    num_heights = len(heights_m_asl)

    max_colour_value = numpy.percentile(count_matrix, 99.)
    colour_norm_object = pyplot.Normalize(vmin=0., vmax=max_colour_value)

    for k in range(num_heights):
        _plot_counts_one_height(
            count_matrix=count_matrix[..., k], metadata_dict=metadata_dict,
            height_index=k, colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object, plot_basemap=plot_basemap,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        count_dir_name=getattr(INPUT_ARG_OBJECT, COUNT_DIR_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        plot_basemap=bool(getattr(INPUT_ARG_OBJECT, PLOT_BASEMAP_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
