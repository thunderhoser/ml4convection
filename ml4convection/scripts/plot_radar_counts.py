"""Plots observation counts for radar data (one figure per height)."""

import copy
import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import radar_plotting
from ml4convection.io import radar_io
from ml4convection.io import border_io
from ml4convection.plotting import plotting_utils
from ml4convection.scripts import count_radar_observations as count_obs

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

COUNT_DIR_ARG_NAME = 'input_count_dir_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

COUNT_DIR_HELP_STRING = (
    'Name of directory with count files.  Names must be in format "counts*.nc".'
    '  Files will be read by `count_radar_observations.read_count_file`.'
)
COLOUR_MAP_HELP_STRING = (
    'Name of colour map (must be accepted by `matplotlib.pyplot.get_cmap`).'
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
        colour_norm_object, border_latitudes_deg_n, border_longitudes_deg_e,
        output_dir_name):
    """Plots observation counts at one height.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border set

    :param count_matrix: M-by-N numpy array of observation counts.
    :param metadata_dict: Dictionary created by `_increment_counts_one_file`.
    :param height_index: Will plot the [k]th height, where k = `height_index`.
    :param colour_map_object: Colour map.
    :param colour_norm_object: Normalizer for colour map.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param output_dir_name: Name of output directory.  Figure will be saved
        here.
    """

    latitudes_deg_n = metadata_dict[count_obs.LATITUDES_KEY]
    longitudes_deg_e = metadata_dict[count_obs.LONGITUDES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
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

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=matrix_to_plot,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=True
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
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


def _run(count_dir_name, colour_map_name, output_dir_name):
    """Plots observation counts for radar data (one figure per height).

    This is effectively the main method.

    :param count_dir_name: See documentation at top of file.
    :param colour_map_name: Same.
    :param output_dir_name: Same.
    """

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

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
            colour_norm_object=colour_norm_object,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        count_dir_name=getattr(INPUT_ARG_OBJECT, COUNT_DIR_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
