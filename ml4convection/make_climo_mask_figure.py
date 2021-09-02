"""Makes figure with radar climatology and radar mask."""

import os
import sys
import pickle
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
import radar_plotting
import gg_plotting_utils
import imagemagick_utils
import radar_io
import border_io
import twb_satellite_io
import radar_utils
import plotting_utils
import prediction_plotting

DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

MASK_COLOUR_MAP_OBJECT = pyplot.get_cmap('winter')
MASK_COLOUR_NORM_OBJECT = pyplot.Normalize(vmin=0., vmax=1.)
MASK_OUTLINE_COLOUR = numpy.array([139, 69, 19], dtype=float) / 255
# MASK_OUTLINE_COLOUR = numpy.full(3, 152. / 255)

INNER_DOMAIN_HALF_WIDTH_PX = 52
COMPLETE_DOMAIN_HALF_WIDTH_PX = 102
STRIDE_LENGTH_PX = 25

INNER_DOMAIN_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
COMPLETE_DOMAIN_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DOMAIN_LINE_WIDTH = 3.

BORDER_COLOUR = numpy.full(3, 0.)
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

CLIMO_FILE_ARG_NAME = 'input_climo_file_name'
MASK_FILE_ARG_NAME = 'input_mask_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

CLIMO_FILE_HELP_STRING = (
    'Path to climatology file.  Will be read by `climatology_io.read_file`.'
)
MASK_FILE_HELP_STRING = (
    'Path to mask file.  Will be read by `radar_io.read_mask_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + CLIMO_FILE_ARG_NAME, type=str, required=True,
    help=CLIMO_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MASK_FILE_ARG_NAME, type=str, required=True,
    help=MASK_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_climo(
        event_freq_matrix, mask_dict, border_latitudes_deg_n,
        border_longitudes_deg_e, letter_label, output_file_name):
    """Plots climatology (convection frequency at each point).

    M = number of rows in grid
    N = number of columns in grid

    :param event_freq_matrix: M-by-N numpy array of frequencies.
    :param mask_dict: See doc for `_plot_mask`.
    :param border_latitudes_deg_n: Same.
    :param border_longitudes_deg_e: Same.
    :param letter_label: Same.
    :param output_file_name: Same.
    """

    latitudes_deg_n = mask_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = mask_dict[radar_io.LONGITUDES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_colour=BORDER_COLOUR
    )

    dummy_target_matrix = numpy.full(event_freq_matrix.shape, 0, dtype=int)
    max_colour_value = numpy.nanmax(event_freq_matrix)

    colour_map_object, colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(
            max_probability=max_colour_value, make_lowest_prob_grey=False
        )
    )

    prediction_plotting.plot_probabilistic(
        probability_matrix=event_freq_matrix, target_matrix=dummy_target_matrix,
        figure_object=figure_object, axes_object=axes_object,
        min_latitude_deg_n=latitudes_deg_n[0],
        min_longitude_deg_e=longitudes_deg_e[0],
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=event_freq_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.4f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    pyplot.contour(
        longitudes_deg_e, latitudes_deg_n,
        mask_dict[radar_io.MASK_MATRIX_KEY].astype(int),
        numpy.array([0.999]),
        colors=(MASK_OUTLINE_COLOUR,), linewidths=4, linestyles='solid',
        axes=axes_object
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )

    axes_object.set_title('Convection frequency from 2016-2018')
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_mask(mask_dict, border_latitudes_deg_n, border_longitudes_deg_e,
               letter_label, output_file_name):
    """Plots radar mask.

    P = number of points in border set

    :param mask_dict: Dictionary returned by `radar_io.read_mask_file`.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param letter_label: Letter label.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    latitudes_deg_n = mask_dict[radar_io.LATITUDES_KEY]
    longitudes_deg_e = mask_dict[radar_io.LONGITUDES_KEY]

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_colour=BORDER_COLOUR
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
        colour_map_object=MASK_COLOUR_MAP_OBJECT,
        colour_norm_object=MASK_COLOUR_NORM_OBJECT
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )

    this_index = numpy.argmin(radar_utils.RADAR_LATITUDES_DEG_N)
    radar_latitude_deg_n = radar_utils.RADAR_LATITUDES_DEG_N[this_index]
    radar_longitude_deg_e = radar_utils.RADAR_LONGITUDES_DEG_E[this_index]

    radar_row = numpy.argmin(numpy.absolute(
        radar_latitude_deg_n - latitudes_deg_n
    ))
    radar_column = numpy.argmin(numpy.absolute(
        radar_longitude_deg_e - longitudes_deg_e
    ))

    inner_polygon_rows = numpy.array([
        radar_row - INNER_DOMAIN_HALF_WIDTH_PX,
        radar_row - INNER_DOMAIN_HALF_WIDTH_PX,
        radar_row + INNER_DOMAIN_HALF_WIDTH_PX,
        radar_row + INNER_DOMAIN_HALF_WIDTH_PX,
        radar_row - INNER_DOMAIN_HALF_WIDTH_PX
    ], dtype=int)

    complete_polygon_rows = numpy.array([
        radar_row - COMPLETE_DOMAIN_HALF_WIDTH_PX,
        radar_row - COMPLETE_DOMAIN_HALF_WIDTH_PX,
        radar_row + COMPLETE_DOMAIN_HALF_WIDTH_PX,
        radar_row + COMPLETE_DOMAIN_HALF_WIDTH_PX,
        radar_row - COMPLETE_DOMAIN_HALF_WIDTH_PX
    ], dtype=int)

    inner_polygon_columns = numpy.array([
        radar_column - INNER_DOMAIN_HALF_WIDTH_PX,
        radar_column + INNER_DOMAIN_HALF_WIDTH_PX,
        radar_column + INNER_DOMAIN_HALF_WIDTH_PX,
        radar_column - INNER_DOMAIN_HALF_WIDTH_PX,
        radar_column - INNER_DOMAIN_HALF_WIDTH_PX
    ], dtype=int)

    complete_polygon_columns = numpy.array([
        radar_column - COMPLETE_DOMAIN_HALF_WIDTH_PX,
        radar_column + COMPLETE_DOMAIN_HALF_WIDTH_PX,
        radar_column + COMPLETE_DOMAIN_HALF_WIDTH_PX,
        radar_column - COMPLETE_DOMAIN_HALF_WIDTH_PX,
        radar_column - COMPLETE_DOMAIN_HALF_WIDTH_PX
    ], dtype=int)

    axes_object.plot(
        longitudes_deg_e[inner_polygon_columns],
        latitudes_deg_n[inner_polygon_rows],
        color=INNER_DOMAIN_COLOUR, linestyle='solid',
        linewidth=DOMAIN_LINE_WIDTH
    )

    axes_object.plot(
        longitudes_deg_e[complete_polygon_columns],
        latitudes_deg_n[complete_polygon_rows],
        color=COMPLETE_DOMAIN_COLOUR, linestyle='solid',
        linewidth=DOMAIN_LINE_WIDTH
    )

    axes_object.set_title('Radar mask (100-km radius)')
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_sliding_windows(
        border_latitudes_deg_n, border_longitudes_deg_e, letter_label,
        output_file_name):
    """Plots sliding windows used for prediction on full grid.

    :param border_latitudes_deg_n: See doc for `_plot_mask`.
    :param border_longitudes_deg_e: Same.
    :param letter_label: Same.
    :param output_file_name: Same.
    """

    latitudes_deg_n = twb_satellite_io.GRID_LATITUDES_DEG_N
    longitudes_deg_e = twb_satellite_io.GRID_LONGITUDES_DEG_E

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_colour=BORDER_COLOUR
    )

    center_row = len(latitudes_deg_n) - 1 - COMPLETE_DOMAIN_HALF_WIDTH_PX
    center_column = COMPLETE_DOMAIN_HALF_WIDTH_PX - STRIDE_LENGTH_PX

    for _ in range(5):
        center_column += STRIDE_LENGTH_PX

        inner_polygon_rows = numpy.array([
            center_row - INNER_DOMAIN_HALF_WIDTH_PX,
            center_row - INNER_DOMAIN_HALF_WIDTH_PX,
            center_row + INNER_DOMAIN_HALF_WIDTH_PX,
            center_row + INNER_DOMAIN_HALF_WIDTH_PX,
            center_row - INNER_DOMAIN_HALF_WIDTH_PX
        ], dtype=int)

        complete_polygon_rows = numpy.array([
            center_row - COMPLETE_DOMAIN_HALF_WIDTH_PX,
            center_row - COMPLETE_DOMAIN_HALF_WIDTH_PX,
            center_row + COMPLETE_DOMAIN_HALF_WIDTH_PX,
            center_row + COMPLETE_DOMAIN_HALF_WIDTH_PX,
            center_row - COMPLETE_DOMAIN_HALF_WIDTH_PX
        ], dtype=int)

        inner_polygon_columns = numpy.array([
            center_column - INNER_DOMAIN_HALF_WIDTH_PX,
            center_column + INNER_DOMAIN_HALF_WIDTH_PX,
            center_column + INNER_DOMAIN_HALF_WIDTH_PX,
            center_column - INNER_DOMAIN_HALF_WIDTH_PX,
            center_column - INNER_DOMAIN_HALF_WIDTH_PX
        ], dtype=int)

        complete_polygon_columns = numpy.array([
            center_column - COMPLETE_DOMAIN_HALF_WIDTH_PX,
            center_column + COMPLETE_DOMAIN_HALF_WIDTH_PX,
            center_column + COMPLETE_DOMAIN_HALF_WIDTH_PX,
            center_column - COMPLETE_DOMAIN_HALF_WIDTH_PX,
            center_column - COMPLETE_DOMAIN_HALF_WIDTH_PX
        ], dtype=int)

        axes_object.plot(
            longitudes_deg_e[inner_polygon_columns],
            latitudes_deg_n[inner_polygon_rows],
            color=INNER_DOMAIN_COLOUR, linestyle='solid',
            linewidth=DOMAIN_LINE_WIDTH
        )

        axes_object.plot(
            longitudes_deg_e[complete_polygon_columns],
            latitudes_deg_n[complete_polygon_rows],
            color=COMPLETE_DOMAIN_COLOUR, linestyle='solid',
            linewidth=DOMAIN_LINE_WIDTH
        )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=latitudes_deg_n,
        plot_longitudes_deg_e=longitudes_deg_e, axes_object=axes_object,
        parallel_spacing_deg=2., meridian_spacing_deg=2.
    )

    axes_object.set_title('Sliding-window approach')
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(climo_file_name, mask_file_name, output_dir_name):
    """Makes figure with radar climatology and radar mask.

    This is effectively the main method.

    :param climo_file_name: See documentation at top of file.
    :param mask_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    print('Reading data from: "{0:s}"...'.format(mask_file_name))
    mask_dict = radar_io.read_mask_file(mask_file_name)

    print('Reading data from: "{0:s}"...'.format(climo_file_name))
    climo_file_handle = open(climo_file_name, 'rb')
    event_freq_matrix = pickle.load(climo_file_handle)
    climo_file_handle.close()

    # climo_figure_file_name = '{0:s}/convection_frequency_climo.jpg'.format(
    #     output_dir_name
    # )
    # _plot_climo(
    #     event_freq_matrix=event_freq_matrix, mask_dict=mask_dict,
    #     border_latitudes_deg_n=border_latitudes_deg_n,
    #     border_longitudes_deg_e=border_longitudes_deg_e,
    #     letter_label='a', output_file_name=climo_figure_file_name
    # )

    mask_figure_file_name = '{0:s}/radar_mask.jpg'.format(output_dir_name)
    _plot_mask(
        mask_dict=mask_dict, border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        letter_label='a', output_file_name=mask_figure_file_name
    )

    window_figure_file_name = '{0:s}/sliding_windows.jpg'.format(
        output_dir_name
    )
    _plot_sliding_windows(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        letter_label='b', output_file_name=window_figure_file_name
    )

    concat_figure_file_name = '{0:s}/climo_and_mask.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    panel_file_names = [
        mask_figure_file_name, window_figure_file_name
    ]

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
        climo_file_name=getattr(INPUT_ARG_OBJECT, CLIMO_FILE_ARG_NAME),
        mask_file_name=getattr(INPUT_ARG_OBJECT, MASK_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
