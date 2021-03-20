"""Plots saliency maps."""

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

import general_utils
import time_conversion
import file_system_utils
import error_checking
import imagemagick_utils
import gg_plotting_utils
import border_io
import saliency
import neural_net
import saliency_plotting
import satellite_plotting
import plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DATE_FORMAT = neural_net.DATE_FORMAT
TIME_FORMAT_FOR_FILES = '%Y-%m-%d-%H%M'
SECONDS_TO_MINUTES = 1. / 60

MARKER_SIZE_GRID_CELLS = 4.
MARKER_TYPE = '*'
MARKER_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15.
FIGURE_HEIGHT_INCHES = 15.
FIGURE_RESOLUTION_DPI = 300

PANEL_SIZE_PX = int(1e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FONT_SIZE = 50
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
LAG_TIMES_ARG_NAME = 'lag_times_sec'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MIN_CONTOUR_VALUE_ARG_NAME = 'min_abs_contour_value'
MAX_CONTOUR_VALUE_ARG_NAME = 'max_abs_contour_value'
HALF_NUM_CONTOURS_ARG_NAME = 'half_num_contours'
LINE_WIDTH_ARG_NAME = 'line_width'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file.  Will be read by `saliency.read_file`.'
)
PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors (brightness temperatures to '
    'plot under saliency contours).  Files therein will be found by '
    '`example_io.find_predictor_file` and read by '
    '`example_io.read_predictor_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets to use.  Files therein will be '
    'found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
LAG_TIMES_HELP_STRING = (
    'Will plot saliency maps for only these lag times (seconds).  To plot '
    'saliency maps for all lag times, leave this argument alone.'
)
BAND_NUMBERS_HELP_STRING = (
    'Will plot saliency maps for only these spectral bands (integers).  To plot'
    ' saliency maps for all bands, leave this argument alone.'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num pixels).  If you do not want '
    'to smooth saliency maps, leave this alone.'
)
COLOUR_MAP_HELP_STRING = (
    'Colour scheme for saliency.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
MIN_CONTOUR_VALUE_HELP_STRING = 'Minimum absolute saliency value for contours.'
MAX_CONTOUR_VALUE_HELP_STRING = (
    'Max absolute saliency value for contours.  Leave this alone if you want '
    'max value to be determined automatically.'
)
HALF_NUM_CONTOURS_HELP_STRING = (
    'Number of saliency contours on either side of zero.'
)
LINE_WIDTH_HELP_STRING = 'Line width for saliency contours.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILE_ARG_NAME, type=str, required=True,
    help=SALIENCY_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTOR_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_DIR_ARG_NAME, type=str, required=True,
    help=TARGET_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAG_TIMES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=LAG_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BAND_NUMBERS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=BAND_NUMBERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SMOOTHING_RADIUS_ARG_NAME, type=float, required=False,
    default=2., help=SMOOTHING_RADIUS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='BuGn',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_CONTOUR_VALUE_ARG_NAME, type=float, required=False,
    default=0.001, help=MIN_CONTOUR_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_CONTOUR_VALUE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_CONTOUR_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HALF_NUM_CONTOURS_ARG_NAME, type=int, required=False, default=10,
    help=HALF_NUM_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LINE_WIDTH_ARG_NAME, type=float, required=False,
    default=saliency_plotting.DEFAULT_CONTOUR_WIDTH, help=LINE_WIDTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _smooth_maps(saliency_dict, smoothing_radius_px):
    """Smooths saliency maps via Gaussian filter.

    :param saliency_dict: Dictionary returned by `saliency.read_file`.
    :param smoothing_radius_px: e-folding radius (num pixels).
    :return: saliency_dict: Same as input but with smoothed maps.
    """

    print((
        'Smoothing saliency maps with Gaussian filter (e-folding radius of '
        '{0:.1f} grid cells)...'
    ).format(
        smoothing_radius_px
    ))

    saliency_matrix = saliency_dict[saliency.SALIENCY_MATRIX_KEY]
    num_examples = saliency_matrix.shape[0]
    num_lag_times = saliency_matrix.shape[-2]
    num_channels = saliency_matrix.shape[-1]

    for i in range(num_examples):
        for j in range(num_lag_times):
            for k in range(num_channels):
                saliency_matrix[i, ..., j, k] = (
                    gg_general_utils.apply_gaussian_filter(
                        input_matrix=saliency_matrix[i, ..., j, k],
                        e_folding_radius_grid_cells=smoothing_radius_px
                    )
                )

    saliency_dict[saliency.SALIENCY_MATRIX_KEY] = saliency_matrix
    return saliency_dict


def _plot_predictors_one_example(
        predictor_dict, valid_time_unix_sec, border_latitudes_deg_n,
        border_longitudes_deg_e, predictor_option_dict):
    """Plots predictors (brightness temperatures) for one example.

    P = number of points in border set
    T = number of lag times
    C = number of channels

    :param predictor_dict: One dictionary returned by
        `neural_net.create_data_partial_grids`.
    :param valid_time_unix_sec: Will plot predictors for this valid time.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param predictor_option_dict: See input doc for
        `neural_net.create_data_partial_grids`.
    :return: figure_object: T-by-C numpy array of figure handles (instances of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: T-by-C numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    example_index = numpy.where(
        predictor_dict[neural_net.VALID_TIMES_KEY] == valid_time_unix_sec
    )[0][0]

    band_numbers = predictor_option_dict[neural_net.BAND_NUMBERS_KEY]
    lag_times_seconds = predictor_option_dict[neural_net.LAG_TIMES_KEY]
    lag_times_minutes = numpy.round(
        lag_times_seconds.astype(float) * SECONDS_TO_MINUTES
    ).astype(int)

    latitudes_deg_n = predictor_dict[neural_net.LATITUDES_KEY]
    longitudes_deg_e = predictor_dict[neural_net.LONGITUDES_KEY]

    brightness_temp_matrix_kelvins = (
        predictor_dict[neural_net.PREDICTOR_MATRIX_KEY][example_index, ...]
    )

    num_lag_times = brightness_temp_matrix_kelvins.shape[-2]
    num_channels = brightness_temp_matrix_kelvins.shape[-1]
    figure_object_matrix = numpy.full(
        (num_lag_times, num_channels), '', dtype=object
    )
    axes_object_matrix = numpy.full(
        (num_lag_times, num_channels), '', dtype=object
    )

    for j in range(num_lag_times):
        for k in range(num_channels):
            figure_object_matrix[j, k], axes_object_matrix[j, k] = (
                pyplot.subplots(
                    1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
                )
            )

            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object_matrix[j, k]
            )

            if j == num_lag_times - 1 and k == 0:
                cbar_orientation_string = 'horizontal'
            else:
                cbar_orientation_string = None

            colour_bar_object = satellite_plotting.plot_2d_grid_latlng(
                brightness_temp_matrix_kelvins=
                brightness_temp_matrix_kelvins[..., j, k],
                axes_object=axes_object_matrix[j, k],
                min_latitude_deg_n=latitudes_deg_n[0],
                min_longitude_deg_e=longitudes_deg_e[0],
                latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
                longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
                cbar_orientation_string=cbar_orientation_string,
                font_size=FONT_SIZE
            )

            if cbar_orientation_string is not None:
                colour_bar_object.set_label('Brightness temp (Kelvins)')

            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=latitudes_deg_n,
                plot_longitudes_deg_e=longitudes_deg_e,
                axes_object=axes_object_matrix[j, k],
                parallel_spacing_deg=0.5, meridian_spacing_deg=1.,
                font_size=FONT_SIZE
            )

            if j == 0:
                axes_object_matrix[j, k].set_title(
                    'Band {0:d}'.format(band_numbers[k])
                )

            if k == 0:
                axes_object_matrix[j, k].set_ylabel('{0:d}-minute lag'.format(
                    lag_times_minutes[j]
                ))
            else:
                y_tick_values = axes_object_matrix[j, k].get_yticks()
                axes_object_matrix[j, k].set_yticks(y_tick_values)
                axes_object_matrix[j, k].set_yticklabels([
                    '' for _ in y_tick_values
                ])

            if j != num_lag_times - 1:
                x_tick_values = axes_object_matrix[j, k].get_xticks()
                axes_object_matrix[j, k].set_xticks(x_tick_values)
                axes_object_matrix[j, k].set_xticklabels([
                    '' for _ in x_tick_values
                ])

    return figure_object_matrix, axes_object_matrix


def _plot_saliency_one_example(
        saliency_dict, example_index, figure_object_matrix, axes_object_matrix,
        colour_map_object, min_abs_contour_value, max_abs_contour_value,
        half_num_contours, line_width):
    """Plots saliency for one example.

    :param saliency_dict: Dictionary returned by `saliency.read_file`.
    :param example_index: Will plot the [i]th example, where
        i = `example_index`.
    :param figure_object_matrix: See doc for `_plot_predictors_one_example`.
    :param axes_object_matrix: Same.
    :param colour_map_object: See documentation at top of file.
    :param min_abs_contour_value: Same.
    :param max_abs_contour_value: Same.
    :param half_num_contours: Same.
    :param line_width: Same.
    """

    latitudes_deg_n = saliency_dict[saliency.LATITUDES_KEY]
    longitudes_deg_e = saliency_dict[saliency.LONGITUDES_KEY]
    neuron_indices = saliency_dict[saliency.NEURON_INDICES_KEY]
    is_layer_output = saliency_dict[saliency.IS_LAYER_OUTPUT_KEY]

    saliency_matrix = (
        saliency_dict[saliency.SALIENCY_MATRIX_KEY][example_index, ...]
    )

    num_lag_times = saliency_matrix.shape[-2]
    num_channels = saliency_matrix.shape[-1]

    for j in range(num_lag_times):
        for k in range(num_channels):
            saliency_plotting.plot_2d_grid_latlng(
                saliency_matrix=saliency_matrix[..., j, k],
                axes_object=axes_object_matrix[j, k],
                min_latitude_deg_n=latitudes_deg_n[0],
                min_longitude_deg_e=longitudes_deg_e[0],
                latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
                longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
                colour_map_object=colour_map_object,
                min_abs_contour_value=min_abs_contour_value,
                max_abs_contour_value=max_abs_contour_value,
                half_num_contours=half_num_contours, line_width=line_width
            )

            if is_layer_output:
                figure_width_px = (
                    figure_object_matrix[j, k].get_size_inches()[0] *
                    figure_object_matrix[j, k].dpi
                )
                marker_size_px = figure_width_px * (
                    float(MARKER_SIZE_GRID_CELLS) / saliency_matrix.shape[1]
                )

                axes_object_matrix[j, k].plot(
                    longitudes_deg_e[neuron_indices[1]],
                    latitudes_deg_n[neuron_indices[0]],
                    linestyle='None', marker=MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=MARKER_COLOUR,
                    markeredgecolor=MARKER_COLOUR
                )

            if not (j == 0 and k == num_channels - 1):
                continue

            colour_bar_object = gg_plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=axes_object_matrix[j, k],
                data_matrix=saliency_matrix[..., j, k],
                colour_map_object=colour_map_object,
                min_value=min_abs_contour_value,
                max_value=max_abs_contour_value,
                orientation_string='vertical',
                extend_min=False, extend_max=True, font_size=FONT_SIZE
            )

            tick_values = colour_bar_object.get_ticks()
            tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
            colour_bar_object.set_ticks(tick_values)
            colour_bar_object.set_ticklabels(tick_strings)

            colour_bar_object.set_label('Absolute saliency')


def _concat_panels_one_example(figure_object_matrix, valid_time_unix_sec,
                               output_dir_name):
    """Concatenates panels for one example.

    :param figure_object_matrix: See doc for `_plot_predictors_one_example`.
    :param valid_time_unix_sec: Valid time.
    :param output_dir_name: Name of output directory.
    :return: concat_figure_file_name: Path to file with concatenated figure.
    """

    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, TIME_FORMAT_FOR_FILES
    )

    num_lag_times = figure_object_matrix.shape[0]
    num_channels = figure_object_matrix.shape[1]
    panel_file_names = []

    for j in range(num_lag_times):
        for k in range(num_channels):
            this_file_name = '{0:s}/{1:s}_lag{2:d}_channel{3:d}.jpg'.format(
                output_dir_name, valid_time_string, j, k
            )

            panel_file_names.append(this_file_name)

            print('Saving figure to file: "{0:s}"...'.format(this_file_name))
            figure_object_matrix[j, k].savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object_matrix[j, k])

            imagemagick_utils.resize_image(
                input_file_name=this_file_name, output_file_name=this_file_name,
                output_size_pixels=PANEL_SIZE_PX
            )

    concat_figure_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, valid_time_string
    )

    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=num_lag_times, num_panel_columns=num_channels
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

    for this_file_name in panel_file_names:
        os.remove(this_file_name)

    return concat_figure_file_name


def _run(saliency_file_name, top_predictor_dir_name, top_target_dir_name,
         lag_times_sec, band_numbers, smoothing_radius_px, colour_map_name,
         min_abs_contour_value, max_abs_contour_value, half_num_contours,
         line_width, output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param lag_times_sec: Same.
    :param band_numbers: Same.
    :param smoothing_radius_px: Same.
    :param colour_map_name: Same.
    :param min_abs_contour_value: Same.
    :param max_abs_contour_value: Same.
    :param half_num_contours: Same.
    :param line_width: Same.
    :param output_dir_name: Same.
    """

    if smoothing_radius_px <= 0:
        smoothing_radius_px = None
    if max_abs_contour_value <= 0:
        max_abs_contour_value = None
    if len(lag_times_sec) == 1 and lag_times_sec[0] < 0:
        lag_times_sec = None
    if len(band_numbers) == 1 and band_numbers[0] <= 0:
        band_numbers = None

    error_checking.assert_is_geq(half_num_contours, 5)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    colour_map_object = pyplot.get_cmap(colour_map_name)
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    print('Reading data from: "{0:s}"...'.format(saliency_file_name))
    saliency_dict = saliency.read_file(saliency_file_name)

    model_metafile_name = neural_net.find_metafile(
        model_file_name=saliency_dict[saliency.MODEL_FILE_KEY],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    saliency_matrix = saliency_dict[saliency.SALIENCY_MATRIX_KEY]

    if len(saliency_matrix.shape) == 4:
        num_lag_times = len(training_option_dict[neural_net.LAG_TIMES_KEY])

        saliency_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=saliency_matrix, num_lag_times=num_lag_times
        )
        saliency_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=saliency_matrix, num_lag_times=num_lag_times,
            add_time_dimension=True
        )

    if lag_times_sec is not None:
        these_indices = numpy.array([
            numpy.where(
                training_option_dict[neural_net.LAG_TIMES_KEY] == t
            )[0][0] for t in lag_times_sec
        ], dtype=int)

        saliency_matrix = saliency_matrix[..., these_indices, :]
        training_option_dict[neural_net.LAG_TIMES_KEY] = lag_times_sec

    if band_numbers is not None:
        these_indices = numpy.array([
            numpy.where(
                training_option_dict[neural_net.BAND_NUMBERS_KEY] == n
            )[0][0] for n in band_numbers
        ], dtype=int)

        saliency_matrix = saliency_matrix[..., these_indices]
        training_option_dict[neural_net.BAND_NUMBERS_KEY] = band_numbers

    saliency_dict[saliency.SALIENCY_MATRIX_KEY] = saliency_matrix

    if smoothing_radius_px is not None:
        saliency_dict = _smooth_maps(
            saliency_dict=saliency_dict, smoothing_radius_px=smoothing_radius_px
        )

    valid_date_string = saliency.file_name_to_date(saliency_file_name)
    radar_number = saliency.file_name_to_radar_num(saliency_file_name)

    # TODO(thunderhoser): Plotting will fail if add_coords == True.
    predictor_option_dict = {
        neural_net.PREDICTOR_DIRECTORY_KEY: top_predictor_dir_name,
        neural_net.TARGET_DIRECTORY_KEY: top_target_dir_name,
        neural_net.VALID_DATE_KEY: valid_date_string,
        neural_net.BAND_NUMBERS_KEY:
            training_option_dict[neural_net.BAND_NUMBERS_KEY],
        neural_net.LEAD_TIME_KEY:
            training_option_dict[neural_net.LEAD_TIME_KEY],
        neural_net.LAG_TIMES_KEY:
            training_option_dict[neural_net.LAG_TIMES_KEY],
        neural_net.INCLUDE_TIME_DIM_KEY: True,
        neural_net.OMIT_NORTH_RADAR_KEY:
            training_option_dict[neural_net.OMIT_NORTH_RADAR_KEY],
        neural_net.NORMALIZE_FLAG_KEY: False,
        neural_net.UNIFORMIZE_FLAG_KEY: False,
        neural_net.ADD_COORDS_KEY:
            training_option_dict[neural_net.ADD_COORDS_KEY]
    }

    # TODO(thunderhoser): Add option to return only predictors, no targets.
    predictor_dict = neural_net.create_data_partial_grids(
        option_dict=predictor_option_dict, return_coords=True,
        radar_number=radar_number
    )[radar_number]

    num_examples = saliency_dict[saliency.SALIENCY_MATRIX_KEY].shape[0]

    for i in range(num_examples):
        valid_time_unix_sec = saliency_dict[saliency.VALID_TIMES_KEY][i]

        figure_object_matrix, axes_object_matrix = _plot_predictors_one_example(
            predictor_dict=predictor_dict,
            valid_time_unix_sec=valid_time_unix_sec,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            predictor_option_dict=predictor_option_dict
        )

        if max_abs_contour_value is None:
            these_abs_values = numpy.absolute(
                saliency_dict[saliency.SALIENCY_MATRIX_KEY][i, ...]
            )
            this_max_value = numpy.percentile(these_abs_values, 99.)
        else:
            this_max_value = max_abs_contour_value + 0.

        _plot_saliency_one_example(
            saliency_dict=saliency_dict, example_index=i,
            figure_object_matrix=figure_object_matrix,
            axes_object_matrix=axes_object_matrix,
            colour_map_object=colour_map_object,
            min_abs_contour_value=min_abs_contour_value,
            max_abs_contour_value=this_max_value,
            half_num_contours=half_num_contours, line_width=line_width
        )

        _concat_panels_one_example(
            figure_object_matrix=figure_object_matrix,
            valid_time_unix_sec=valid_time_unix_sec,
            output_dir_name=output_dir_name
        )

        if i == num_examples - 1:
            continue

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        lag_times_sec=numpy.array(
            getattr(INPUT_ARG_OBJECT, LAG_TIMES_ARG_NAME), dtype=int
        ),
        band_numbers=numpy.array(
            getattr(INPUT_ARG_OBJECT, BAND_NUMBERS_ARG_NAME), dtype=int
        ),
        smoothing_radius_px=getattr(
            INPUT_ARG_OBJECT, SMOOTHING_RADIUS_ARG_NAME
        ),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        min_abs_contour_value=getattr(
            INPUT_ARG_OBJECT, MIN_CONTOUR_VALUE_ARG_NAME
        ),
        max_abs_contour_value=getattr(
            INPUT_ARG_OBJECT, MAX_CONTOUR_VALUE_ARG_NAME
        ),
        half_num_contours=getattr(INPUT_ARG_OBJECT, HALF_NUM_CONTOURS_ARG_NAME),
        line_width=getattr(INPUT_ARG_OBJECT, LINE_WIDTH_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
