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

import time_conversion
import file_system_utils
import error_checking
import gg_plotting_utils
import border_io
import saliency
import neural_net
import saliency_plotting
import satellite_plotting
import plotting_utils

# TODO(thunderhoser): Allow smoothing of saliency maps.

DATE_FORMAT = neural_net.DATE_FORMAT
TIME_FORMAT_FOR_FILES = '%Y-%m-%d-%H%M'
SECONDS_TO_MINUTES = 1. / 60

FIGURE_RESOLUTION_DPI = 300

SALIENCY_FILE_ARG_NAME = 'input_saliency_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MAX_COLOUR_VALUE_ARG_NAME = 'max_colour_value'
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
COLOUR_MAP_HELP_STRING = (
    'Colour scheme for saliency.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
MAX_COLOUR_VALUE_HELP_STRING = (
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
    '--' + COLOUR_MAP_ARG_NAME, type=str, required=False, default='binary',
    help=COLOUR_MAP_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_COLOUR_VALUE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_COLOUR_VALUE_HELP_STRING
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


def _plot_predictors_one_example(
        predictor_dict, valid_time_unix_sec, border_latitudes_deg_n,
        border_longitudes_deg_e, predictor_option_dict):
    """Plots predictors (brightness temperatures) for one example.

    P = number of points in border set

    :param predictor_dict: One dictionary returned by
        `neural_net.create_data_partial_grids`.
    :param valid_time_unix_sec: Will plot predictors for this valid time.
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param predictor_option_dict: See input doc for
        `neural_net.create_data_partial_grids`.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: 2-D numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    # TODO(thunderhoser): Deal with case of only one lag time.

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

    figure_object, axes_object_matrix = gg_plotting_utils.create_paneled_figure(
        num_rows=num_lag_times, num_columns=num_channels,
        horizontal_spacing=0., vertical_spacing=0., shared_x_axis=False,
        shared_y_axis=False, keep_aspect_ratio=True
    )

    for i in range(num_lag_times):
        for j in range(num_channels):
            plotting_utils.plot_borders(
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                axes_object=axes_object_matrix[i, j]
            )

    satellite_plotting.plot_4d_grid_latlng(
        brightness_temp_matrix_kelvins=brightness_temp_matrix_kelvins,
        axes_object_matrix=axes_object_matrix,
        min_latitude_deg_n=latitudes_deg_n[0],
        min_longitude_deg_e=longitudes_deg_e[0],
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0]
    )

    for i in range(num_lag_times):
        for j in range(num_channels):
            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=latitudes_deg_n,
                plot_longitudes_deg_e=longitudes_deg_e,
                axes_object=axes_object_matrix[i, j],
                parallel_spacing_deg=1., meridian_spacing_deg=1.
            )

            if i == 0:
                axes_object_matrix[i, j].set_title(
                    'Band {0:d}'.format(band_numbers[j])
                )

            if j == 0:
                axes_object_matrix[i, j].set_ylabel('{0:d}-minute lag'.format(
                    lag_times_minutes[i]
                ))
            else:
                y_tick_values = axes_object_matrix[i, j].get_yticks()
                axes_object_matrix[i, j].set_yticks(y_tick_values)
                axes_object_matrix[i, j].set_yticklabels([
                    '' for _ in y_tick_values
                ])

            if i != num_lag_times - 1:
                x_tick_values = axes_object_matrix[i, j].get_xticks()
                axes_object_matrix[i, j].set_xticks(x_tick_values)
                axes_object_matrix[i, j].set_xticklabels([
                    '' for _ in x_tick_values
                ])

    return figure_object, axes_object_matrix


def _plot_saliency_one_example(
        saliency_dict, example_index, axes_object_matrix, colour_map_object,
        max_colour_value, half_num_contours, line_width):
    """Plots saliency for one example.

    :param saliency_dict: Dictionary returned by `saliency.read_file`.
    :param example_index: Will plot the [i]th example, where
        i = `example_index`.
    :param axes_object_matrix: See doc for `_plot_predictors_one_example`.
    :param colour_map_object: See documentation at top of file.
    :param max_colour_value: Same.
    :param half_num_contours: Same.
    :param line_width: Same.
    """

    latitudes_deg_n = saliency_dict[saliency.LATITUDES_KEY]
    longitudes_deg_e = saliency_dict[saliency.LONGITUDES_KEY]

    saliency_plotting.plot_4d_grid_latlng(
        saliency_matrix=
        saliency_dict[saliency.SALIENCY_MATRIX_KEY][example_index, ...],
        axes_object_matrix=axes_object_matrix,
        min_latitude_deg_n=latitudes_deg_n[0],
        min_longitude_deg_e=longitudes_deg_e[0],
        latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        max_absolute_contour_level=max_colour_value,
        contour_interval=max_colour_value / half_num_contours,
        line_width=line_width
    )


def _run(saliency_file_name, top_predictor_dir_name, top_target_dir_name,
         colour_map_name, max_colour_value, half_num_contours, line_width,
         output_dir_name):
    """Plots saliency maps.

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param colour_map_name: Same.
    :param max_colour_value: Same.
    :param half_num_contours: Same.
    :param line_width: Same.
    :param output_dir_name: Same.
    """

    # TODO(thunderhoser): Deal with lag times and channels being lumped on same
    # axis.

    if max_colour_value <= 0:
        max_colour_value = None

    error_checking.assert_is_geq(half_num_contours, 5)
    colour_map_object = pyplot.get_cmap(colour_map_name)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
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

    # TODO(thunderhoser): Need option to return only one radar.
    # TODO(thunderhoser): Need option to return only predictors, no targets.
    # TODO(thunderhoser): Need way to get these two vars from saliency-file name

    valid_date_string = time_conversion.unix_sec_to_string(
        saliency_dict[saliency.VALID_TIMES_KEY][0], DATE_FORMAT
    )
    radar_number = 2

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
        neural_net.INCLUDE_TIME_DIM_KEY:
            training_option_dict[neural_net.INCLUDE_TIME_DIM_KEY],
        neural_net.OMIT_NORTH_RADAR_KEY:
            training_option_dict[neural_net.OMIT_NORTH_RADAR_KEY],
        neural_net.NORMALIZE_FLAG_KEY: False,
        neural_net.UNIFORMIZE_FLAG_KEY: False,
        neural_net.ADD_COORDS_KEY:
            training_option_dict[neural_net.ADD_COORDS_KEY]
    }

    predictor_dict = neural_net.create_data_partial_grids(
        option_dict=predictor_option_dict, return_coords=True
    )[radar_number]

    num_examples = saliency_dict[saliency.SALIENCY_MATRIX_KEY].shape[0]

    for i in range(num_examples):
        figure_object, axes_object_matrix = _plot_predictors_one_example(
            predictor_dict=predictor_dict,
            valid_time_unix_sec=saliency_dict[saliency.VALID_TIMES_KEY][i],
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            predictor_option_dict=predictor_option_dict
        )

        if max_colour_value is None:
            these_abs_values = numpy.absolute(
                saliency_dict[saliency.SALIENCY_MATRIX_KEY][i, ...]
            )
            this_max_colour_value = numpy.percentile(these_abs_values, 99.)
        else:
            this_max_colour_value = max_colour_value + 0.

        _plot_saliency_one_example(
            saliency_dict=saliency_dict, example_index=i,
            axes_object_matrix=axes_object_matrix,
            colour_map_object=colour_map_object,
            max_colour_value=this_max_colour_value,
            half_num_contours=half_num_contours, line_width=line_width
        )

        valid_time_string = time_conversion.unix_sec_to_string(
            saliency_dict[saliency.VALID_TIMES_KEY][i], TIME_FORMAT_FOR_FILES
        )
        figure_file_name = '{0:s}/saliency_{1:s}.jpg'.format(
            output_dir_name, valid_time_string
        )

        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
        top_predictor_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_DIR_ARG_NAME
        ),
        top_target_dir_name=getattr(INPUT_ARG_OBJECT, TARGET_DIR_ARG_NAME),
        colour_map_name=getattr(INPUT_ARG_OBJECT, COLOUR_MAP_ARG_NAME),
        max_colour_value=getattr(INPUT_ARG_OBJECT, MAX_COLOUR_VALUE_ARG_NAME),
        half_num_contours=getattr(INPUT_ARG_OBJECT, HALF_NUM_CONTOURS_ARG_NAME),
        line_width=getattr(INPUT_ARG_OBJECT, LINE_WIDTH_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
