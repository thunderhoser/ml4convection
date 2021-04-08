"""Plots composite saliency map (average over many examples)."""

import os
import shutil
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4convection.machine_learning import saliency
from ml4convection.machine_learning import neural_net
from ml4convection.plotting import plotting_utils
from ml4convection.plotting import saliency_plotting
from ml4convection.plotting import satellite_plotting
from ml4convection.scripts import plot_saliency_maps

SECONDS_TO_MINUTES = 1. / 60
DUMMY_GRID_SPACING_DEG = 0.0125

MARKER_SIZE_GRID_CELLS = 4.
MARKER_TYPE = '*'
MARKER_COLOUR = numpy.full(3, 0.)

FIGURE_WIDTH_INCHES = 15.
FIGURE_HEIGHT_INCHES = 15.
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 50
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

SALIENCY_FILE_ARG_NAME = plot_saliency_maps.SALIENCY_FILE_ARG_NAME
LAG_TIMES_ARG_NAME = plot_saliency_maps.LAG_TIMES_ARG_NAME
BAND_NUMBERS_ARG_NAME = plot_saliency_maps.BAND_NUMBERS_ARG_NAME
SMOOTHING_RADIUS_ARG_NAME = plot_saliency_maps.SMOOTHING_RADIUS_ARG_NAME
COLOUR_MAP_ARG_NAME = plot_saliency_maps.COLOUR_MAP_ARG_NAME
MIN_CONTOUR_VALUE_ARG_NAME = plot_saliency_maps.MIN_CONTOUR_VALUE_ARG_NAME
MAX_CONTOUR_VALUE_ARG_NAME = plot_saliency_maps.MAX_CONTOUR_VALUE_ARG_NAME
HALF_NUM_CONTOURS_ARG_NAME = plot_saliency_maps.HALF_NUM_CONTOURS_ARG_NAME
LINE_WIDTH_ARG_NAME = plot_saliency_maps.LINE_WIDTH_ARG_NAME
OUTPUT_DIR_ARG_NAME = plot_saliency_maps.OUTPUT_DIR_ARG_NAME

SALIENCY_FILE_HELP_STRING = (
    'Path to saliency file.  Will be read by `saliency.read_composite_file`.'
)
LAG_TIMES_HELP_STRING = plot_saliency_maps.LAG_TIMES_HELP_STRING
BAND_NUMBERS_HELP_STRING = plot_saliency_maps.BAND_NUMBERS_HELP_STRING
SMOOTHING_RADIUS_HELP_STRING = plot_saliency_maps.SMOOTHING_RADIUS_HELP_STRING
COLOUR_MAP_HELP_STRING = plot_saliency_maps.COLOUR_MAP_HELP_STRING
MIN_CONTOUR_VALUE_HELP_STRING = plot_saliency_maps.MIN_CONTOUR_VALUE_HELP_STRING
MAX_CONTOUR_VALUE_HELP_STRING = plot_saliency_maps.MAX_CONTOUR_VALUE_HELP_STRING
HALF_NUM_CONTOURS_HELP_STRING = plot_saliency_maps.HALF_NUM_CONTOURS_HELP_STRING
LINE_WIDTH_HELP_STRING = plot_saliency_maps.LINE_WIDTH_HELP_STRING
OUTPUT_DIR_HELP_STRING = plot_saliency_maps.OUTPUT_DIR_HELP_STRING

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + SALIENCY_FILE_ARG_NAME, type=str, required=True,
    help=SALIENCY_FILE_HELP_STRING
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


def _plot_predictors(brightness_temp_matrix_kelvins, band_numbers,
                     lag_times_seconds):
    """Plots predictors (brightness temperatures).

    M = number of rows in grid
    N = number of columns in grid
    T = number of lag times
    C = number of channels

    :param brightness_temp_matrix_kelvins: M-by-N-by-T-by-C numpy array of
        brightness temperatures (Kelvins).
    :param band_numbers: length-C numpy array of band numbers (integers).
    :param lag_times_seconds: length-T numpy array of lag times.
    :return: figure_object: T-by-C numpy array of figure handles (instances of
        `matplotlib.figure.Figure`).
    :return: axes_object_matrix: T-by-C numpy array of axes handles (instances
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    num_grid_rows = brightness_temp_matrix_kelvins.shape[0]
    num_grid_columns = brightness_temp_matrix_kelvins.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    lag_times_minutes = numpy.round(
        lag_times_seconds.astype(float) * SECONDS_TO_MINUTES
    ).astype(int)

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

            if j == num_lag_times - 1 and k == 0:
                cbar_orientation_string = 'horizontal'
            else:
                cbar_orientation_string = None

            colour_bar_object = satellite_plotting.plot_2d_grid_xy(
                brightness_temp_matrix_kelvins=
                brightness_temp_matrix_kelvins[..., j, k],
                axes_object=axes_object_matrix[j, k],
                cbar_orientation_string=cbar_orientation_string,
                font_size=FONT_SIZE
            )

            if cbar_orientation_string is not None:
                colour_bar_object.set_label('Brightness temp (Kelvins)')

            plotting_utils.plot_grid_lines(
                plot_latitudes_deg_n=y_coords,
                plot_longitudes_deg_e=x_coords,
                axes_object=axes_object_matrix[j, k],
                parallel_spacing_deg=0.2, meridian_spacing_deg=0.2,
                font_size=FONT_SIZE
            )

            x_tick_values = axes_object_matrix[j, k].get_xticks()
            axes_object_matrix[j, k].set_xticks(x_tick_values)
            axes_object_matrix[j, k].set_xticklabels([
                '' for _ in x_tick_values
            ])

            y_tick_values = axes_object_matrix[j, k].get_yticks()
            axes_object_matrix[j, k].set_yticks(y_tick_values)
            axes_object_matrix[j, k].set_yticklabels([
                '' for _ in y_tick_values
            ])

            if j == 0:
                axes_object_matrix[j, k].set_title(
                    'Band {0:d}'.format(band_numbers[k])
                )

            if k == 0:
                axes_object_matrix[j, k].set_ylabel(
                    '{0:d}-minute lag'.format(lag_times_minutes[j])
                )

    return figure_object_matrix, axes_object_matrix


def _plot_saliency(
        mean_saliency_dict, figure_object_matrix, axes_object_matrix,
        colour_map_object, min_abs_contour_value, max_abs_contour_value,
        half_num_contours, line_width):
    """Plots saliency maps on top of predictors.

    :param mean_saliency_dict: Dictionary returned by
        `saliency.read_composite_file`.
    :param figure_object_matrix: See doc for `_plot_predictors_one_example`.
    :param axes_object_matrix: Same.
    :param colour_map_object: See documentation at top of file.
    :param min_abs_contour_value: Same.
    :param max_abs_contour_value: Same.
    :param half_num_contours: Same.
    :param line_width: Same.
    """

    saliency_matrix = mean_saliency_dict[saliency.MEAN_SALIENCY_KEY]
    neuron_indices = mean_saliency_dict[saliency.NEURON_INDICES_KEY]
    is_layer_output = mean_saliency_dict[saliency.IS_LAYER_OUTPUT_KEY]

    num_grid_rows = saliency_matrix.shape[0]
    num_grid_columns = saliency_matrix.shape[1]
    x_coord_spacing = num_grid_columns ** -1
    y_coord_spacing = num_grid_rows ** -1

    x_coords, y_coords = grids.get_xy_grid_points(
        x_min_metres=x_coord_spacing / 2, y_min_metres=y_coord_spacing / 2,
        x_spacing_metres=x_coord_spacing, y_spacing_metres=y_coord_spacing,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )

    num_lag_times = saliency_matrix.shape[-2]
    num_channels = saliency_matrix.shape[-1]

    for j in range(num_lag_times):
        for k in range(num_channels):
            saliency_plotting.plot_2d_grid_xy(
                saliency_matrix=saliency_matrix[..., j, k],
                axes_object=axes_object_matrix[j, k],
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
                    x_coords[neuron_indices[1]],
                    y_coords[neuron_indices[0]],
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


def _run(saliency_file_name, lag_times_sec, band_numbers, smoothing_radius_px,
         colour_map_name, min_abs_contour_value, max_abs_contour_value,
         half_num_contours, line_width, output_dir_name):
    """Plots composite saliency map (average over many examples).

    This is effectively the main method.

    :param saliency_file_name: See documentation at top of file.
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

    colour_map_object = pyplot.get_cmap(colour_map_name)

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

    print('Reading data from: "{0:s}"...'.format(saliency_file_name))
    mean_saliency_dict = saliency.read_composite_file(saliency_file_name)

    model_metafile_name = neural_net.find_metafile(
        model_file_name=mean_saliency_dict[saliency.MODEL_FILE_KEY],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    saliency_matrix = numpy.expand_dims(
        mean_saliency_dict[saliency.MEAN_SALIENCY_KEY], axis=0
    )
    predictor_matrix = numpy.expand_dims(
        mean_saliency_dict[saliency.MEAN_DENORM_PREDICTORS_KEY], axis=0
    )

    if len(saliency_matrix.shape) == 4:
        num_lag_times = len(training_option_dict[neural_net.LAG_TIMES_KEY])

        saliency_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=saliency_matrix, num_lag_times=num_lag_times
        )
        saliency_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=saliency_matrix, num_lag_times=num_lag_times,
            add_time_dimension=True
        )

        predictor_matrix = neural_net.predictor_matrix_from_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times
        )
        predictor_matrix = neural_net.predictor_matrix_to_keras(
            predictor_matrix=predictor_matrix, num_lag_times=num_lag_times,
            add_time_dimension=True
        )

    if lag_times_sec is None:
        lag_times_sec = training_option_dict[neural_net.LAG_TIMES_KEY]
    else:
        these_indices = numpy.array([
            numpy.where(
                training_option_dict[neural_net.LAG_TIMES_KEY] == t
            )[0][0] for t in lag_times_sec
        ], dtype=int)

        saliency_matrix = saliency_matrix[..., these_indices, :]
        predictor_matrix = predictor_matrix[..., these_indices, :]

    if band_numbers is None:
        band_numbers = training_option_dict[neural_net.BAND_NUMBERS_KEY]
    else:
        these_indices = numpy.array([
            numpy.where(
                training_option_dict[neural_net.BAND_NUMBERS_KEY] == n
            )[0][0] for n in band_numbers
        ], dtype=int)

        saliency_matrix = saliency_matrix[..., these_indices]
        predictor_matrix = predictor_matrix[..., these_indices]

    mean_saliency_dict[saliency.SALIENCY_MATRIX_KEY] = saliency_matrix

    if smoothing_radius_px is not None:
        mean_saliency_dict = plot_saliency_maps._smooth_maps(
            saliency_dict=mean_saliency_dict,
            smoothing_radius_px=smoothing_radius_px
        )

    mean_saliency_dict[saliency.MEAN_SALIENCY_KEY] = saliency_matrix[0, ...]

    figure_object_matrix, axes_object_matrix = _plot_predictors(
        brightness_temp_matrix_kelvins=predictor_matrix[0, ...],
        band_numbers=band_numbers, lag_times_seconds=lag_times_sec
    )

    if max_abs_contour_value is None:
        max_abs_contour_value = numpy.percentile(
            numpy.absolute(mean_saliency_dict[saliency.MEAN_SALIENCY_KEY]),
            99.
        )

    _plot_saliency(
        mean_saliency_dict=mean_saliency_dict,
        figure_object_matrix=figure_object_matrix,
        axes_object_matrix=axes_object_matrix,
        colour_map_object=colour_map_object,
        min_abs_contour_value=min_abs_contour_value,
        max_abs_contour_value=max_abs_contour_value,
        half_num_contours=half_num_contours, line_width=line_width
    )

    orig_file_name = plot_saliency_maps._concat_panels_one_example(
        figure_object_matrix=figure_object_matrix,
        valid_time_unix_sec=0, output_dir_name=output_dir_name
    )

    new_file_name = '{0:s}/composite_saliency_map.jpg'.format(
        os.path.split(orig_file_name)[0]
    )

    print('Moving file to: "{0:s}"...'.format(new_file_name))
    shutil.move(orig_file_name, new_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        saliency_file_name=getattr(INPUT_ARG_OBJECT, SALIENCY_FILE_ARG_NAME),
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
