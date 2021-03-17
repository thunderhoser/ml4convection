"""Plots class-activation maps."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import general_utils
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4convection.io import border_io
from ml4convection.machine_learning import gradcam
from ml4convection.machine_learning import neural_net
from ml4convection.plotting import cam_plotting
from ml4convection.scripts import plot_saliency_maps

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MARKER_SIZE_GRID_CELLS = 4.
MARKER_TYPE = '*'
MARKER_COLOUR = numpy.full(3, 0.)

FONT_SIZE = 50
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

GRADCAM_FILE_ARG_NAME = 'input_gradcam_file_name'
PREDICTOR_DIR_ARG_NAME = 'input_predictor_dir_name'
TARGET_DIR_ARG_NAME = 'input_target_dir_name'
LAG_TIMES_ARG_NAME = 'lag_times_sec'
BAND_NUMBERS_ARG_NAME = 'band_numbers'
SMOOTHING_RADIUS_ARG_NAME = 'smoothing_radius_px'
COLOUR_MAP_ARG_NAME = 'colour_map_name'
MIN_CONTOUR_VALUE_ARG_NAME = 'min_contour_value'
MAX_CONTOUR_VALUE_ARG_NAME = 'max_contour_value'
NUM_CONTOURS_ARG_NAME = 'num_contours'
LINE_WIDTH_ARG_NAME = 'line_width'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

GRADCAM_FILE_HELP_STRING = (
    'Path to file with class activations.  Will be read by `gradcam.read_file`.'
)
PREDICTOR_DIR_HELP_STRING = (
    'Name of top-level directory with predictors (brightness temperatures to '
    'plot under class-activation contours).  Files therein will be found by '
    '`example_io.find_predictor_file` and read by '
    '`example_io.read_predictor_file`.'
)
TARGET_DIR_HELP_STRING = (
    'Name of top-level directory with targets to use.  Files therein will be '
    'found by `example_io.find_target_file` and read by '
    '`example_io.read_target_file`.'
)
LAG_TIMES_HELP_STRING = (
    'Will plot CAMs overlain with only these lag times (seconds).  To plot CAMs'
    ' overlain with all lag times, leave this argument alone.'
)
BAND_NUMBERS_HELP_STRING = (
    'Will plot CAMs overlain with only these spectral bands (integers).  To '
    'plot CAMs overlain with all bands, leave this argument alone.'
)
SMOOTHING_RADIUS_HELP_STRING = (
    'e-folding radius for Gaussian smoother (num pixels).  If you do not want '
    'to smooth class-activation maps, leave this alone.'
)
COLOUR_MAP_HELP_STRING = (
    'Colour scheme for class activation.  Must be accepted by '
    '`matplotlib.pyplot.get_cmap`.'
)
MIN_CONTOUR_VALUE_HELP_STRING = 'Minimum class activation for contours.'
MAX_CONTOUR_VALUE_HELP_STRING = (
    'Max class activation for contours.  Leave this alone if you want '
    'max value to be determined automatically.'
)
NUM_CONTOURS_HELP_STRING = 'Number of class-activation contours.'
LINE_WIDTH_HELP_STRING = 'Line width for class-activation contours.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + GRADCAM_FILE_ARG_NAME, type=str, required=True,
    help=GRADCAM_FILE_HELP_STRING
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
    default=0.01, help=MIN_CONTOUR_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_CONTOUR_VALUE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_CONTOUR_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_CONTOURS_ARG_NAME, type=int, required=False, default=15,
    help=NUM_CONTOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LINE_WIDTH_ARG_NAME, type=float, required=False,
    default=cam_plotting.DEFAULT_CONTOUR_WIDTH, help=LINE_WIDTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _smooth_maps(gradcam_dict, smoothing_radius_px):
    """Smooths class-activation maps via Gaussian filter.

    :param gradcam_dict: Dictionary returned by `gradcam.read_file`.
    :param smoothing_radius_px: e-folding radius (num pixels).
    :return: gradcam_dict: Same as input but with smoothed maps.
    """

    print((
        'Smoothing class-activation maps with Gaussian filter (e-folding radius'
        ' of {0:.1f} grid cells)...'
    ).format(
        smoothing_radius_px
    ))

    class_activation_matrix = gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY]
    num_examples = class_activation_matrix.shape[0]

    for i in range(num_examples):
        class_activation_matrix[i, ...] = general_utils.apply_gaussian_filter(
            input_matrix=class_activation_matrix[i, ...],
            e_folding_radius_grid_cells=smoothing_radius_px
        )

    gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY] = class_activation_matrix
    return gradcam_dict


def _plot_cam_one_example(
        gradcam_dict, example_index, figure_object_matrix, axes_object_matrix,
        colour_map_object, min_contour_value, max_contour_value, num_contours,
        line_width):
    """Plots class-activation map for one example.

    T = number of lag times
    C = number of channels

    :param gradcam_dict: Dictionary returned by `gradcam.read_file`.
    :param example_index: Will plot the [i]th example, where
        i = `example_index`.
    :param figure_object_matrix: T-by-C numpy array of figure handles (instances
        of `matplotlib.figure.Figure`).
    :param axes_object_matrix: T-by-C numpy array of axes handles (instances of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: See documentation at top of file.
    :param min_contour_value: Same.
    :param max_contour_value: Same.
    :param num_contours: Same.
    :param line_width: Same.
    """

    latitudes_deg_n = gradcam_dict[gradcam.LATITUDES_KEY]
    longitudes_deg_e = gradcam_dict[gradcam.LONGITUDES_KEY]
    output_row = gradcam_dict[gradcam.OUTPUT_ROW_KEY]
    output_column = gradcam_dict[gradcam.OUTPUT_COLUMN_KEY]

    class_activation_matrix = (
        gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY][example_index, ...] + 0.
    )
    class_activation_matrix_log10 = numpy.log10(
        numpy.maximum(class_activation_matrix, 1e-6)
    )

    min_contour_value_log10 = numpy.log10(min_contour_value)
    if max_contour_value is None:
        max_contour_value_log10 = numpy.percentile(
            class_activation_matrix_log10, 99.
        )
    else:
        max_contour_value_log10 = numpy.log10(max_contour_value)

    num_lag_times = axes_object_matrix.shape[0]
    num_channels = axes_object_matrix.shape[1]

    for j in range(num_lag_times):
        for k in range(num_channels):
            cam_plotting.plot_2d_grid_latlng(
                class_activation_matrix=class_activation_matrix_log10,
                axes_object=axes_object_matrix[j, k],
                min_latitude_deg_n=latitudes_deg_n[0],
                min_longitude_deg_e=longitudes_deg_e[0],
                latitude_spacing_deg=numpy.diff(latitudes_deg_n[:2])[0],
                longitude_spacing_deg=numpy.diff(longitudes_deg_e[:2])[0],
                min_contour_value=min_contour_value_log10,
                max_contour_value=max_contour_value_log10,
                num_contours=num_contours, colour_map_object=colour_map_object,
                line_width=line_width
            )

            figure_width_px = (
                figure_object_matrix[j, k].get_size_inches()[0] *
                figure_object_matrix[j, k].dpi
            )
            marker_size_px = figure_width_px * (
                float(MARKER_SIZE_GRID_CELLS) /
                class_activation_matrix_log10.shape[1]
            )
            axes_object_matrix[j, k].plot(
                longitudes_deg_e[output_column], latitudes_deg_n[output_row],
                linestyle='None', marker=MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

            if not (j == 0 and k == num_channels - 1):
                continue

            colour_bar_object = gg_plotting_utils.plot_linear_colour_bar(
                axes_object_or_matrix=axes_object_matrix[j, k],
                data_matrix=class_activation_matrix_log10,
                colour_map_object=colour_map_object,
                min_value=min_contour_value_log10,
                max_value=max_contour_value_log10,
                orientation_string='vertical',
                extend_min=False, extend_max=True, font_size=FONT_SIZE
            )

            tick_values = colour_bar_object.get_ticks()
            tick_strings = ['{0:.2g}'.format(10 ** v) for v in tick_values]
            colour_bar_object.set_ticks(tick_values)
            colour_bar_object.set_ticklabels(tick_strings)

            colour_bar_object.set_label('Class activation')


def _run(gradcam_file_name, top_predictor_dir_name, top_target_dir_name,
         lag_times_sec, band_numbers, smoothing_radius_px, colour_map_name,
         min_contour_value, max_contour_value, num_contours, line_width,
         output_dir_name):
    """Plots class-activation maps.

    This is effectively the main method.

    :param gradcam_file_name: See documentation at top of file.
    :param top_predictor_dir_name: Same.
    :param top_target_dir_name: Same.
    :param lag_times_sec: Same.
    :param band_numbers: Same.
    :param smoothing_radius_px: Same.
    :param colour_map_name: Same.
    :param min_contour_value: Same.
    :param max_contour_value: Same.
    :param num_contours: Same.
    :param line_width: Same.
    :param output_dir_name: Same.
    """

    if smoothing_radius_px <= 0:
        smoothing_radius_px = None
    if max_contour_value <= 0:
        max_contour_value = None
    if len(lag_times_sec) == 1 and lag_times_sec[0] < 0:
        lag_times_sec = None
    if len(band_numbers) == 1 and band_numbers[0] <= 0:
        band_numbers = None

    min_contour_value = numpy.maximum(min_contour_value, 1e-6)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    colour_map_object = pyplot.get_cmap(colour_map_name)
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    print('Reading data from: "{0:s}"...'.format(gradcam_file_name))
    gradcam_dict = gradcam.read_file(gradcam_file_name)

    if smoothing_radius_px is not None:
        gradcam_dict = _smooth_maps(
            gradcam_dict=gradcam_dict, smoothing_radius_px=smoothing_radius_px
        )

    model_metafile_name = neural_net.find_metafile(
        model_file_name=gradcam_dict[gradcam.MODEL_FILE_KEY],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    if lag_times_sec is not None:
        training_option_dict[neural_net.LAG_TIMES_KEY] = lag_times_sec
    if band_numbers is not None:
        training_option_dict[neural_net.BAND_NUMBERS_KEY] = band_numbers

    valid_date_string = gradcam.file_name_to_date(gradcam_file_name)
    radar_number = gradcam.file_name_to_radar_num(gradcam_file_name)

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

    num_examples = gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY].shape[0]

    for i in range(num_examples):
        valid_time_unix_sec = gradcam_dict[gradcam.VALID_TIMES_KEY][i]

        figure_object_matrix, axes_object_matrix = (
            plot_saliency_maps._plot_predictors_one_example(
                predictor_dict=predictor_dict,
                valid_time_unix_sec=valid_time_unix_sec,
                border_latitudes_deg_n=border_latitudes_deg_n,
                border_longitudes_deg_e=border_longitudes_deg_e,
                predictor_option_dict=predictor_option_dict
            )
        )

        _plot_cam_one_example(
            gradcam_dict=gradcam_dict, example_index=i,
            figure_object_matrix=figure_object_matrix,
            axes_object_matrix=axes_object_matrix,
            colour_map_object=colour_map_object,
            min_contour_value=min_contour_value,
            max_contour_value=max_contour_value,
            num_contours=num_contours, line_width=line_width
        )

        plot_saliency_maps._concat_panels_one_example(
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
        gradcam_file_name=getattr(INPUT_ARG_OBJECT, GRADCAM_FILE_ARG_NAME),
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
        min_contour_value=getattr(INPUT_ARG_OBJECT, MIN_CONTOUR_VALUE_ARG_NAME),
        max_contour_value=getattr(INPUT_ARG_OBJECT, MAX_CONTOUR_VALUE_ARG_NAME),
        num_contours=getattr(INPUT_ARG_OBJECT, NUM_CONTOURS_ARG_NAME),
        line_width=getattr(INPUT_ARG_OBJECT, LINE_WIDTH_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
