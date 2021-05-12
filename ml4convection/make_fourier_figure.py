"""Makes schematic to show Fourier decomposition."""

import os
import sys
import copy
import shutil
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import tensorflow
from keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import gg_plotting_utils
import imagemagick_utils
import border_io
import prediction_io
import fourier_utils
import prediction_plotting
import plot_predictions

LARGE_NUMBER = 1e10
TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = prediction_io.DATE_FORMAT

GRID_SPACING_DEG = 0.0125
WEIGHT_COLOUR_MAP_OBJECT = pyplot.get_cmap('Reds')
BLACKMAN_COLOUR_MAP_OBJECT = pyplot.get_cmap('cividis')
BLACKMAN_COLOUR_NORM_OBJECT = pyplot.Normalize(vmin=0., vmax=1.)
BUTTERWORTH_COLOUR_MAP_OBJECT = pyplot.get_cmap('cividis')
BUTTERWORTH_COLOUR_NORM_OBJECT = pyplot.Normalize(vmin=0., vmax=1.)

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 200
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

TARGET_CONTOUR_LEVELS = numpy.array([
    0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
])
TARGET_COLOUR_MAP_OBJECT = pyplot.get_cmap('gist_gray')

MAX_COLOUR_PERCENTILE = 99.5
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

LARGE_BORDER_WIDTH_PX = 225
SMALL_BORDER_WIDTH_PX = 10
NUM_PANEL_ROWS = 4
NUM_PANEL_COLUMNS = 2
CONCAT_FIGURE_SIZE_PX = int(1e7)

FONT_SIZE = 50
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
VALID_TIME_ARG_NAME = 'valid_time_string'
RADAR_NUMBER_ARG_NAME = 'radar_number'
PLOT_TARGETS_ARG_NAME = 'plot_targets'
MIN_RESOLUTION_ARG_NAME = 'min_resolution_deg'
MAX_RESOLUTION_ARG_NAME = 'max_resolution_deg'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with prediction files.  Files therein will be '
    'found by `prediction_io.find_file` and read by `prediction_io.read_file`.'
)
VALID_TIME_HELP_STRING = (
    'Valid time (format "yyyy-mm-dd-HHMM").  Will use predictions valid at this'
    ' time.'
)
RADAR_NUMBER_HELP_STRING = (
    'Radar number (non-negative integer).  This script handles only partial '
    'grids.'
)
PLOT_TARGETS_HELP_STRING = 'Boolean flag.  If 1 (0), will (not) plot targets.'
MIN_RESOLUTION_HELP_STRING = (
    'Minimum spatial resolution to allow through band-pass filter.'
)
MAX_RESOLUTION_HELP_STRING = (
    'Max spatial resolution to allow through band-pass filter.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_DIR_ARG_NAME, type=str, required=True,
    help=PREDICTION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALID_TIME_ARG_NAME, type=str, required=True,
    help=VALID_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RADAR_NUMBER_ARG_NAME, type=int, required=True,
    help=RADAR_NUMBER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_TARGETS_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_TARGETS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_RESOLUTION_ARG_NAME, type=float, required=True,
    help=MIN_RESOLUTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RESOLUTION_ARG_NAME, type=float, required=True,
    help=MAX_RESOLUTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Overlays text on image.

    :param image_file_name: Path to image file.
    :param x_offset_from_left_px: Left-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_fourier_weights(
        weight_matrix, colour_map_object, colour_norm_object, title_string,
        output_file_name):
    """Plots Fourier weights in 2-D grid.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param weight_matrix: M-by-N numpy array of weights.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    :param title_string: Figure title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    absolute_weight_matrix = numpy.absolute(weight_matrix)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    x_resolution_matrix_deg, y_resolution_matrix_deg = (
        fourier_utils._get_spatial_resolutions(
            num_grid_rows=weight_matrix.shape[0],
            num_grid_columns=weight_matrix.shape[1],
            grid_spacing_metres=GRID_SPACING_DEG
        )
    )

    num_half_rows = int(numpy.round(
        (weight_matrix.shape[0] - 1) / 2
    ))
    num_half_columns = int(numpy.round(
        (weight_matrix.shape[1] - 1) / 2
    ))

    x_resolution_matrix_deg[:, (num_half_columns + 1):] *= -1
    y_resolution_matrix_deg[(num_half_rows + 1):, :] *= -1

    x_wavenumber_matrix_deg01 = (2 * x_resolution_matrix_deg) ** -1
    y_wavenumber_matrix_deg01 = (2 * y_resolution_matrix_deg) ** -1

    sort_indices = numpy.argsort(x_wavenumber_matrix_deg01[0, :])
    x_wavenumber_matrix_deg01 = x_wavenumber_matrix_deg01[:, sort_indices]
    absolute_weight_matrix = absolute_weight_matrix[:, sort_indices]

    sort_indices = numpy.argsort(y_wavenumber_matrix_deg01[:, 0])
    y_wavenumber_matrix_deg01 = y_wavenumber_matrix_deg01[sort_indices, :]
    absolute_weight_matrix = absolute_weight_matrix[sort_indices, :]

    axes_object.pcolormesh(
        x_wavenumber_matrix_deg01[0, :], y_wavenumber_matrix_deg01[:, 0],
        absolute_weight_matrix, cmap=colour_map_object, norm=colour_norm_object,
        shading='flat', edgecolors='None', zorder=-1e11
    )

    tick_wavenumbers_deg01 = axes_object.get_xticks()
    tick_wavelengths_deg = tick_wavenumbers_deg01 ** -1
    tick_labels = ['{0:.2g}'.format(w) for w in tick_wavelengths_deg]
    tick_labels = [l.replace('inf', r'$\infty$') for l in tick_labels]

    axes_object.set_xticks(tick_wavenumbers_deg01)
    axes_object.set_xticklabels(tick_labels, rotation=90.)
    axes_object.set_yticks(tick_wavenumbers_deg01)
    axes_object.set_yticklabels(tick_labels)

    axes_object.set_xlabel(r'Zonal wavelength ($^{\circ}$)')
    axes_object.set_ylabel(r'Meridional wavelength ($^{\circ}$)')
    axes_object.set_title(title_string)

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=absolute_weight_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=FONT_SIZE,
        extend_min=False, extend_max='Butterworth' not in title_string
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(top_prediction_dir_name, valid_time_string, radar_number, plot_targets,
         min_resolution_deg, max_resolution_deg, output_dir_name):
    """Makes schematic to show Fourier decomposition.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param valid_time_string: Same.
    :param radar_number: Same.
    :param plot_targets: Same.
    :param min_resolution_deg: Same.
    :param max_resolution_deg: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if max_resolution_deg >= LARGE_NUMBER:
        max_resolution_deg = numpy.inf

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    valid_date_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, DATE_FORMAT
    )

    prediction_file_name = prediction_io.find_file(
        top_directory_name=top_prediction_dir_name,
        valid_date_string=valid_date_string, radar_number=radar_number,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    orig_prediction_dict = copy.deepcopy(prediction_dict)

    prediction_dict = prediction_io.subset_by_time(
        prediction_dict=prediction_dict,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )[0]

    if not plot_targets:
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][:] = 0

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    # Plot original predictions.
    mask_matrix = numpy.full(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[1:],
        1, dtype=bool
    )

    prob_colour_map_object, prob_colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(
            max_probability=1., make_lowest_prob_grey=True
        )
    )

    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=output_dir_name,
        title_string='Original probability field', font_size=FONT_SIZE
    )[0]

    new_file_name = '{0:s}/original_field.jpg'.format(output_dir_name)
    shutil.move(orig_file_name, new_file_name)

    num_panels = NUM_PANEL_ROWS * NUM_PANEL_COLUMNS
    panel_file_names = [''] * num_panels
    panel_file_names[0] = new_file_name

    letter_label = 'a'

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[0],
        output_file_name=panel_file_names[0],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[0],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )

    # Plot tapered predictions.
    probability_matrix = fourier_utils.taper_spatial_data(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...]
    )
    probability_matrix = numpy.expand_dims(probability_matrix, axis=0)
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        probability_matrix
    )

    target_matrix = fourier_utils.taper_spatial_data(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][0, ...].astype(float)
    )
    target_matrix = numpy.expand_dims(target_matrix, axis=0)
    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = (
        numpy.round(target_matrix).astype(int)
    )

    latitudes_deg_n = prediction_dict[prediction_io.LATITUDES_KEY]
    latitude_extent_deg = (
        (latitudes_deg_n[1] - latitudes_deg_n[0]) * len(latitudes_deg_n)
    )
    latitudes_deg_n = numpy.concatenate((
        latitudes_deg_n - latitude_extent_deg,
        latitudes_deg_n,
        latitudes_deg_n + latitude_extent_deg
    ))
    prediction_dict[prediction_io.LATITUDES_KEY] = latitudes_deg_n

    longitudes_deg_e = prediction_dict[prediction_io.LONGITUDES_KEY]
    longitude_extent_deg = (
        (longitudes_deg_e[1] - longitudes_deg_e[0]) * len(longitudes_deg_e)
    )
    longitudes_deg_e = numpy.concatenate((
        longitudes_deg_e - longitude_extent_deg,
        longitudes_deg_e,
        longitudes_deg_e + longitude_extent_deg
    ))
    prediction_dict[prediction_io.LONGITUDES_KEY] = longitudes_deg_e

    mask_matrix = numpy.full(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[1:],
        1, dtype=bool
    )

    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=output_dir_name,
        title_string='Tapered probability field', font_size=FONT_SIZE
    )[0]

    new_file_name = '{0:s}/tapered_field.jpg'.format(output_dir_name)
    shutil.move(orig_file_name, new_file_name)
    panel_file_names[2] = new_file_name

    letter_label = chr(ord(letter_label) + 1)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[2],
        output_file_name=panel_file_names[2],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[2],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )

    # Plot Blackman-Harris window.
    probability_matrix = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...] + 0.
    )
    target_matrix = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][0, ...].astype(float)
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY][:] = 0

    bh_window_matrix = fourier_utils.apply_blackman_window(
        numpy.ones(probability_matrix.shape)
    )
    bh_window_matrix = numpy.maximum(bh_window_matrix, 0.)
    bh_window_matrix = numpy.minimum(bh_window_matrix, 1.)

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...] = (
        bh_window_matrix + 0.
    )

    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=BLACKMAN_COLOUR_MAP_OBJECT,
        colour_norm_object=BLACKMAN_COLOUR_NORM_OBJECT,
        output_dir_name=output_dir_name,
        title_string='Blackman-Harris window', font_size=FONT_SIZE,
        cbar_extend_min=False
    )[0]

    new_file_name = '{0:s}/blackman_harris_window.jpg'.format(output_dir_name)
    shutil.move(orig_file_name, new_file_name)
    panel_file_names[4] = new_file_name

    letter_label = chr(ord(letter_label) + 1)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[4],
        output_file_name=panel_file_names[4],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[4],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )

    # Plot windowed predictions.
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...] = (
        fourier_utils.apply_blackman_window(probability_matrix)
    )
    target_matrix = fourier_utils.apply_blackman_window(target_matrix)

    orig_file_name, figure_object, axes_object = (
        plot_predictions._plot_predictions_one_example(
            prediction_dict=prediction_dict, example_index=0,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            mask_matrix=mask_matrix,
            plot_deterministic=False, probability_threshold=None,
            colour_map_object=prob_colour_map_object,
            colour_norm_object=prob_colour_norm_object,
            output_dir_name=None,
            title_string='Windowed probability field', font_size=FONT_SIZE
        )
    )

    if plot_targets:
        target_contour_object = axes_object.contour(
            prediction_dict[prediction_io.LONGITUDES_KEY],
            prediction_dict[prediction_io.LATITUDES_KEY],
            target_matrix, TARGET_CONTOUR_LEVELS,
            cmap=TARGET_COLOUR_MAP_OBJECT,
            vmin=numpy.min(TARGET_CONTOUR_LEVELS),
            vmax=2 * numpy.max(TARGET_CONTOUR_LEVELS),
            linewidths=2, linestyles='solid', zorder=1e12
        )
        pyplot.clabel(
            target_contour_object, inline=True, inline_spacing=10,
            fmt='%.1g', fontsize=FONT_SIZE
        )

    panel_file_names[6] = '{0:s}/windowed_field.jpg'.format(output_dir_name)

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[6]))
    figure_object.savefig(
        panel_file_names[6], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    letter_label = chr(ord(letter_label) + 1)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[6],
        output_file_name=panel_file_names[6],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[6],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )

    # Plot original Fourier weights.
    probability_tensor = tensorflow.constant(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
        dtype=tensorflow.complex128
    )
    weight_tensor = tensorflow.signal.fft2d(probability_tensor)
    weight_matrix = K.eval(weight_tensor)[0, ...]

    target_tensor = tensorflow.constant(
        numpy.expand_dims(target_matrix, axis=0), dtype=tensorflow.complex128
    )
    target_weight_tensor = tensorflow.signal.fft2d(target_tensor)
    target_weight_matrix = K.eval(target_weight_tensor)[0, ...]

    # max_colour_value = numpy.percentile(
    #     numpy.absolute(numpy.real(weight_matrix)), MAX_COLOUR_PERCENTILE
    # )
    max_colour_value = numpy.percentile(
        numpy.absolute(weight_matrix), MAX_COLOUR_PERCENTILE
    )
    this_colour_norm_object = pyplot.Normalize(vmin=0., vmax=max_colour_value)

    this_file_name = '{0:s}/original_weights.jpg'.format(output_dir_name)
    panel_file_names[1] = this_file_name

    _plot_fourier_weights(
        weight_matrix=weight_matrix,
        colour_map_object=WEIGHT_COLOUR_MAP_OBJECT,
        colour_norm_object=this_colour_norm_object,
        title_string='Original Fourier spectrum',
        output_file_name=panel_file_names[1]
    )

    letter_label = chr(ord(letter_label) + 1)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[1],
        output_file_name=panel_file_names[1],
        border_width_pixels=LARGE_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[1],
        x_offset_from_left_px=0,
        y_offset_from_top_px=2 * LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[1],
        output_file_name=panel_file_names[1],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )

    # Plot Butterworth filter.
    butterworth_filter_matrix = fourier_utils.apply_butterworth_filter(
        coefficient_matrix=numpy.ones(weight_matrix.shape), filter_order=2.,
        grid_spacing_metres=GRID_SPACING_DEG,
        min_resolution_metres=min_resolution_deg,
        max_resolution_metres=max_resolution_deg
    )

    this_file_name = '{0:s}/butterworth_filter.jpg'.format(output_dir_name)
    panel_file_names[3] = this_file_name

    _plot_fourier_weights(
        weight_matrix=butterworth_filter_matrix,
        colour_map_object=BUTTERWORTH_COLOUR_MAP_OBJECT,
        colour_norm_object=BUTTERWORTH_COLOUR_NORM_OBJECT,
        title_string='Butterworth filter',
        output_file_name=panel_file_names[3]
    )

    letter_label = chr(ord(letter_label) + 1)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[3],
        output_file_name=panel_file_names[3],
        border_width_pixels=LARGE_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[3],
        x_offset_from_left_px=0,
        y_offset_from_top_px=2 * LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[3],
        output_file_name=panel_file_names[3],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )

    # Plot filtered Fourier weights.
    weight_matrix = fourier_utils.apply_butterworth_filter(
        coefficient_matrix=weight_matrix, filter_order=2.,
        grid_spacing_metres=GRID_SPACING_DEG,
        min_resolution_metres=min_resolution_deg,
        max_resolution_metres=max_resolution_deg
    )
    target_weight_matrix = fourier_utils.apply_butterworth_filter(
        coefficient_matrix=target_weight_matrix, filter_order=2.,
        grid_spacing_metres=GRID_SPACING_DEG,
        min_resolution_metres=min_resolution_deg,
        max_resolution_metres=max_resolution_deg
    )

    # max_colour_value = numpy.percentile(
    #     numpy.absolute(numpy.real(weight_matrix)), MAX_COLOUR_PERCENTILE
    # )
    max_colour_value = numpy.percentile(
        numpy.absolute(weight_matrix), MAX_COLOUR_PERCENTILE
    )
    this_colour_norm_object = pyplot.Normalize(vmin=0., vmax=max_colour_value)

    this_file_name = '{0:s}/filtered_weights.jpg'.format(output_dir_name)
    panel_file_names[5] = this_file_name

    _plot_fourier_weights(
        weight_matrix=weight_matrix,
        colour_map_object=WEIGHT_COLOUR_MAP_OBJECT,
        colour_norm_object=this_colour_norm_object,
        title_string='Filtered Fourier spectrum',
        output_file_name=panel_file_names[5]
    )

    letter_label = chr(ord(letter_label) + 1)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[5],
        output_file_name=panel_file_names[5],
        border_width_pixels=LARGE_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[5],
        x_offset_from_left_px=0,
        y_offset_from_top_px=2 * LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[5],
        output_file_name=panel_file_names[5],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )

    # Plot filtered spatial field.
    weight_tensor = tensorflow.constant(
        weight_matrix, dtype=tensorflow.complex128
    )
    weight_tensor = tensorflow.expand_dims(weight_tensor, 0)

    probability_tensor = tensorflow.signal.ifft2d(weight_tensor)
    probability_tensor = tensorflow.math.real(probability_tensor)
    probability_matrix = K.eval(probability_tensor)[0, ...]

    probability_matrix = fourier_utils.untaper_spatial_data(probability_matrix)
    probability_matrix = numpy.maximum(probability_matrix, 0.)
    probability_matrix = numpy.minimum(probability_matrix, 1.)

    target_weight_tensor = tensorflow.constant(
        target_weight_matrix, dtype=tensorflow.complex128
    )
    target_weight_tensor = tensorflow.expand_dims(target_weight_tensor, 0)

    target_tensor = tensorflow.signal.ifft2d(target_weight_tensor)
    target_tensor = tensorflow.math.real(target_tensor)
    target_matrix = K.eval(target_tensor)[0, ...]

    target_matrix = fourier_utils.untaper_spatial_data(target_matrix)
    target_matrix = numpy.maximum(target_matrix, 0.)
    target_matrix = numpy.minimum(target_matrix, 1.)

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = numpy.expand_dims(
        probability_matrix, axis=0
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = numpy.full(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY].shape,
        0, dtype=int
    )
    mask_matrix = numpy.full(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[1:],
        1, dtype=bool
    )

    prediction_dict[prediction_io.LATITUDES_KEY] = (
        orig_prediction_dict[prediction_io.LATITUDES_KEY]
    )
    prediction_dict[prediction_io.LONGITUDES_KEY] = (
        orig_prediction_dict[prediction_io.LONGITUDES_KEY]
    )

    orig_file_name, figure_object, axes_object = (
        plot_predictions._plot_predictions_one_example(
            prediction_dict=prediction_dict, example_index=0,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            mask_matrix=mask_matrix,
            plot_deterministic=False, probability_threshold=None,
            colour_map_object=prob_colour_map_object,
            colour_norm_object=prob_colour_norm_object,
            output_dir_name=None,
            title_string='Filtered probability field', font_size=FONT_SIZE
        )
    )

    if plot_targets:
        target_contour_object = axes_object.contour(
            prediction_dict[prediction_io.LONGITUDES_KEY],
            prediction_dict[prediction_io.LATITUDES_KEY],
            target_matrix, TARGET_CONTOUR_LEVELS,
            cmap=TARGET_COLOUR_MAP_OBJECT,
            vmin=numpy.min(TARGET_CONTOUR_LEVELS),
            vmax=2 * numpy.max(TARGET_CONTOUR_LEVELS),
            linewidths=2, linestyles='solid', zorder=1e12
        )
        pyplot.clabel(
            target_contour_object, inline=True, inline_spacing=10,
            fmt='%.1g', fontsize=FONT_SIZE
        )

    panel_file_names[-1] = '{0:s}/filtered_field.jpg'.format(output_dir_name)

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    letter_label = chr(ord(letter_label) + 1)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[-1],
        output_file_name=panel_file_names[-1],
        border_width_pixels=LARGE_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[-1],
        x_offset_from_left_px=0,
        y_offset_from_top_px=2 * LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[-1],
        output_file_name=panel_file_names[-1],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )

    # Concatenate panels.
    concat_figure_file_name = '{0:s}/fourier_procedure.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME
        ),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        radar_number=getattr(INPUT_ARG_OBJECT, RADAR_NUMBER_ARG_NAME),
        plot_targets=bool(getattr(INPUT_ARG_OBJECT, PLOT_TARGETS_ARG_NAME)),
        min_resolution_deg=getattr(INPUT_ARG_OBJECT, MIN_RESOLUTION_ARG_NAME),
        max_resolution_deg=getattr(INPUT_ARG_OBJECT, MAX_RESOLUTION_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
