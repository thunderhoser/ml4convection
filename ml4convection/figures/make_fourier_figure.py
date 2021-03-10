"""Makes schematic to show Fourier decomposition."""

import os
import copy
import shutil
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import tensorflow
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.utils import fourier_utils
from ml4convection.scripts import plot_predictions

LARGE_NUMBER = 1e10
TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = prediction_io.DATE_FORMAT

GRID_SPACING_DEG = 0.0125
WEIGHT_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 200
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

LARGE_BORDER_WIDTH_PX = 225
SMALL_BORDER_WIDTH_PX = 10
NUM_PANEL_ROWS = 3
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
        weight_matrix, max_colour_value, title_string, output_file_name):
    """Plots Fourier weights in 2-D grid.

    M = number of rows in spatial grid
    N = number of columns in spatial grid

    :param weight_matrix: M-by-N numpy array of weights.
    :param max_colour_value: Max absolute value in colour scheme.
    :param title_string: Figure title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    real_weight_matrix = numpy.real(weight_matrix)

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

    x_wavenumber_matrix_deg01 = (2 * x_resolution_matrix_deg) ** -1
    y_wavenumber_matrix_deg01 = (2 * y_resolution_matrix_deg) ** -1

    axes_object.pcolormesh(
        x_wavenumber_matrix_deg01[0, :], y_wavenumber_matrix_deg01[:, 0],
        real_weight_matrix, cmap=WEIGHT_COLOUR_MAP_OBJECT,
        vmin=-1 * max_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', zorder=-1e11
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

    gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=real_weight_matrix,
        colour_map_object=WEIGHT_COLOUR_MAP_OBJECT,
        min_value=-1 * max_colour_value, max_value=max_colour_value,
        orientation_string='vertical', font_size=FONT_SIZE
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(top_prediction_dir_name, valid_time_string, radar_number,
         min_resolution_deg, max_resolution_deg, output_dir_name):
    """Makes schematic to show Fourier decomposition.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param valid_time_string: Same.
    :param radar_number: Same.
    :param min_resolution_deg: Same.
    :param max_resolution_deg: Same.
    :param output_dir_name: Same.
    """

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

    prediction_dict[prediction_io.TARGET_MATRIX_KEY][:] = 0
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    # Plot original predictions.
    mask_matrix = numpy.full(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[1:],
        1, dtype=bool
    )

    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix, plot_deterministic=False,
        probability_threshold=None, max_prob_in_colour_bar=1.,
        title_string='Original probability field',
        output_dir_name=output_dir_name, font_size=FONT_SIZE
    )

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
        mask_matrix=mask_matrix, plot_deterministic=False,
        probability_threshold=None, max_prob_in_colour_bar=1.,
        title_string='Tapered probability field',
        output_dir_name=output_dir_name, font_size=FONT_SIZE
    )

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

    # Plot windowed predictions.
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...] = (
        fourier_utils.apply_blackman_window(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...]
        )
    )

    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix, plot_deterministic=False,
        probability_threshold=None, max_prob_in_colour_bar=1.,
        title_string='Windowed probability field',
        output_dir_name=output_dir_name, font_size=FONT_SIZE
    )

    new_file_name = '{0:s}/windowed_field.jpg'.format(output_dir_name)
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

    # Plot original Fourier weights.
    probability_tensor = tensorflow.constant(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
        dtype=tensorflow.complex128
    )

    weight_tensor = tensorflow.signal.fft2d(probability_tensor)
    weight_matrix = K.eval(weight_tensor)[0, ...]

    max_colour_value = numpy.percentile(
        numpy.absolute(numpy.real(weight_matrix)), 99.
    )
    this_file_name = '{0:s}/original_weights.jpg'.format(output_dir_name)
    panel_file_names[1] = this_file_name

    _plot_fourier_weights(
        weight_matrix=weight_matrix, max_colour_value=max_colour_value,
        title_string='Original Fourier weights',
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

    # Plot filtered Fourier weights.
    weight_matrix = fourier_utils.apply_butterworth_filter(
        coefficient_matrix=weight_matrix, filter_order=2.,
        grid_spacing_metres=GRID_SPACING_DEG,
        min_resolution_metres=min_resolution_deg,
        max_resolution_metres=max_resolution_deg
    )

    max_colour_value = numpy.percentile(
        numpy.absolute(numpy.real(weight_matrix)), 99.
    )
    this_file_name = '{0:s}/filtered_weights.jpg'.format(output_dir_name)
    panel_file_names[3] = this_file_name

    _plot_fourier_weights(
        weight_matrix=weight_matrix, max_colour_value=max_colour_value,
        title_string='Filtered Fourier weights',
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

    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix, plot_deterministic=False,
        probability_threshold=None, max_prob_in_colour_bar=1.,
        title_string='Filtered probability field',
        output_dir_name=output_dir_name, font_size=FONT_SIZE
    )

    new_file_name = '{0:s}/filtered_field.jpg'.format(output_dir_name)
    shutil.move(orig_file_name, new_file_name)
    panel_file_names[-1] = new_file_name

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
        min_resolution_deg=getattr(INPUT_ARG_OBJECT, MIN_RESOLUTION_ARG_NAME),
        max_resolution_deg=getattr(INPUT_ARG_OBJECT, MAX_RESOLUTION_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )