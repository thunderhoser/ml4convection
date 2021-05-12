"""Makes figure with recomposition of field from Fourier-filtered fields."""

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
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.utils import fourier_utils
from ml4convection.plotting import prediction_plotting
from ml4convection.scripts import plot_predictions

LARGE_NUMBER = 1e10
TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = prediction_io.DATE_FORMAT

GRID_SPACING_DEG = 0.0125

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 200
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

LARGE_BORDER_WIDTH_PX = 225
SMALL_BORDER_WIDTH_PX = 10
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
MIN_RESOLUTIONS_ARG_NAME = 'min_resolutions_deg'
MAX_RESOLUTIONS_ARG_NAME = 'max_resolutions_deg'
FILTER_ORDER_ARG_NAME = 'filter_order'
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
MIN_RESOLUTIONS_HELP_STRING = (
    'List of minimum spatial resolutions (one for each band-pass filter).'
)
MAX_RESOLUTIONS_HELP_STRING = (
    'List of max spatial resolutions (one for each band-pass filter).'
)
FILTER_ORDER_HELP_STRING = 'Order of Butterworth filter.'
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
    '--' + MIN_RESOLUTIONS_ARG_NAME, type=float, nargs='+', required=True,
    help=MIN_RESOLUTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RESOLUTIONS_ARG_NAME, type=float, nargs='+', required=True,
    help=MAX_RESOLUTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FILTER_ORDER_ARG_NAME, type=float, required=False, default=2.,
    help=FILTER_ORDER_HELP_STRING
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


def _run(top_prediction_dir_name, valid_time_string, radar_number,
         min_resolutions_deg, max_resolutions_deg, filter_order,
         output_dir_name):
    """Makes figure with recomposition of field from Fourier-filtered fields.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param valid_time_string: Same.
    :param radar_number: Same.
    :param min_resolutions_deg: Same.
    :param max_resolutions_deg: Same.
    :param filter_order: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Process input args.
    max_resolutions_deg[max_resolutions_deg >= LARGE_NUMBER] = numpy.inf

    num_bands = len(max_resolutions_deg)
    expected_dim = numpy.array([num_bands], dtype=int)
    error_checking.assert_is_numpy_array(
        min_resolutions_deg, exact_dimensions=expected_dim
    )

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    valid_date_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, DATE_FORMAT
    )

    # Read predictions.
    prediction_file_name = prediction_io.find_file(
        top_directory_name=top_prediction_dir_name,
        valid_date_string=valid_date_string, radar_number=radar_number,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    prediction_dict[prediction_io.TARGET_MATRIX_KEY][:] = 0
    orig_prediction_dict = copy.deepcopy(prediction_dict)

    prediction_dict = prediction_io.subset_by_time(
        prediction_dict=prediction_dict,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )[0]

    mask_matrix = numpy.full(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[1:],
        1, dtype=bool
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()
    prob_colour_map_object, prob_colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(
            max_probability=1., make_lowest_prob_grey=True
        )
    )

    # Plot original predictions.
    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=output_dir_name,
        title_string='Original field', font_size=FONT_SIZE
    )[0]

    new_file_name = '{0:s}/original_field.jpg'.format(output_dir_name)
    shutil.move(orig_file_name, new_file_name)

    num_panels = num_bands + 2
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

    # Apply window to original field.
    probability_matrix = fourier_utils.taper_spatial_data(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][0, ...]
    )
    probability_matrix = fourier_utils.apply_blackman_window(probability_matrix)
    probability_matrix = numpy.expand_dims(probability_matrix, axis=0)
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        probability_matrix
    )

    # Plot filtered predictions.
    tapered_prediction_dict = copy.deepcopy(prediction_dict)
    prediction_dict_by_band = [dict()] * num_bands

    for k in range(num_bands):
        this_prob_tensor = tensorflow.constant(
            tapered_prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
            dtype=tensorflow.complex128
        )

        this_weight_tensor = tensorflow.signal.fft2d(this_prob_tensor)
        this_weight_matrix = K.eval(this_weight_tensor)[0, ...]

        this_weight_matrix = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=this_weight_matrix, filter_order=filter_order,
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=min_resolutions_deg[k],
            max_resolution_metres=max_resolutions_deg[k]
        )

        this_weight_tensor = tensorflow.constant(
            this_weight_matrix, dtype=tensorflow.complex128
        )
        this_weight_tensor = tensorflow.expand_dims(this_weight_tensor, 0)

        this_prob_tensor = tensorflow.signal.ifft2d(this_weight_tensor)
        this_prob_tensor = tensorflow.math.real(this_prob_tensor)
        this_prob_matrix = K.eval(this_prob_tensor)[0, ...]

        this_prob_matrix = fourier_utils.untaper_spatial_data(this_prob_matrix)
        this_prob_matrix = numpy.maximum(this_prob_matrix, 0.)
        this_prob_matrix = numpy.minimum(this_prob_matrix, 1.)

        prediction_dict_by_band[k] = copy.deepcopy(orig_prediction_dict)
        prediction_dict_by_band[k][prediction_io.PROBABILITY_MATRIX_KEY] = (
            numpy.expand_dims(this_prob_matrix, axis=0)
        )

        this_title_string = (
            r'$\delta_{min}$ = ' + '{0:.3g}'.format(min_resolutions_deg[k]) +
            r'$^{\circ}$; $\delta_{max}$ = ' +
            '{0:.3g}'.format(max_resolutions_deg[k]) + r'$^{\circ}$'
        )
        this_title_string = this_title_string.replace('inf', r'$\infty$')

        orig_file_name = plot_predictions._plot_predictions_one_example(
            prediction_dict=prediction_dict_by_band[k], example_index=0,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            mask_matrix=mask_matrix,
            plot_deterministic=False, probability_threshold=None,
            colour_map_object=prob_colour_map_object,
            colour_norm_object=prob_colour_norm_object,
            output_dir_name=output_dir_name,
            title_string=this_title_string, font_size=FONT_SIZE
        )[0]

        new_file_name = '{0:s}/filtered_field{1:02d}.jpg'.format(
            output_dir_name, k + 1
        )
        shutil.move(orig_file_name, new_file_name)
        panel_file_names[k + 1] = new_file_name

        letter_label = chr(ord(letter_label) + 1)

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[k + 1],
            output_file_name=panel_file_names[k + 1],
            border_width_pixels=SMALL_BORDER_WIDTH_PX
        )
        _overlay_text(
            image_file_name=panel_file_names[k + 1],
            x_offset_from_left_px=0,
            y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
            text_string='({0:s})'.format(letter_label)
        )

    # Try to reconstruct original field.
    reconstructed_prob_matrix = numpy.concatenate([
        d[prediction_io.PROBABILITY_MATRIX_KEY] for d in prediction_dict_by_band
    ], axis=0)

    reconstructed_prob_matrix = numpy.sum(reconstructed_prob_matrix, axis=0)
    reconstructed_prob_matrix = numpy.maximum(reconstructed_prob_matrix, 0.)
    reconstructed_prob_matrix = numpy.minimum(reconstructed_prob_matrix, 1.)

    prediction_dict = copy.deepcopy(orig_prediction_dict)
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        numpy.expand_dims(reconstructed_prob_matrix, axis=0)
    )

    # Plot reconstructed predictions.
    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=output_dir_name,
        title_string='Reconstructed field', font_size=FONT_SIZE
    )[0]

    new_file_name = '{0:s}/reconstructed_field.jpg'.format(output_dir_name)
    shutil.move(orig_file_name, new_file_name)
    panel_file_names[-1] = new_file_name

    letter_label = chr(ord(letter_label) + 1)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[-1],
        output_file_name=panel_file_names[-1],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[-1],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
        text_string='({0:s})'.format(letter_label)
    )

    # Concatenate panels.
    concat_figure_file_name = '{0:s}/fourier_recomposition.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    num_panel_rows = int(numpy.ceil(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
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
        min_resolutions_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_RESOLUTIONS_ARG_NAME), dtype=float
        ),
        max_resolutions_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_RESOLUTIONS_ARG_NAME), dtype=float
        ),
        filter_order=getattr(INPUT_ARG_OBJECT, FILTER_ORDER_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
