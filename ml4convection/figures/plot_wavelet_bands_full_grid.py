"""Plots signal from different wavelength bands, according to wavelet decomp.

This script uses only the target field, not the predicted-probability field.
"""

import os
import copy
import shutil
import argparse
import numpy
import tensorflow
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.utils import wavelet_utils
from ml4convection.utils import fourier_utils
from ml4convection.plotting import prediction_plotting
from ml4convection.scripts import plot_predictions
from wavetf import WaveTFFactory

LARGE_NUMBER = 1e10
TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = prediction_io.DATE_FORMAT

GRID_SPACING_DEG = 0.0125
MIN_LATITUDE_DEG_N = 21.
MAX_LATITUDE_DEG_N = 25.
MIN_LONGITUDE_DEG_E = 119.
MAX_LONGITUDE_DEG_E = 123.

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 200
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

MAX_COLOUR_PERCENTILE = 99.5
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
MIN_RESOLUTIONS_ARG_NAME = 'min_resolutions_deg'
MAX_RESOLUTIONS_ARG_NAME = 'max_resolutions_deg'
USE_FOURIER_ARG_NAME = 'use_fourier'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with prediction files.  Files therein will be '
    'found by `prediction_io.find_file` and read by `prediction_io.read_file`.'
)
VALID_TIME_HELP_STRING = (
    'Valid time (format "yyyy-mm-dd-HHMM").  Will use predictions valid at this'
    ' time.'
)
MIN_RESOLUTIONS_HELP_STRING = (
    'List of minimum spatial resolutions, one per band-pass filter.'
)
MAX_RESOLUTIONS_HELP_STRING = (
    'List of max spatial resolutions, one per band-pass filter.'
)
USE_FOURIER_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use Fourier (wavelet) decomposition.'
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
    '--' + MIN_RESOLUTIONS_ARG_NAME, type=float, nargs='+', required=True,
    help=MIN_RESOLUTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RESOLUTIONS_ARG_NAME, type=float, nargs='+', required=True,
    help=MAX_RESOLUTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_FOURIER_ARG_NAME, type=int, required=True,
    help=USE_FOURIER_HELP_STRING
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


def _run(top_prediction_dir_name, valid_time_string, min_resolutions_deg,
         max_resolutions_deg, use_fourier, output_dir_name):
    """Plots signal from different wavelength bands, selon wavelet decomp.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param valid_time_string: Same.
    :param min_resolutions_deg: Same.
    :param max_resolutions_deg: Same.
    :param use_fourier: Same.
    :param output_dir_name: Same.
    """

    # TODO(thunderhoser): Change name of script, since Fourier is now an option.

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    max_resolutions_deg[max_resolutions_deg >= LARGE_NUMBER] = numpy.inf

    num_bands = len(max_resolutions_deg)
    error_checking.assert_is_numpy_array(
        min_resolutions_deg,
        exact_dimensions=numpy.array([num_bands], dtype=int)
    )

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        valid_time_string, TIME_FORMAT
    )
    valid_date_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, DATE_FORMAT
    )

    prediction_file_name = prediction_io.find_file(
        top_directory_name=top_prediction_dir_name,
        valid_date_string=valid_date_string,
        prefer_zipped=False, allow_other_format=True,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    prediction_dict = prediction_io.subset_by_time(
        prediction_dict=prediction_dict,
        desired_times_unix_sec=numpy.array([valid_time_unix_sec], dtype=int)
    )[0]

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    # Plot original predictions.
    orig_prediction_dict = copy.deepcopy(prediction_dict)
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].astype(float) + 0.
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY][:] = 0

    good_row_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.LATITUDES_KEY] >= MIN_LATITUDE_DEG_N,
        prediction_dict[prediction_io.LATITUDES_KEY] <= MAX_LATITUDE_DEG_N
    ))[0]

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][
            :, good_row_indices, :
        ]
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][:, good_row_indices, :]
    )
    prediction_dict[prediction_io.LATITUDES_KEY] = (
        prediction_dict[prediction_io.LATITUDES_KEY][good_row_indices]
    )

    good_column_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.LONGITUDES_KEY] >= MIN_LONGITUDE_DEG_E,
        prediction_dict[prediction_io.LONGITUDES_KEY] <= MAX_LONGITUDE_DEG_E
    ))[0]

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][
            ..., good_column_indices
        ]
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = (
        prediction_dict[prediction_io.TARGET_MATRIX_KEY][
            ..., good_column_indices
        ]
    )
    prediction_dict[prediction_io.LONGITUDES_KEY] = (
        prediction_dict[prediction_io.LONGITUDES_KEY][good_column_indices]
    )

    mask_matrix = numpy.full(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[1:],
        1, dtype=bool
    )

    prob_colour_map_object, prob_colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(
            max_probability=1., make_lowest_prob_grey=True
        )
    )

    sum_of_probs = numpy.nansum(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY]
    )
    title_string = 'Original (sum = {0:.2f})'.format(sum_of_probs)

    orig_file_name = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=output_dir_name, title_string=title_string,
        font_size=FONT_SIZE
    )[0]

    new_file_name = '{0:s}/original_field.jpg'.format(output_dir_name)
    shutil.move(orig_file_name, new_file_name)

    num_panels = num_bands + 1
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

    for j in range(num_bands):
        target_matrix = (
            orig_prediction_dict[prediction_io.TARGET_MATRIX_KEY] + 0.
        )

        if use_fourier:
            num_examples = target_matrix.shape[0]

            target_matrix = numpy.stack([
                fourier_utils.taper_spatial_data(target_matrix[i, ...])
                for i in range(num_examples)
            ], axis=0)

            blackman_matrix = fourier_utils.apply_blackman_window(
                numpy.ones(target_matrix.shape[1:])
            )
            target_matrix = numpy.stack([
                target_matrix[i, ...] * blackman_matrix
                for i in range(num_examples)
            ], axis=0)

            target_tensor = tensorflow.constant(
                target_matrix, dtype=tensorflow.complex128
            )
            target_weight_tensor = tensorflow.signal.fft2d(target_tensor)
            target_weight_matrix = K.eval(target_weight_tensor)

            butterworth_matrix = fourier_utils.apply_butterworth_filter(
                coefficient_matrix=numpy.ones(target_matrix.shape[1:]),
                filter_order=2, grid_spacing_metres=GRID_SPACING_DEG,
                min_resolution_metres=min_resolutions_deg[j],
                max_resolution_metres=max_resolutions_deg[j]
            )

            target_weight_matrix = numpy.stack([
                target_weight_matrix[i, ...] * butterworth_matrix
                for i in range(num_examples)
            ], axis=0)

            target_weight_tensor = tensorflow.constant(
                target_weight_matrix, dtype=tensorflow.complex128
            )
            target_tensor = tensorflow.signal.ifft2d(target_weight_tensor)
            target_tensor = tensorflow.math.real(target_tensor)
            target_matrix = K.eval(target_tensor)

            target_matrix = numpy.stack([
                fourier_utils.untaper_spatial_data(target_matrix[i, ...])
                for i in range(num_examples)
            ], axis=0)

            target_matrix = numpy.maximum(target_matrix, 0.)
            target_matrix = numpy.minimum(target_matrix, 1.)
        else:
            target_matrix, padding_arg = wavelet_utils.taper_spatial_data(
                target_matrix
            )

            coeff_tensor_by_level = wavelet_utils.do_forward_transform(
                target_matrix
            )
            coeff_tensor_by_level = wavelet_utils.filter_coefficients(
                coeff_tensor_by_level=coeff_tensor_by_level,
                grid_spacing_metres=GRID_SPACING_DEG,
                min_resolution_metres=min_resolutions_deg[j],
                max_resolution_metres=max_resolutions_deg[j], verbose=True
            )

            inverse_dwt_object = WaveTFFactory().build('haar', dim=2, inverse=True)
            target_tensor = inverse_dwt_object.call(coeff_tensor_by_level[0])
            target_matrix = K.eval(target_tensor)[..., 0]

            target_matrix = wavelet_utils.untaper_spatial_data(
                spatial_data_matrix=target_matrix, numpy_pad_width=padding_arg
            )

        target_matrix = numpy.maximum(target_matrix, 0.)
        target_matrix = numpy.minimum(target_matrix, 1.)
        target_matrix = target_matrix[:, good_row_indices, :]
        target_matrix = target_matrix[..., good_column_indices]

        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = target_matrix
        title_string = r'$\lambda \in$ ['
        title_string += '{0:.3f}-{1:.3f}'.format(
            2 * min_resolutions_deg[j], 2 * max_resolutions_deg[j]
        )
        title_string += r']$^{\circ}$'
        title_string += ' (sum = {0:.2f})'.format(
            numpy.nansum(target_matrix)
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
            title_string=title_string, font_size=FONT_SIZE
        )[0]

        new_file_name = '{0:s}/wavelengths_{1:.3f}-{2:.3f}-deg.jpg'.format(
            output_dir_name,
            2 * min_resolutions_deg[j],
            2 * max_resolutions_deg[j]
        )
        shutil.move(orig_file_name, new_file_name)

        panel_file_names[j + 1] = new_file_name
        letter_label = chr(ord(letter_label) + 1)

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[j + 1],
            output_file_name=panel_file_names[j + 1],
            border_width_pixels=SMALL_BORDER_WIDTH_PX
        )
        _overlay_text(
            image_file_name=panel_file_names[j + 1],
            x_offset_from_left_px=0,
            y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
            text_string='({0:s})'.format(letter_label)
        )

    # Concatenate panels.
    concat_figure_file_name = '{0:s}/wavelet_bands.jpg'.format(output_dir_name)
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
        min_resolutions_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, MIN_RESOLUTIONS_ARG_NAME), dtype=float
        ),
        max_resolutions_deg=numpy.array(
            getattr(INPUT_ARG_OBJECT, MAX_RESOLUTIONS_ARG_NAME), dtype=float
        ),
        use_fourier=bool(getattr(INPUT_ARG_OBJECT, USE_FOURIER_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
