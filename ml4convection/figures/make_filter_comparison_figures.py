"""Makes figure to compare effects of different spatial filters on targets."""

import os
import copy
import shutil
import argparse
import numpy
import tensorflow
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from PIL import Image
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from tensorflow.keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.utils import fourier_utils
from ml4convection.utils import wavelet_utils
from ml4convection.plotting import prediction_plotting
from ml4convection.scripts import plot_predictions
from wavetf import WaveTFFactory

TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = prediction_io.DATE_FORMAT

# MIN_RESOLUTIONS_DEG = numpy.array([0, 0.0125, 0.025,  0.05, 0.1, 0.2, 0.4, 0.8])
# MAX_RESOLUTIONS_DEG = numpy.array([
#     0.0125, 0.025,  0.05, 0.1, 0.2, 0.4, 0.8, numpy.inf
# ])

MIN_RESOLUTIONS_DEG = numpy.array([0, 0, 0, 0, 0.05, 0.1, 0.2, 0.4])
MAX_RESOLUTIONS_DEG = numpy.array([
    0.05, 0.1, 0.2, 0.4, numpy.inf, numpy.inf, numpy.inf, numpy.inf
])
HALF_WINDOW_SIZES_PX = numpy.array([2, 2, 4, 4, 6, 6, 8, 8], dtype=int)
MAX_FILTER_FLAGS = numpy.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)

GRID_SPACING_DEG = 0.0125
MIN_LATITUDE_DEG_N = 21.
MAX_LATITUDE_DEG_N = 25.
MIN_LONGITUDE_DEG_E = 119.
MAX_LONGITUDE_DEG_E = 123.

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 300
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_RESOLUTION_DPI = 300
LARGE_BORDER_WIDTH_PX = 225
SMALL_BORDER_WIDTH_PX = 10
CONCAT_FIGURE_SIZE_PX = int(1e7)

DEFAULT_FONT_SIZE = 69
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

PREDICTION_DIR_ARG_NAME = 'input_prediction_dir_name'
VALID_TIME_ARG_NAME = 'valid_time_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_DIR_HELP_STRING = (
    'Name of top-level directory with prediction/target files.  Files therein '
    'will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
VALID_TIME_HELP_STRING = (
    'Valid time (format "yyyy-mm-dd-HHMM").  Will use targets valid at this '
    'time.'
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


def _add_colour_bar(
        figure_file_name, colour_map_object, colour_norm_object,
        temporary_dir_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param colour_norm_object: Normalizer for colour scheme (instance of
        `matplotlib.pyplot.Normalize` or similar).
    :param temporary_dir_name: Name of temporary output directory.
    """

    this_image_matrix = Image.open(figure_file_name)
    figure_width_px, figure_height_px = this_image_matrix.size
    figure_width_inches = float(figure_width_px) / FIGURE_RESOLUTION_DPI
    figure_height_inches = float(figure_height_px) / FIGURE_RESOLUTION_DPI

    extra_figure_object, extra_axes_object = pyplot.subplots(
        1, 1, figsize=(figure_width_inches, figure_height_inches)
    )
    extra_axes_object.axis('off')
    dummy_values = numpy.array([0.05, 0.95])

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=1.25, font_size=25
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    extra_file_name = '{0:s}/extra_colour_bar.jpg'.format(temporary_dir_name)
    print('Saving colour bar to: "{0:s}"...'.format(extra_file_name))

    extra_figure_object.savefig(
        extra_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(extra_figure_object)

    print('Concatenating colour bar to: "{0:s}"...'.format(figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[figure_file_name, extra_file_name],
        output_file_name=figure_file_name,
        num_panel_rows=1, num_panel_columns=2,
        extra_args_string='-gravity Center'
    )

    os.remove(extra_file_name)
    imagemagick_utils.trim_whitespace(
        input_file_name=figure_file_name, output_file_name=figure_file_name
    )


def _run(top_prediction_dir_name, valid_time_string, output_dir_name):
    """Makes figure to compare effects of different spatial filters on targets.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param valid_time_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
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

    # Copy original predictions.
    good_row_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.LATITUDES_KEY] >= MIN_LATITUDE_DEG_N,
        prediction_dict[prediction_io.LATITUDES_KEY] <= MAX_LATITUDE_DEG_N
    ))[0]
    prediction_dict[prediction_io.LATITUDES_KEY] = (
        prediction_dict[prediction_io.LATITUDES_KEY][good_row_indices]
    )

    good_column_indices = numpy.where(numpy.logical_and(
        prediction_dict[prediction_io.LONGITUDES_KEY] >= MIN_LONGITUDE_DEG_E,
        prediction_dict[prediction_io.LONGITUDES_KEY] <= MAX_LONGITUDE_DEG_E
    ))[0]
    prediction_dict[prediction_io.LONGITUDES_KEY] = (
        prediction_dict[prediction_io.LONGITUDES_KEY][good_column_indices]
    )

    orig_prediction_dict = copy.deepcopy(prediction_dict)
    orig_target_matrix = (
        orig_prediction_dict[prediction_io.TARGET_MATRIX_KEY][
            :, good_row_indices, :
        ]
    )
    orig_target_matrix = orig_target_matrix[..., good_column_indices]

    mask_matrix = numpy.full(
        (len(good_row_indices), len(good_column_indices)), 1, dtype=bool
    )
    prob_colour_map_object, prob_colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme(
            max_probability=1., make_lowest_prob_grey=True
        )
    )

    num_bands = len(MIN_RESOLUTIONS_DEG)
    panel_file_names = [''] * num_bands
    letter_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    for j in range(num_bands):
        target_matrix = (
            orig_prediction_dict[prediction_io.TARGET_MATRIX_KEY] + 0.
        )
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
            min_resolution_metres=MIN_RESOLUTIONS_DEG[j],
            max_resolution_metres=MAX_RESOLUTIONS_DEG[j]
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
        target_matrix = target_matrix[:, good_row_indices, :]
        target_matrix = target_matrix[..., good_column_indices]
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
            numpy.expand_dims(target_matrix, axis=-1)
        )
        prediction_dict[prediction_io.TARGET_MATRIX_KEY] = orig_target_matrix

        title_string = '  Fourier {0:.2g}-{1:.2g}'.format(
            2 * MIN_RESOLUTIONS_DEG[j], 2 * MAX_RESOLUTIONS_DEG[j]
        )
        title_string = title_string.replace('inf', r'$\infty$')
        title_string += r'$^{\circ}$'
        title_string += '\n  (sum = {0:.1f})'.format(numpy.sum(target_matrix))

        orig_file_name = plot_predictions._plot_predictions_one_example(
            prediction_dict=prediction_dict, example_index=0,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            mask_matrix=mask_matrix,
            plot_deterministic=False, probability_threshold=None,
            colour_map_object=prob_colour_map_object,
            colour_norm_object=prob_colour_norm_object,
            output_dir_name=output_dir_name,
            title_string=title_string, font_size=DEFAULT_FONT_SIZE,
            plot_colour_bar=False, latlng_visible=False
        )[0]

        panel_file_names[j] = (
            '{0:s}/fourier_wavelengths={1:.3f}-{2:.3f}-deg.jpg'
        ).format(
            output_dir_name,
            2 * MIN_RESOLUTIONS_DEG[j],
            2 * MAX_RESOLUTIONS_DEG[j]
        )
        shutil.move(orig_file_name, panel_file_names[j])

        # imagemagick_utils.trim_whitespace(
        #     input_file_name=panel_file_names[j],
        #     output_file_name=panel_file_names[j],
        #     border_width_pixels=LARGE_BORDER_WIDTH_PX
        # )
        # _overlay_text(
        #     image_file_name=panel_file_names[j],
        #     x_offset_from_left_px=0,
        #     y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
        #     text_string='({0:s})'.format(letter_labels[j])
        # )

        _overlay_text(
            image_file_name=panel_file_names[j],
            x_offset_from_left_px=0,
            y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
            text_string='({0:s})'.format(letter_labels[j])
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[j],
            output_file_name=panel_file_names[j],
            border_width_pixels=SMALL_BORDER_WIDTH_PX
        )

    concat_fourier_file_name = '{0:s}/fourier_filters.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        concat_fourier_file_name
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_fourier_file_name,
        num_panel_rows=2, num_panel_columns=4
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_fourier_file_name,
        output_file_name=concat_fourier_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_fourier_file_name,
        output_file_name=concat_fourier_file_name,
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )

    panel_file_names = [''] * num_bands
    letter_labels = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

    for j in range(num_bands):
        target_matrix = (
            orig_prediction_dict[prediction_io.TARGET_MATRIX_KEY] + 0.
        )
        target_matrix, padding_arg = wavelet_utils.taper_spatial_data(
            target_matrix
        )

        coeff_tensor_by_level = wavelet_utils.do_forward_transform(
            target_matrix
        )
        coeff_tensor_by_level = wavelet_utils.filter_coefficients(
            coeff_tensor_by_level=coeff_tensor_by_level,
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=MIN_RESOLUTIONS_DEG[j],
            max_resolution_metres=MAX_RESOLUTIONS_DEG[j], verbose=True
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
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
            numpy.expand_dims(target_matrix, axis=-1)
        )
        prediction_dict[prediction_io.TARGET_MATRIX_KEY] = orig_target_matrix

        title_string = '   Wavelet {0:.2g}-{1:.2g}'.format(
            2 * MIN_RESOLUTIONS_DEG[j], 2 * MAX_RESOLUTIONS_DEG[j]
        )
        title_string = title_string.replace('inf', r'$\infty$')
        title_string += r'$^{\circ}$'
        title_string += '\n   (sum = {0:.1f})'.format(numpy.sum(target_matrix))

        orig_file_name = plot_predictions._plot_predictions_one_example(
            prediction_dict=prediction_dict, example_index=0,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            mask_matrix=mask_matrix,
            plot_deterministic=False, probability_threshold=None,
            colour_map_object=prob_colour_map_object,
            colour_norm_object=prob_colour_norm_object,
            output_dir_name=output_dir_name,
            title_string=title_string, font_size=DEFAULT_FONT_SIZE,
            plot_colour_bar=False, latlng_visible=j == 4
        )[0]

        panel_file_names[j] = (
            '{0:s}/wavelet_wavelengths={1:.3f}-{2:.3f}-deg.jpg'
        ).format(
            output_dir_name,
            2 * MIN_RESOLUTIONS_DEG[j],
            2 * MAX_RESOLUTIONS_DEG[j]
        )
        shutil.move(orig_file_name, panel_file_names[j])

        # imagemagick_utils.trim_whitespace(
        #     input_file_name=panel_file_names[j],
        #     output_file_name=panel_file_names[j],
        #     border_width_pixels=LARGE_BORDER_WIDTH_PX
        # )
        # _overlay_text(
        #     image_file_name=panel_file_names[j],
        #     x_offset_from_left_px=0,
        #     y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
        #     text_string='({0:s})'.format(letter_labels[j])
        # )

        _overlay_text(
            image_file_name=panel_file_names[j],
            x_offset_from_left_px=0,
            y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
            text_string='({0:s})'.format(letter_labels[j])
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[j],
            output_file_name=panel_file_names[j],
            border_width_pixels=SMALL_BORDER_WIDTH_PX
        )

    concat_wavelet_file_name = '{0:s}/wavelet_filters.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        concat_wavelet_file_name
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_wavelet_file_name,
        num_panel_rows=2, num_panel_columns=4
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_wavelet_file_name,
        output_file_name=concat_wavelet_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_wavelet_file_name,
        output_file_name=concat_wavelet_file_name,
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )

    concat_figure_file_name = '{0:s}/transform_filters.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=[concat_fourier_file_name, concat_wavelet_file_name],
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=1
    )
    _add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        temporary_dir_name=output_dir_name
    )

    num_neighs = len(HALF_WINDOW_SIZES_PX)
    panel_file_names = [''] * num_neighs
    letter_labels = ['a', 'e', 'b', 'f', 'c', 'g', 'd', 'h']

    for j in range(num_neighs):
        target_matrix = (
            orig_prediction_dict[prediction_io.TARGET_MATRIX_KEY] + 0.
        )
        num_examples = target_matrix.shape[0]

        dimensions = (
            2 * HALF_WINDOW_SIZES_PX[j] + 1,
            2 * HALF_WINDOW_SIZES_PX[j] + 1
        )

        if MAX_FILTER_FLAGS[j]:
            structure_matrix = numpy.full(dimensions, 1, dtype=bool)

            target_matrix = numpy.stack([
                maximum_filter(
                    target_matrix[i, ...].astype(float),
                    footprint=structure_matrix, mode='constant', cval=0.
                )
                for i in range(num_examples)
            ], axis=0)
        else:
            weight_matrix = numpy.full(dimensions, 1, dtype=float)
            weight_matrix = weight_matrix / weight_matrix.size

            target_matrix = numpy.stack([
                convolve2d(
                    target_matrix[i, ...].astype(float),
                    weight_matrix, mode='same'
                )
                for i in range(num_examples)
            ], axis=0)

        target_matrix = numpy.maximum(target_matrix, 0.)
        target_matrix = numpy.minimum(target_matrix, 1.)
        target_matrix = target_matrix[:, good_row_indices, :]
        target_matrix = target_matrix[..., good_column_indices]
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
            numpy.expand_dims(target_matrix, axis=-1)
        )
        prediction_dict[prediction_io.TARGET_MATRIX_KEY] = orig_target_matrix

        title_string = '{0:d}x{0:d} {1:s} filter\n(sum = {2:.1f})'.format(
            2 * HALF_WINDOW_SIZES_PX[j] + 1,
            'MAX' if MAX_FILTER_FLAGS[j] else 'MEAN',
            numpy.sum(target_matrix)
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
            title_string=title_string, font_size=DEFAULT_FONT_SIZE,
            plot_colour_bar=False, latlng_visible=j == 6
        )[0]

        panel_file_names[j] = '{0:s}/{1:s}_filter_neigh{2:d}.jpg'.format(
            output_dir_name,
            'max' if MAX_FILTER_FLAGS[j] else 'mean',
            HALF_WINDOW_SIZES_PX[j]
        )
        shutil.move(orig_file_name, panel_file_names[j])

        # imagemagick_utils.trim_whitespace(
        #     input_file_name=panel_file_names[j],
        #     output_file_name=panel_file_names[j],
        #     border_width_pixels=LARGE_BORDER_WIDTH_PX
        # )
        # _overlay_text(
        #     image_file_name=panel_file_names[j],
        #     x_offset_from_left_px=0,
        #     y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
        #     text_string='({0:s})'.format(letter_labels[j])
        # )

        _overlay_text(
            image_file_name=panel_file_names[j],
            x_offset_from_left_px=0,
            y_offset_from_top_px=LARGE_BORDER_WIDTH_PX,
            text_string='({0:s})'.format(letter_labels[j])
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[j],
            output_file_name=panel_file_names[j],
            border_width_pixels=SMALL_BORDER_WIDTH_PX
        )

    concat_figure_file_name = '{0:s}/neigh_filters.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=4, num_panel_columns=2
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
    _add_colour_bar(
        figure_file_name=concat_figure_file_name,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        temporary_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_prediction_dir_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_DIR_ARG_NAME
        ),
        valid_time_string=getattr(INPUT_ARG_OBJECT, VALID_TIME_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
