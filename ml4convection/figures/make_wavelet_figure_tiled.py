"""Makes schematic with tiled figures to show wavelet decomposition."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from keras import backend as K
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.io import border_io
from ml4convection.io import prediction_io
from ml4convection.utils import wavelet_utils
from ml4convection.plotting import prediction_plotting
from ml4convection.scripts import plot_predictions
from wavetf import WaveTFFactory

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LARGE_NUMBER = 1e10
TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = prediction_io.DATE_FORMAT

GRID_SPACING_DEG = 0.0125
NUM_DECOMP_LEVELS = 8

CONVERT_EXE_NAME = '/usr/bin/convert'
PANEL_LETTER_FONT_SIZE = 200
PANEL_LETTER_FONT_NAME = 'DejaVu-Sans-Bold'

GRID_LINE_WIDTH = 2.
GRID_LINE_COLOUR = numpy.full(3, 0.)

MAX_COLOUR_PERCENTILE = 99.
COEFF_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

LARGE_BORDER_WIDTH_PX = 225
SMALL_BORDER_WIDTH_PX = 10
NUM_PANEL_ROWS = 5
NUM_PANEL_COLUMNS = 2
CONCAT_FIGURE_SIZE_PX = int(1e7)

TITLE_FONT_SIZE = 36
DEFAULT_FONT_SIZE = 50

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

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
        CONVERT_EXE_NAME, image_file_name,
        PANEL_LETTER_FONT_SIZE, PANEL_LETTER_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_coeffs_one_level(coeff_matrix, title_string, output_file_name,
                           include_text_markers=False):
    """Plots wavelet coefficients at one level of decomposition.

    N = number of rows in grid = number of columns in grid

    :param coeff_matrix: N-by-N-by-4 numpy array of weights.
        coeff_matrix[..., 0] contains mean coefficients;
        coeff_matrix[..., 1] contains vertical-detail coefficients;
        coeff_matrix[..., 2] contains horizontal-detail coefficients; and
        coeff_matrix[..., 3] contains diagonal-detail coefficients.
    :param title_string: Title.
    :param output_file_name: Path to output file (figure will be saved here).
    :param include_text_markers: Boolean flag.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_rows_in_grid = coeff_matrix.shape[0]
    num_rows_in_figure = 2 * num_rows_in_grid

    matrix_to_plot = numpy.vstack((
        numpy.hstack((coeff_matrix[..., 1], coeff_matrix[..., 3])),
        numpy.hstack((coeff_matrix[..., 0], coeff_matrix[..., 2]))
    ))

    max_colour_value = numpy.percentile(numpy.absolute(matrix_to_plot), 95)
    min_colour_value = -1 * max_colour_value

    axes_object.imshow(
        matrix_to_plot, cmap=COEFF_COLOUR_MAP_OBJECT, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    tick_position = float(num_rows_in_figure) / 2 - 0.5
    axes_object.set_xticks([tick_position])
    axes_object.set_xticklabels([''])
    axes_object.set_yticks([tick_position])
    axes_object.set_yticklabels([''])
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)

    axes_object.grid(
        b=True, which='major', axis='both', linestyle='--',
        linewidth=GRID_LINE_WIDTH, color=GRID_LINE_COLOUR
    )

    for tick_object in axes_object.get_xticklines():
        tick_object.set_color(
            numpy.full(3, 1.)
        )

    for tick_object in axes_object.get_yticklines():
        tick_object.set_color(
            numpy.full(3, 1.)
        )

    if include_text_markers:
        axes_object.text(
            0.25 * float(num_rows_in_figure) - 0.5,
            0.5 * float(num_rows_in_figure),
            'LL', fontsize=4 * DEFAULT_FONT_SIZE, color='k',
            horizontalalignment='center', verticalalignment='center'
        )
        axes_object.text(
            0.75 * float(num_rows_in_figure) - 0.5,
            0.5 * float(num_rows_in_figure),
            'HL', fontsize=4 * DEFAULT_FONT_SIZE, color='k',
            horizontalalignment='center', verticalalignment='center'
        )
        axes_object.text(
            0.25 * float(num_rows_in_figure) - 0.5, 0,
            'LH', fontsize=4 * DEFAULT_FONT_SIZE, color='k',
            horizontalalignment='center', verticalalignment='center'
        )
        axes_object.text(
            0.75 * float(num_rows_in_figure) - 0.5, 0,
            'HH', fontsize=4 * DEFAULT_FONT_SIZE, color='k',
            horizontalalignment='center', verticalalignment='center'
        )

    colour_bar_object = gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=coeff_matrix,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        min_value=min_colour_value, max_value=max_colour_value,
        orientation_string='vertical', font_size=DEFAULT_FONT_SIZE,
        extend_min=True, extend_max=True, fraction_of_axis_length=0.85
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.3f}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )


def _run(top_prediction_dir_name, valid_time_string, radar_number,
         min_resolution_deg, max_resolution_deg, top_output_dir_name):
    """Makes schematic with tiled figures to show wavelet decomposition.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param valid_time_string: Same.
    :param radar_number: Same.
    :param min_resolution_deg: Same.
    :param max_resolution_deg: Same.
    :param top_output_dir_name: Same.
    """

    before_output_dir_name = '{0:s}/before_filtering'.format(
        top_output_dir_name
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=before_output_dir_name
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

    prob_colour_map_object, prob_colour_norm_object = (
        prediction_plotting.get_prob_colour_scheme_hail(
            max_probability=1., make_lowest_prob_grey=True
        )
    )

    figure_object, axes_object = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=None, title_string='',
        font_size=DEFAULT_FONT_SIZE, latlng_visible=False
    )[1:]

    axes_object.set_title(
        'Original probability field', fontsize=TITLE_FONT_SIZE
    )

    panel_file_names = [''] * (2 + NUM_DECOMP_LEVELS)
    panel_file_names[0] = '{0:s}/original_field.jpg'.format(
        before_output_dir_name
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[0]))
    figure_object.savefig(
        panel_file_names[0], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[0],
        output_file_name=panel_file_names[0],
        border_width_pixels=LARGE_BORDER_WIDTH_PX + 50
    )
    _overlay_text(
        image_file_name=panel_file_names[0],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX * 2, text_string='(a)'
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[0],
        output_file_name=panel_file_names[0]
    )

    # Plot tapered predictions.

    # TODO(thunderhoser): Modularize tapering.
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], padding_arg = (
        wavelet_utils.taper_spatial_data(
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY]
        )
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = (
        wavelet_utils.taper_spatial_data(
            prediction_dict[prediction_io.TARGET_MATRIX_KEY]
        )[0]
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = numpy.round(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY]
    ).astype(int)

    start_padding_px = padding_arg[1][0]
    end_padding_px = padding_arg[1][-1]

    latitudes_deg_n = prediction_dict[prediction_io.LATITUDES_KEY]
    latitude_spacing_deg = numpy.diff(latitudes_deg_n[:2])[0]
    start_latitudes_deg_n = (
        latitudes_deg_n[:start_padding_px] -
        start_padding_px * latitude_spacing_deg
    )
    end_latitudes_deg_n = (
        latitudes_deg_n[-end_padding_px:] +
        end_padding_px * latitude_spacing_deg
    )
    latitudes_deg_n = numpy.concatenate((
        start_latitudes_deg_n, latitudes_deg_n, end_latitudes_deg_n
    ))
    prediction_dict[prediction_io.LATITUDES_KEY] = latitudes_deg_n

    start_padding_px = padding_arg[2][0]
    end_padding_px = padding_arg[2][-1]

    longitudes_deg_e = prediction_dict[prediction_io.LONGITUDES_KEY]
    longitude_spacing_deg = numpy.diff(longitudes_deg_e[:2])[0]
    start_longitudes_deg_e = (
        longitudes_deg_e[:start_padding_px] -
        start_padding_px * longitude_spacing_deg
    )
    end_longitudes_deg_e = (
        longitudes_deg_e[-end_padding_px:] +
        end_padding_px * longitude_spacing_deg
    )
    longitudes_deg_e = numpy.concatenate((
        start_longitudes_deg_e, longitudes_deg_e, end_longitudes_deg_e
    ))
    prediction_dict[prediction_io.LONGITUDES_KEY] = longitudes_deg_e

    mask_matrix = numpy.full(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[1:],
        1, dtype=bool
    )

    figure_object, axes_object = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=None, title_string='',
        font_size=DEFAULT_FONT_SIZE, latlng_visible=False
    )[1:]

    axes_object.set_title(
        'Tapered probability field', fontsize=TITLE_FONT_SIZE
    )
    panel_file_names[1] = '{0:s}/tapered_field.jpg'.format(
        before_output_dir_name
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[1]))
    figure_object.savefig(
        panel_file_names[1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[1],
        output_file_name=panel_file_names[1],
        border_width_pixels=LARGE_BORDER_WIDTH_PX + 50
    )
    _overlay_text(
        image_file_name=panel_file_names[1],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX * 2, text_string='(b)'
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[1],
        output_file_name=panel_file_names[1]
    )

    # Plot unfiltered WT coefficients.
    print(SEPARATOR_STRING)
    coeff_tensor_by_level = wavelet_utils.do_forward_transform(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY]
    )
    print(SEPARATOR_STRING)

    panel_letter = 'b'

    for k in range(NUM_DECOMP_LEVELS):
        title_string = 'Level-{0:d} coeffs'.format(k + 1)
        title_string += r' ($\lambda$ = '
        title_string += '{0:.2g}'.format(
            GRID_SPACING_DEG * numpy.power(2, k + 1)
        )
        title_string += r'$^{\circ}$ and '
        title_string += '{0:.2g}'.format(
            GRID_SPACING_DEG * numpy.power(2, k + 2)
        )
        title_string += r'$^{\circ}$)'

        panel_file_names[k + 2] = '{0:s}/coeffs_level{1:d}.jpg'.format(
            before_output_dir_name, k + 1
        )

        _plot_coeffs_one_level(
            coeff_matrix=K.eval(coeff_tensor_by_level[k])[0, ...],
            title_string=title_string,
            output_file_name=panel_file_names[k + 2],
            include_text_markers=k == NUM_DECOMP_LEVELS - 1
        )

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[k + 2],
            output_file_name=panel_file_names[k + 2],
            border_width_pixels=LARGE_BORDER_WIDTH_PX + 50
        )

        panel_letter = chr(ord(panel_letter) + 1)
        _overlay_text(
            image_file_name=panel_file_names[k + 2],
            x_offset_from_left_px=0,
            y_offset_from_top_px=LARGE_BORDER_WIDTH_PX * 2,
            text_string='({0:s})'.format(panel_letter)
        )

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[k + 2],
            output_file_name=panel_file_names[k + 2]
        )

    # Concatenate panels.
    concat_figure_file_name = '{0:s}/wavelet_procedure_with_tiles.jpg'.format(
        before_output_dir_name
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

    after_output_dir_name = '{0:s}/after_filtering'.format(top_output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=after_output_dir_name
    )

    print(SEPARATOR_STRING)
    coeff_tensor_by_level = wavelet_utils.filter_coefficients(
        coeff_tensor_by_level=coeff_tensor_by_level,
        grid_spacing_metres=GRID_SPACING_DEG,
        min_resolution_metres=min_resolution_deg,
        max_resolution_metres=max_resolution_deg
    )
    print(SEPARATOR_STRING)

    panel_letter = chr(ord('a') - 1)
    panel_file_names = [''] * (NUM_DECOMP_LEVELS + 2)

    for k in range(NUM_DECOMP_LEVELS):
        title_string = 'Level-{0:d} coeffs'.format(k + 1)
        title_string += r' ($\lambda$ = '
        title_string += '{0:.2g}'.format(
            GRID_SPACING_DEG * numpy.power(2, k + 1)
        )
        title_string += r'$^{\circ}$ and '
        title_string += '{0:.2g}'.format(
            GRID_SPACING_DEG * numpy.power(2, k + 2)
        )
        title_string += r'$^{\circ}$)'

        panel_file_names[k] = '{0:s}/coeffs_level{1:d}.jpg'.format(
            after_output_dir_name, k + 1
        )

        _plot_coeffs_one_level(
            coeff_matrix=K.eval(coeff_tensor_by_level[k])[0, ...],
            title_string=title_string,
            output_file_name=panel_file_names[k],
            include_text_markers=k == NUM_DECOMP_LEVELS - 1
        )

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k],
            border_width_pixels=LARGE_BORDER_WIDTH_PX + 50
        )

        panel_letter = chr(ord(panel_letter) + 1)
        _overlay_text(
            image_file_name=panel_file_names[k],
            x_offset_from_left_px=0,
            y_offset_from_top_px=LARGE_BORDER_WIDTH_PX * 2,
            text_string='({0:s})'.format(panel_letter)
        )

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[k],
            output_file_name=panel_file_names[k]
        )

    inverse_dwt_object = WaveTFFactory().build('haar', dim=2, inverse=True)
    probability_tensor = inverse_dwt_object.call(coeff_tensor_by_level[0])
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        K.eval(probability_tensor)[..., 0]
    )
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = numpy.maximum(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], 0.
    )
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = numpy.minimum(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], 1.
    )

    figure_object, axes_object = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=None, title_string='',
        font_size=DEFAULT_FONT_SIZE, latlng_visible=False
    )[1:]

    axes_object.set_title(
        'Reconstructed probs with tapering', fontsize=TITLE_FONT_SIZE
    )
    panel_file_names[-2] = '{0:s}/tapered_field.jpg'.format(
        after_output_dir_name
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-2]))
    figure_object.savefig(
        panel_file_names[-2], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[-2],
        output_file_name=panel_file_names[-2],
        border_width_pixels=LARGE_BORDER_WIDTH_PX + 50
    )
    panel_letter = chr(ord(panel_letter) + 1)
    _overlay_text(
        image_file_name=panel_file_names[-2],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX * 2,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[-2],
        output_file_name=panel_file_names[-2]
    )

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY],
            numpy_pad_width=padding_arg
        )
    )
    prediction_dict[prediction_io.TARGET_MATRIX_KEY] = (
        wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=
            prediction_dict[prediction_io.TARGET_MATRIX_KEY],
            numpy_pad_width=padding_arg
        )
    )

    start_padding_px = padding_arg[1][0]
    end_padding_px = padding_arg[1][-1]
    prediction_dict[prediction_io.LATITUDES_KEY] = (
        prediction_dict[prediction_io.LATITUDES_KEY][
            start_padding_px:-end_padding_px
        ]
    )

    start_padding_px = padding_arg[2][0]
    end_padding_px = padding_arg[2][-1]
    prediction_dict[prediction_io.LONGITUDES_KEY] = (
        prediction_dict[prediction_io.LONGITUDES_KEY][
            start_padding_px:-end_padding_px
        ]
    )
    mask_matrix = numpy.full(
        prediction_dict[prediction_io.TARGET_MATRIX_KEY].shape[1:],
        1, dtype=bool
    )

    figure_object, axes_object = plot_predictions._plot_predictions_one_example(
        prediction_dict=prediction_dict, example_index=0,
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        mask_matrix=mask_matrix,
        plot_deterministic=False, probability_threshold=None,
        colour_map_object=prob_colour_map_object,
        colour_norm_object=prob_colour_norm_object,
        output_dir_name=None, title_string='',
        font_size=DEFAULT_FONT_SIZE, latlng_visible=False
    )[1:]

    axes_object.set_title(
        'Reconstructed probs without tapering', fontsize=TITLE_FONT_SIZE
    )
    panel_file_names[-1] = '{0:s}/untapered_field.jpg'.format(
        after_output_dir_name
    )

    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[-1],
        output_file_name=panel_file_names[-1],
        border_width_pixels=LARGE_BORDER_WIDTH_PX + 50
    )
    panel_letter = chr(ord(panel_letter) + 1)
    _overlay_text(
        image_file_name=panel_file_names[-1],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX * 2,
        text_string='({0:s})'.format(panel_letter)
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[-1],
        output_file_name=panel_file_names[-1]
    )

    # Concatenate panels.
    concat_figure_file_name = '{0:s}/wavelet_procedure_with_tiles.jpg'.format(
        after_output_dir_name
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
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
