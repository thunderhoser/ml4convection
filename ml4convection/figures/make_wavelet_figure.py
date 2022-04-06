"""Makes schematic to show wavelet decomposition."""

import os
import shutil
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

LARGE_NUMBER = 1e10
TIME_FORMAT = '%Y-%m-%d-%H%M'
DATE_FORMAT = prediction_io.DATE_FORMAT

GRID_SPACING_DEG = 0.0125

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 200
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

MAX_COLOUR_PERCENTILE = 99.
COEFF_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

LARGE_BORDER_WIDTH_PX = 225
SMALL_BORDER_WIDTH_PX = 10
NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 4
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


def _plot_wavelet_coeffs(
        coeff_matrix, plotting_mean_coeffs, colour_map_object,
        colour_norm_object, title_string, output_file_name):
    """Plots wavelet coefficients in 2-D grid.

    N = number of rows in grid = number of columns in grid

    :param coeff_matrix: N-by-N numpy array of weights.
    :param plotting_mean_coeffs: Boolean flag.  If True (False), plotting mean
        (detail) coefficients.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    :param title_string: Figure title.
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_grid_rows = coeff_matrix.shape[0]
    wavelengths_deg = numpy.full(num_grid_rows, numpy.nan)
    num_levels = int(numpy.round(numpy.log2(num_grid_rows)))

    i = 0

    for k in range(num_levels):
        this_num_rows = int(numpy.round(
            num_grid_rows / (2 ** (k + 1))
        ))

        if plotting_mean_coeffs:
            wavelengths_deg[i:(i + this_num_rows)] = (
                GRID_SPACING_DEG * numpy.power(2, k + 2)
            )
        else:
            wavelengths_deg[i:(i + this_num_rows)] = (
                GRID_SPACING_DEG * numpy.power(2, k + 1)
            )

        i += this_num_rows

    row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=int
    )
    axes_object.pcolormesh(
        row_indices, row_indices, coeff_matrix, cmap=colour_map_object,
        norm=colour_norm_object, shading='flat', edgecolors='None', zorder=-1e11
    )

    tick_values = row_indices[::50]
    tick_labels = [
        '{0:.2g}'.format(wavelengths_deg[i]) for i in tick_values
    ]
    tick_labels = [l.replace('inf', r'$\infty$') for l in tick_labels]

    axes_object.set_xticks(tick_values)
    axes_object.set_xticklabels(tick_labels, rotation=90.)
    axes_object.set_yticks(tick_values)
    axes_object.set_yticklabels(tick_labels)

    axes_object.set_xlabel(r'Wavelength ($^{\circ}$)')
    axes_object.set_ylabel(r'Wavelength ($^{\circ}$)')
    axes_object.set_title(title_string)

    gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=coeff_matrix,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', font_size=FONT_SIZE,
        extend_min=True, extend_max=True
    )

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
         min_resolution_deg, max_resolution_deg, output_dir_name):
    """Makes schematic to show wavelet decomposition.

    This is effectively the main method.

    :param top_prediction_dir_name: See documentation at top of file.
    :param valid_time_string: Same.
    :param radar_number: Same.
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

    panel_file_names = [''] * 11
    panel_file_names[8] = new_file_name

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[8],
        output_file_name=panel_file_names[8],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[8],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(i)'
    )

    # Plot tapered predictions.

    # TODO(thunderhoser): Modularize tapering.
    (
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][..., 0],
        padding_arg
    ) = wavelet_utils.taper_spatial_data(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][..., 0]
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
    panel_file_names[9] = new_file_name

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[9],
        output_file_name=panel_file_names[9],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[9],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(j)'
    )

    # Plot unfiltered WT coefficients.
    coeff_tensor_by_level = wavelet_utils.do_forward_transform(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][..., 0]
    )

    (
        mean_coeff_matrix,
        horizontal_coeff_matrix,
        vertical_coeff_matrix,
        diagonal_coeff_matrix
    ) = wavelet_utils.coeff_tensors_to_numpy(coeff_tensor_by_level)

    all_coeffs = numpy.concatenate((
        numpy.ravel(horizontal_coeff_matrix),
        numpy.ravel(vertical_coeff_matrix),
        numpy.ravel(diagonal_coeff_matrix)
    ))
    this_max_value = numpy.nanpercentile(
        numpy.absolute(all_coeffs), MAX_COLOUR_PERCENTILE
    )
    detail_colour_norm_object = pyplot.Normalize(
        vmin=-this_max_value, vmax=this_max_value
    )

    this_max_value = numpy.nanpercentile(
        numpy.absolute(mean_coeff_matrix), MAX_COLOUR_PERCENTILE
    )
    mean_colour_norm_object = pyplot.Normalize(
        vmin=-this_max_value, vmax=this_max_value
    )

    panel_file_names[0] = '{0:s}/mean_coeffs.jpg'.format(output_dir_name)
    _plot_wavelet_coeffs(
        coeff_matrix=mean_coeff_matrix[0, ...],
        plotting_mean_coeffs=True,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        colour_norm_object=mean_colour_norm_object,
        title_string='Mean coeffs',
        output_file_name=panel_file_names[0]
    )
    _overlay_text(
        image_file_name=panel_file_names[0],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(a)'
    )

    panel_file_names[1] = '{0:s}/horizontal_coeffs.jpg'.format(output_dir_name)
    _plot_wavelet_coeffs(
        coeff_matrix=horizontal_coeff_matrix[0, ...],
        plotting_mean_coeffs=False,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        colour_norm_object=detail_colour_norm_object,
        title_string='Horizontal coeffs',
        output_file_name=panel_file_names[1]
    )
    _overlay_text(
        image_file_name=panel_file_names[1],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(b)'
    )

    panel_file_names[2] = '{0:s}/vertical_coeffs.jpg'.format(output_dir_name)
    _plot_wavelet_coeffs(
        coeff_matrix=vertical_coeff_matrix[0, ...],
        plotting_mean_coeffs=False,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        colour_norm_object=detail_colour_norm_object,
        title_string='Vertical coeffs',
        output_file_name=panel_file_names[2]
    )
    _overlay_text(
        image_file_name=panel_file_names[2],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(c)'
    )

    panel_file_names[3] = '{0:s}/diagonal_coeffs.jpg'.format(output_dir_name)
    _plot_wavelet_coeffs(
        coeff_matrix=diagonal_coeff_matrix[0, ...],
        plotting_mean_coeffs=False,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        colour_norm_object=detail_colour_norm_object,
        title_string='Diagonal coeffs',
        output_file_name=panel_file_names[3]
    )
    _overlay_text(
        image_file_name=panel_file_names[3],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(d)'
    )

    wavelet_utils.filter_coefficients(
        coeff_tensor_by_level=coeff_tensor_by_level,
        grid_spacing_metres=GRID_SPACING_DEG,
        min_resolution_metres=min_resolution_deg,
        max_resolution_metres=max_resolution_deg
    )

    (
        mean_coeff_matrix,
        horizontal_coeff_matrix,
        vertical_coeff_matrix,
        diagonal_coeff_matrix
    ) = wavelet_utils.coeff_tensors_to_numpy(coeff_tensor_by_level)

    # Plot filtered WT coefficients.
    panel_file_names[4] = '{0:s}/filtered_mean_coeffs.jpg'.format(
        output_dir_name
    )
    _plot_wavelet_coeffs(
        coeff_matrix=mean_coeff_matrix[0, ...],
        plotting_mean_coeffs=True,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        colour_norm_object=mean_colour_norm_object,
        title_string='Filtered mean coeffs',
        output_file_name=panel_file_names[4]
    )
    _overlay_text(
        image_file_name=panel_file_names[4],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(e)'
    )

    panel_file_names[5] = '{0:s}/filtered_horizontal_coeffs.jpg'.format(
        output_dir_name
    )
    _plot_wavelet_coeffs(
        coeff_matrix=horizontal_coeff_matrix[0, ...],
        plotting_mean_coeffs=False,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        colour_norm_object=detail_colour_norm_object,
        title_string='Filtered horizontal coeffs',
        output_file_name=panel_file_names[5]
    )
    _overlay_text(
        image_file_name=panel_file_names[5],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(f)'
    )

    panel_file_names[6] = '{0:s}/filtered_vertical_coeffs.jpg'.format(
        output_dir_name
    )
    _plot_wavelet_coeffs(
        coeff_matrix=vertical_coeff_matrix[0, ...],
        plotting_mean_coeffs=False,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        colour_norm_object=detail_colour_norm_object,
        title_string='Filtered vertical coeffs',
        output_file_name=panel_file_names[6]
    )
    _overlay_text(
        image_file_name=panel_file_names[6],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(g)'
    )

    panel_file_names[7] = '{0:s}/filtered_diagonal_coeffs.jpg'.format(
        output_dir_name
    )
    _plot_wavelet_coeffs(
        coeff_matrix=diagonal_coeff_matrix[0, ...],
        plotting_mean_coeffs=False,
        colour_map_object=COEFF_COLOUR_MAP_OBJECT,
        colour_norm_object=detail_colour_norm_object,
        title_string='Filtered diagonal coeffs',
        output_file_name=panel_file_names[7]
    )
    _overlay_text(
        image_file_name=panel_file_names[7],
        x_offset_from_left_px=0,
        y_offset_from_top_px=LARGE_BORDER_WIDTH_PX, text_string='(h)'
    )

    inverse_dwt_object = WaveTFFactory().build('haar', dim=2, inverse=True)
    probability_tensor = inverse_dwt_object.call(coeff_tensor_by_level[0])
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = (
        K.eval(probability_tensor)
    )
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = numpy.maximum(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], 0.
    )
    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY] = numpy.minimum(
        prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY], 1.
    )

    prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][..., 0] = (
        wavelet_utils.untaper_spatial_data(
            spatial_data_matrix=
            prediction_dict[prediction_io.PROBABILITY_MATRIX_KEY][..., 0],
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

    orig_file_name, figure_object = (
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
        )[:2]
    )

    panel_file_names[10] = '{0:s}/filtered_field.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(panel_file_names[10]))
    figure_object.savefig(
        panel_file_names[10], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[10],
        output_file_name=panel_file_names[10],
        border_width_pixels=LARGE_BORDER_WIDTH_PX
    )
    _overlay_text(
        image_file_name=panel_file_names[10],
        x_offset_from_left_px=0,
        y_offset_from_top_px=2 * LARGE_BORDER_WIDTH_PX, text_string='(k)'
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=panel_file_names[10],
        output_file_name=panel_file_names[10],
        border_width_pixels=SMALL_BORDER_WIDTH_PX
    )

    # Concatenate panels.
    concat_figure_file_name = '{0:s}/wavelet_procedure.jpg'.format(
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
