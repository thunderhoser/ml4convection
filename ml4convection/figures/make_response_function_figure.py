"""Makes figure with response functions for Butterworth filter @ diff bands."""

import os
import argparse
import numpy
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4convection.utils import fourier_utils

GRID_SPACING_DEG = 0.0125
MIN_RESOLUTIONS_DEG = numpy.array([0, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8])
MAX_RESOLUTIONS_DEG = numpy.array([
    0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, numpy.inf
])

COLOUR_MAP_OBJECT = pyplot.get_cmap('cividis')
COLOUR_NORM_OBJECT = pyplot.Normalize(vmin=0., vmax=1.)

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 200
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

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

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
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


def _run(output_dir_name):
    """Makes figure with response functions for Butterworth filter @ diff bands.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    letter_label = None

    num_panels = len(MIN_RESOLUTIONS_DEG)
    panel_file_names = [''] * num_panels

    for i in range(num_panels):
        this_filter_matrix = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=numpy.ones((615, 615)), filter_order=2.,
            grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=MIN_RESOLUTIONS_DEG[i],
            max_resolution_metres=MAX_RESOLUTIONS_DEG[i]
        )

        panel_file_names[i] = (
            '{0:s}/butterworth_filter_{1:.4f}deg_{2:.4f}deg.jpg'
        ).format(
            output_dir_name, MIN_RESOLUTIONS_DEG[i], MAX_RESOLUTIONS_DEG[i]
        )

        this_title_string = (
            r'$\delta_{min}$ = ' + '{0:.3g}'.format(MIN_RESOLUTIONS_DEG[i]) +
            r'$^{\circ}$; $\delta_{max}$ = ' +
            '{0:.3g}'.format(MAX_RESOLUTIONS_DEG[i]) + r'$^{\circ}$'
        )

        _plot_fourier_weights(
            weight_matrix=this_filter_matrix,
            colour_map_object=COLOUR_MAP_OBJECT,
            colour_norm_object=COLOUR_NORM_OBJECT,
            title_string=this_title_string,
            output_file_name=panel_file_names[i]
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            border_width_pixels=LARGE_BORDER_WIDTH_PX
        )
        _overlay_text(
            image_file_name=panel_file_names[i],
            x_offset_from_left_px=0,
            y_offset_from_top_px=2 * LARGE_BORDER_WIDTH_PX,
            text_string='({0:s})'.format(letter_label)
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            border_width_pixels=SMALL_BORDER_WIDTH_PX
        )

    # Concatenate panels.
    concat_figure_file_name = '{0:s}/butterworth_filters.jpg'.format(
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
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
