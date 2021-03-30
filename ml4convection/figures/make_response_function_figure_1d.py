"""Makes figure with 1-D response functions for Butter filter @ diff bands."""

import argparse
import numpy
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4convection.utils import fourier_utils

COLOUR_LIST = [
    numpy.array([27, 158, 119]), numpy.array([217, 95, 2]),
    numpy.array([117, 112, 179]), numpy.full(3, 0.)
]

for k in range(len(COLOUR_LIST)):
    COLOUR_LIST[k] = COLOUR_LIST[k].astype(float) / 255

GRID_SPACING_DEG = 0.0125
NUM_GRID_ROWS = 615

FIRST_MIN_RESOLUTIONS_DEG = numpy.array([0, 0.0125, 0.025, 0.05])
FIRST_MAX_RESOLUTIONS_DEG = numpy.array([0.0125, 0.025, 0.05, 0.1])
SECOND_MIN_RESOLUTIONS_DEG = numpy.array([0.1, 0.2, 0.4, 0.8])
SECOND_MAX_RESOLUTIONS_DEG = numpy.array([0.2, 0.4, 0.8, numpy.inf])

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FILTER_ORDER_ARG_NAME = 'filter_order'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

FILTER_ORDER_HELP_STRING = 'Order of Butterworth filter.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + FILTER_ORDER_ARG_NAME, type=float, required=False, default=2.,
    help=FILTER_ORDER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_fourier_weights(
        weight_matrix_3d, min_resolutions_deg, max_resolutions_deg,
        max_resolution_to_plot_deg, output_file_name):
    """Plots Fourier weights in 1-D space.

    M = number of rows in spatial grid
    N = number of columns in spatial grid
    B = number of bands

    :param weight_matrix_3d: M-by-N-by-B numpy array of weights.
    :param min_resolutions_deg: length-B numpy array of minimum resolutions
        allowed (degrees).
    :param max_resolutions_deg: length-B numpy array of max resolutions allowed
        (degrees).
    :param max_resolution_to_plot_deg: Max resolution to plot (degrees).
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    x_resolution_matrix_deg, y_resolution_matrix_deg = (
        fourier_utils._get_spatial_resolutions(
            num_grid_rows=weight_matrix_3d.shape[0],
            num_grid_columns=weight_matrix_3d.shape[1],
            grid_spacing_metres=GRID_SPACING_DEG
        )
    )

    resolution_matrix_deg = numpy.sqrt(
        x_resolution_matrix_deg ** 2 + y_resolution_matrix_deg ** 2
    )

    num_half_rows = int(float(NUM_GRID_ROWS + 1) / 2)
    resolutions_deg = numpy.diag(
        resolution_matrix_deg[:num_half_rows, :num_half_rows]
    )

    num_resolutions = len(resolutions_deg)
    num_bands = weight_matrix_3d.shape[2]
    absolute_weight_matrix_2d = numpy.full(
        (num_resolutions, num_bands), numpy.nan
    )

    for j in range(num_bands):
        absolute_weight_matrix_2d[:, j] = numpy.diag(
            weight_matrix_3d[:num_half_rows, :num_half_rows, j]
        )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    legend_handles = [None] * num_bands
    legend_strings = [''] * num_bands

    for j in range(num_bands):
        legend_handles[j] = axes_object.plot(
            2 * resolutions_deg, absolute_weight_matrix_2d[:, j],
            color=COLOUR_LIST[j], linestyle='solid', linewidth=3
        )[0]

        legend_strings[j] = (
            r'$\delta_{min}$ = ' + '{0:.3g}'.format(min_resolutions_deg[j]) +
            r'$^{\circ}$; $\delta_{max}$ = ' +
            '{0:.3g}'.format(max_resolutions_deg[j]) + r'$^{\circ}$'
        )
        legend_strings[j] = legend_strings[j].replace('inf', r'$\infty$')

    axes_object.set_xlim(0., 2 * max_resolution_to_plot_deg)
    axes_object.set_xlabel(r'Total wavelength ($^{\circ}$)')
    axes_object.set_ylabel(r'Response')

    axes_object.legend(
        legend_handles, legend_strings, loc='center right',
        bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(filter_order, output_dir_name):
    """Makes figure with 1-D response functions for Butter filter @ diff bands.

    This is effectively the main method.

    :param filter_order: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_bands = len(FIRST_MIN_RESOLUTIONS_DEG)
    filter_matrix = numpy.full(
        (NUM_GRID_ROWS, NUM_GRID_ROWS, num_bands), numpy.nan
    )

    for j in range(num_bands):
        filter_matrix[..., j] = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=numpy.ones((NUM_GRID_ROWS, NUM_GRID_ROWS)),
            filter_order=filter_order, grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=FIRST_MIN_RESOLUTIONS_DEG[j],
            max_resolution_metres=FIRST_MAX_RESOLUTIONS_DEG[j]
        )

    _plot_fourier_weights(
        weight_matrix_3d=filter_matrix,
        min_resolutions_deg=FIRST_MIN_RESOLUTIONS_DEG,
        max_resolutions_deg=FIRST_MAX_RESOLUTIONS_DEG,
        max_resolution_to_plot_deg=1.,
        output_file_name=
        '{0:s}/first_response_functions.jpg'.format(output_dir_name)
    )

    num_bands = len(SECOND_MIN_RESOLUTIONS_DEG)
    filter_matrix = numpy.full(
        (NUM_GRID_ROWS, NUM_GRID_ROWS, num_bands), numpy.nan
    )

    for j in range(num_bands):
        filter_matrix[..., j] = fourier_utils.apply_butterworth_filter(
            coefficient_matrix=numpy.ones((NUM_GRID_ROWS, NUM_GRID_ROWS)),
            filter_order=filter_order, grid_spacing_metres=GRID_SPACING_DEG,
            min_resolution_metres=SECOND_MIN_RESOLUTIONS_DEG[j],
            max_resolution_metres=SECOND_MAX_RESOLUTIONS_DEG[j]
        )

    _plot_fourier_weights(
        weight_matrix_3d=filter_matrix,
        min_resolutions_deg=SECOND_MIN_RESOLUTIONS_DEG,
        max_resolutions_deg=SECOND_MAX_RESOLUTIONS_DEG,
        max_resolution_to_plot_deg=5.,
        output_file_name=
        '{0:s}/second_response_functions.jpg'.format(output_dir_name)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        filter_order=getattr(INPUT_ARG_OBJECT, FILTER_ORDER_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
