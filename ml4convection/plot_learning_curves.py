"""Plots learning curves."""

import os
import sys
import glob
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import learning_curves
import radar_utils

TOLERANCE = 1e-6
SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

GRID_SPACING_DEG = 0.0125
LARGE_RESOLUTION_DEG = 1e6
NUM_RADARS = len(radar_utils.RADAR_LATITUDES_DEG_N)

ORANGE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
PURPLE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
GREEN_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
BLACK_COLOUR = numpy.full(3, 0.)

LINE_COLOURS = [ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR] * 2
LINE_STYLES = ['solid'] * 4 + ['dashed'] * 4
MAX_LINES_PER_GRAPH = 8

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
LEGEND_FONT_SIZE = 24

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

INPUT_DIR_ARG_NAME = 'input_top_model_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level model directory, containing one subdirectory for each '
    'epoch.  Will find scores at each epoch in these subdirectories.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _input_file_name_to_epoch(input_file_name):
    """Parses epoch from name of input file (containing advanced scores).

    :param input_file_name: Path to input file.
    :return: epoch_index: Epoch (integer).
    """

    subdir_name = input_file_name.split('/')[-5]
    epoch_word = subdir_name.split('_')[1]

    assert epoch_word.startswith('epoch=')
    return int(epoch_word.replace('epoch=', ''))


def _plot_one_learning_curve(score_matrix, epoch_indices, legend_strings,
                             is_positively_oriented, is_dice_coeff):
    """Plots one learning curve.

    E = number of epochs
    C = number of configurations for score (either neigh distances or resolution
        bands)

    :param score_matrix: C-by-E numpy array of scores.
    :param epoch_indices: length-E numpy array of epoch indices.
    :param legend_strings: length-C list of legend strings.
    :param is_positively_oriented: Boolean flag.
    :param is_dice_coeff: Boolean flag.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_configs = score_matrix.shape[1]
    legend_handles = [None] * num_configs

    for j in range(num_configs):
        legend_handles[j] = axes_object.plot(
            epoch_indices, score_matrix[:, j], color=LINE_COLOURS[j],
            linestyle=LINE_STYLES[j], linewidth=LINE_WIDTH,
            marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
            markerfacecolor=LINE_COLOURS[j], markeredgecolor=LINE_COLOURS[j]
        )[0]

    if is_dice_coeff:
        axes_object.legend(
            legend_handles, legend_strings, loc='lower right',
            bbox_to_anchor=(1, 0), fancybox=True, shadow=True, ncol=2,
            fontsize=LEGEND_FONT_SIZE
        )

        max_value = axes_object.get_ylim()[1]
        min_value = numpy.percentile(score_matrix, 1.)
        axes_object.set_ylim(min_value, max_value)

    elif is_positively_oriented:
        axes_object.legend(
            legend_handles, legend_strings, loc='upper left',
            bbox_to_anchor=(0, 1), fancybox=True, shadow=True, ncol=2,
            fontsize=LEGEND_FONT_SIZE
        )
    else:
        axes_object.legend(
            legend_handles, legend_strings, loc='upper right',
            bbox_to_anchor=(1, 1), fancybox=True, shadow=True, ncol=2,
            fontsize=LEGEND_FONT_SIZE
        )

        min_value = axes_object.get_ylim()[0]
        max_value = numpy.percentile(score_matrix, 97.5)
        axes_object.set_ylim(min_value, max_value)

    axes_object.set_xlabel('Epoch')

    x_tick_values = axes_object.get_xticks()
    x_tick_labels = ['{0:d}'.format(int(numpy.round(v))) for v in x_tick_values]
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels)

    return figure_object, axes_object


def _run(top_model_dir_name, output_dir_name):
    """Plots learning curves.

    This is effectively the main method.

    :param top_model_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    :raises: ValueError: if cannot find input files, containing scores to plot.
    """

    input_file_pattern = (
        '{0:s}/model_epoch=[0-9][0-9][0-9]_'
        'val-loss=[0-9].[0-9][0-9][0-9][0-9][0-9][0-9]/'
        'validation/partial_grids/learning_curves/advanced_scores.nc'
    ).format(top_model_dir_name)

    input_file_names = glob.glob(input_file_pattern)

    if len(input_file_names) == 0:
        error_string = 'Cannot find any files with pattern: {0:s}'.format(
            input_file_pattern
        )
        raise ValueError(error_string)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    advanced_score_tables_xarray = []
    epoch_indices = []
    num_neigh_distances = 0
    num_fourier_bands = 0

    for this_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        advanced_score_tables_xarray.append(
            learning_curves.read_scores(this_file_name)
        )
        epoch_indices.append(
            _input_file_name_to_epoch(this_file_name)
        )

        a = advanced_score_tables_xarray
        if len(a) == 1:
            continue

        if learning_curves.NEIGH_DISTANCE_DIM in a[0].coords:
            num_neigh_distances = len(
                a[0].coords[learning_curves.NEIGH_DISTANCE_DIM].values
            )
            assert num_neigh_distances <= MAX_LINES_PER_GRAPH

        if learning_curves.MIN_RESOLUTION_DIM in a[0].coords:
            num_fourier_bands = len(
                a[0].coords[learning_curves.MIN_RESOLUTION_DIM].values
            )
            assert num_fourier_bands <= MAX_LINES_PER_GRAPH

        # assert (
        #     a[-1].attrs[learning_curves.MODEL_FILE_KEY] ==
        #     a[0].attrs[learning_curves.MODEL_FILE_KEY]
        # )

        if learning_curves.NEIGH_DISTANCE_DIM in a[0].coords:
            assert numpy.allclose(
                a[0].coords[learning_curves.NEIGH_DISTANCE_DIM].values,
                a[-1].coords[learning_curves.NEIGH_DISTANCE_DIM].values,
                atol=TOLERANCE
            )

        if learning_curves.MIN_RESOLUTION_DIM in a[0].coords:
            assert numpy.allclose(
                a[0].coords[learning_curves.MIN_RESOLUTION_DIM].values,
                a[-1].coords[learning_curves.MIN_RESOLUTION_DIM].values,
                atol=TOLERANCE, equal_nan=True
            )

            first_max_resolutions_deg = (
                a[0].coords[learning_curves.MAX_RESOLUTION_DIM].values + 0.
            )
            last_max_resolutions_deg = (
                a[-1].coords[learning_curves.MAX_RESOLUTION_DIM].values + 0.
            )

            first_max_resolutions_deg[
                numpy.isinf(first_max_resolutions_deg)
            ] = numpy.nan

            last_max_resolutions_deg[
                numpy.isinf(last_max_resolutions_deg)
            ] = numpy.nan

            assert numpy.allclose(
                first_max_resolutions_deg, last_max_resolutions_deg,
                atol=TOLERANCE, equal_nan=True
            )

    epoch_indices = numpy.array(epoch_indices, dtype=int)
    sort_indices = numpy.argsort(epoch_indices)
    epoch_indices = epoch_indices[sort_indices]
    advanced_score_tables_xarray = [
        advanced_score_tables_xarray[k] for k in sort_indices
    ]

    if num_neigh_distances > 0:
        neigh_distances_px = advanced_score_tables_xarray[0].coords[
            learning_curves.NEIGH_DISTANCE_DIM
        ].values

        neigh_distances_deg = GRID_SPACING_DEG * neigh_distances_px
        legend_strings = ['{0:.3g}'.format(d) for d in neigh_distances_deg]
        legend_strings = [s + r'$^{\circ}$' for s in legend_strings]

        # Plot Brier score.
        brier_score_matrix = numpy.vstack([
            a[learning_curves.NEIGH_BRIER_SCORE_KEY].values
            for a in advanced_score_tables_xarray
        ])

        figure_object, axes_object = _plot_one_learning_curve(
            score_matrix=brier_score_matrix, epoch_indices=epoch_indices,
            legend_strings=legend_strings, is_positively_oriented=False,
            is_dice_coeff=False
        )
        axes_object.set_ylim(bottom=0.)
        axes_object.set_ylabel('Brier score')
        axes_object.set_title('Neighbourhood-based Brier score')

        output_file_name = '{0:s}/neigh_brier_score.jpg'.format(output_dir_name)
        print('Saving figure to file: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot FSS.
        fss_matrix = numpy.vstack([
            a[learning_curves.NEIGH_FSS_KEY].values
            for a in advanced_score_tables_xarray
        ])

        figure_object, axes_object = _plot_one_learning_curve(
            score_matrix=fss_matrix, epoch_indices=epoch_indices,
            legend_strings=legend_strings, is_positively_oriented=True,
            is_dice_coeff=False
        )
        axes_object.set_ylim(bottom=0.)
        axes_object.set_ylabel('FSS')
        axes_object.set_title('Neighbourhood-based fractions skill score (FSS)')

        output_file_name = '{0:s}/neigh_fss.jpg'.format(output_dir_name)
        print('Saving figure to file: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot IOU.
        iou_matrix = numpy.vstack([
            a[learning_curves.NEIGH_IOU_KEY].values
            for a in advanced_score_tables_xarray
        ])

        figure_object, axes_object = _plot_one_learning_curve(
            score_matrix=iou_matrix, epoch_indices=epoch_indices,
            legend_strings=legend_strings, is_positively_oriented=True,
            is_dice_coeff=False
        )
        axes_object.set_ylim(bottom=0.)
        axes_object.set_ylabel('IOU')
        axes_object.set_title(
            'Neighbourhood-based intersection over union (IOU)'
        )

        output_file_name = '{0:s}/neigh_iou.jpg'.format(output_dir_name)
        print('Saving figure to file: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot Dice coefficient.
        dice_coeff_matrix = numpy.vstack([
            a[learning_curves.NEIGH_DICE_COEFF_KEY].values
            for a in advanced_score_tables_xarray
        ])

        figure_object, axes_object = _plot_one_learning_curve(
            score_matrix=dice_coeff_matrix, epoch_indices=epoch_indices,
            legend_strings=legend_strings, is_positively_oriented=True,
            is_dice_coeff=True
        )
        axes_object.set_ylim(top=1.)
        axes_object.set_ylabel('Dice coefficient')
        axes_object.set_title(
            'Neighbourhood-based Dice coefficient'
        )

        output_file_name = '{0:s}/neigh_dice_coeff.jpg'.format(output_dir_name)
        print('Saving figure to file: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot CSI.
        csi_matrix = numpy.vstack([
            a[learning_curves.NEIGH_CSI_KEY].values
            for a in advanced_score_tables_xarray
        ])

        figure_object, axes_object = _plot_one_learning_curve(
            score_matrix=csi_matrix, epoch_indices=epoch_indices,
            legend_strings=legend_strings, is_positively_oriented=True,
            is_dice_coeff=False
        )
        axes_object.set_ylim(bottom=0.)
        axes_object.set_ylabel('CSI')
        axes_object.set_title(
            'Neighbourhood-based critical success index (CSI)'
        )

        output_file_name = '{0:s}/neigh_csi.jpg'.format(output_dir_name)
        print('Saving figure to file: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    if num_fourier_bands == 0:
        return

    min_resolutions_deg = advanced_score_tables_xarray[0].coords[
        learning_curves.MIN_RESOLUTION_DIM
    ].values

    max_resolutions_deg = advanced_score_tables_xarray[0].coords[
        learning_curves.MAX_RESOLUTION_DIM
    ].values

    max_resolutions_deg[max_resolutions_deg > LARGE_RESOLUTION_DEG] = numpy.inf

    legend_strings = [
        '[{0:.3g}, {1:.3g}]'.format(a, b)
        for a, b in zip(min_resolutions_deg, max_resolutions_deg)
    ]
    legend_strings = [s.replace('inf]', r'$\infty$)') for s in legend_strings]
    legend_strings = [s + r'$^{\circ}$' for s in legend_strings]

    # Plot Brier score.
    brier_score_matrix = numpy.vstack([
        a[learning_curves.FOURIER_BRIER_SCORE_KEY].values
        for a in advanced_score_tables_xarray
    ])

    figure_object, axes_object = _plot_one_learning_curve(
        score_matrix=brier_score_matrix, epoch_indices=epoch_indices,
        legend_strings=legend_strings, is_positively_oriented=False,
        is_dice_coeff=False
    )
    axes_object.set_ylim(bottom=0.)
    axes_object.set_ylabel('Brier score')
    axes_object.set_title('Fourier-based Brier score')

    output_file_name = '{0:s}/fourier_brier_score.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot FSS.
    fss_matrix = numpy.vstack([
        a[learning_curves.FOURIER_FSS_KEY].values
        for a in advanced_score_tables_xarray
    ])

    figure_object, axes_object = _plot_one_learning_curve(
        score_matrix=fss_matrix, epoch_indices=epoch_indices,
        legend_strings=legend_strings, is_positively_oriented=True,
        is_dice_coeff=False
    )
    axes_object.set_ylim(bottom=0.)
    axes_object.set_ylabel('FSS')
    axes_object.set_title('Fourier-based fractions skill score (FSS)')

    output_file_name = '{0:s}/fourier_fss.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot IOU.
    iou_matrix = numpy.vstack([
        a[learning_curves.FOURIER_IOU_KEY].values
        for a in advanced_score_tables_xarray
    ])

    figure_object, axes_object = _plot_one_learning_curve(
        score_matrix=iou_matrix, epoch_indices=epoch_indices,
        legend_strings=legend_strings, is_positively_oriented=True,
        is_dice_coeff=False
    )
    axes_object.set_ylim(bottom=0.)
    axes_object.set_ylabel('IOU')
    axes_object.set_title('Fourier-based intersection over union (IOU)')

    output_file_name = '{0:s}/fourier_iou.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot Dice coefficient.
    dice_coeff_matrix = numpy.vstack([
        a[learning_curves.FOURIER_DICE_COEFF_KEY].values
        for a in advanced_score_tables_xarray
    ])

    figure_object, axes_object = _plot_one_learning_curve(
        score_matrix=dice_coeff_matrix, epoch_indices=epoch_indices,
        legend_strings=legend_strings, is_positively_oriented=True,
        is_dice_coeff=True
    )
    axes_object.set_ylim(top=1.)
    axes_object.set_ylabel('Dice coefficient')
    axes_object.set_title('Fourier-based Dice coefficient')

    output_file_name = '{0:s}/fourier_dice_coeff.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot CSI.
    csi_matrix = numpy.vstack([
        a[learning_curves.FOURIER_CSI_KEY].values
        for a in advanced_score_tables_xarray
    ])

    figure_object, axes_object = _plot_one_learning_curve(
        score_matrix=csi_matrix, epoch_indices=epoch_indices,
        legend_strings=legend_strings, is_positively_oriented=True,
        is_dice_coeff=False
    )
    axes_object.set_ylim(bottom=0.)
    axes_object.set_ylabel('CSI')
    axes_object.set_title('Fourier-based critical success index (CSI)')

    output_file_name = '{0:s}/fourier_csi.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot real part of frequency-space MSE.
    freq_mse_matrix_real = numpy.vstack([
        a[learning_curves.FREQ_MSE_REAL_KEY].values
        for a in advanced_score_tables_xarray
    ])

    figure_object, axes_object = _plot_one_learning_curve(
        score_matrix=freq_mse_matrix_real, epoch_indices=epoch_indices,
        legend_strings=legend_strings, is_positively_oriented=False,
        is_dice_coeff=False
    )
    axes_object.set_ylim(bottom=0.)
    axes_object.set_ylabel('MSE')
    axes_object.set_title('MSE for real part of Fourier spectrum')

    output_file_name = '{0:s}/freq_mse_real.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot imaginary part of frequency-space MSE.
    freq_mse_matrix_imag = numpy.vstack([
        a[learning_curves.FREQ_MSE_IMAGINARY_KEY].values
        for a in advanced_score_tables_xarray
    ])

    figure_object, axes_object = _plot_one_learning_curve(
        score_matrix=freq_mse_matrix_imag, epoch_indices=epoch_indices,
        legend_strings=legend_strings, is_positively_oriented=False,
        is_dice_coeff=False
    )
    axes_object.set_ylim(bottom=0.)
    axes_object.set_ylabel('MSE')
    axes_object.set_title('MSE for imaginary part of Fourier spectrum')

    output_file_name = '{0:s}/freq_mse_imaginary.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot total frequency-space MSE.
    freq_mse_matrix_total = numpy.vstack([
        a[learning_curves.FREQ_MSE_TOTAL_KEY].values
        for a in advanced_score_tables_xarray
    ])

    figure_object, axes_object = _plot_one_learning_curve(
        score_matrix=freq_mse_matrix_total, epoch_indices=epoch_indices,
        legend_strings=legend_strings, is_positively_oriented=False,
        is_dice_coeff=False
    )
    axes_object.set_ylim(bottom=0.)
    axes_object.set_ylabel('MSE')
    axes_object.set_title('MSE for total Fourier spectrum')

    output_file_name = '{0:s}/freq_mse_total.jpg'.format(output_dir_name)
    print('Saving figure to file: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_model_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
