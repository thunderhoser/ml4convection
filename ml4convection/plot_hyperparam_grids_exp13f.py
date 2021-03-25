"""Plots scores on hyperparameter grid for Experiment 13f."""

import os
import sys
import argparse
from PIL import Image
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import evaluation
import gg_model_evaluation as gg_model_eval
import gg_plotting_utils
import imagemagick_utils
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

L2_WEIGHTS = numpy.logspace(-7, -5, num=5)

LAG_TIME_STRINGS_SEC = [
    '0', '0-600', '0-600-1200', '0-600-1200-1800', '0-600-1200-1800-2400',
    '0-600-1200-1800-2400-3000', '0-600-1200-1800-2400-3000-3600',
    '0-600-1200-1800-2400-3000-3600-4200',
    '0-600-1200-1800-2400-3000-3600-4200-4800',
    '0-600-1200-1800-2400-3000-3600-4200-4800-5400',
    '0-600-1200-1800-2400-3000-3600-4200-4800-5400-6000',
    '0-600-1200-1800-2400-3000-3600-4200-4800-5400-6000-6600',
    '0-600-1200-1800-2400-3000-3600-4200-4800-5400-6000-6600-7200',
    '0-1200', '0-1200-2400', '0-1200-2400-3600', '0-1200-2400-3600-4800',
    '0-1200-2400-3600-4800-6000', '0-1200-2400-3600-4800-6000-7200',
    '0-1200-2400-3600-4800-6000-7200-8400',
    '0-1200-2400-3600-4800-6000-7200-8400-9600',
    '0-1200-2400-3600-4800-6000-7200-8400-9600-10800',
    '0-1200-2400-3600-4800-6000-7200-8400-9600-10800-12000',
    '0-1200-2400-3600-4800-6000-7200-8400-9600-10800-12000-13200',
    '0-1200-2400-3600-4800-6000-7200-8400-9600-10800-12000-13200-14400',
    '0-1800', '0-1800-3600', '0-1800-3600-5400', '0-1800-3600-5400-7200',
    '0-1800-3600-5400-7200-9000', '0-1800-3600-5400-7200-9000-10800',
    '0-1800-3600-5400-7200-9000-10800-12600',
    '0-1800-3600-5400-7200-9000-10800-12600-14400',
    '0-1800-3600-5400-7200-9000-10800-12600-14400-16200',
    '0-1800-3600-5400-7200-9000-10800-12600-14400-16200-18000',
    '0-1800-3600-5400-7200-9000-10800-12600-14400-16200-18000-19800',
    '0-1800-3600-5400-7200-9000-10800-12600-14400-16200-18000-19800-21600',
    '0-2400', '0-2400-4800', '0-2400-4800-7200', '0-2400-4800-7200-9600',
    '0-2400-4800-7200-9600-12000', '0-2400-4800-7200-9600-12000-14400',
    '0-2400-4800-7200-9600-12000-14400-16800',
    '0-2400-4800-7200-9600-12000-14400-16800-19200',
    '0-2400-4800-7200-9600-12000-14400-16800-19200-21600',
    '0-3600', '0-3600-7200', '0-3600-7200-10800', '0-3600-7200-10800-14400',
    '0-3600-7200-10800-14400-18000', '0-3600-7200-10800-14400-18000-21600'
]

LAG_TIME_STRINGS_MINUTES = [
    '0', '0,10', '0,10,20', '0,10,20,30', '0,10,20,30,40',
    '0,10,20,30,40,50', '0,10,20,30,40,50,60',
    '0,10,20,30,40,50,60,70',
    '0,10,20,30,40,50,60,70,80',
    '0,10,20,30,40,50,60,70,80,90',
    '0,10,20,30,40,50,60,70,80,90,100',
    '0,10,20,30,40,50,60,70,80,90,100,110',
    '0,10,20,30,40,50,60,70,80,90,100,110,120',
    '0,20', '0,20,40', '0,20,40,60', '0,20,40,60,80',
    '0,20,40,60,80,100', '0,20,40,60,80,100,120',
    '0,20,40,60,80,100,120,140',
    '0,20,40,60,80,100,120,140,160',
    '0,20,40,60,80,100,120,140,160,180',
    '0,20,40,60,80,100,120,140,160,180,200',
    '0,20,40,60,80,100,120,140,160,180,200,220',
    '0,20,40,60,80,100,120,140,160,180,200,220,240',
    '0,30', '0,30,60', '0,30,60,90', '0,30,60,90,120',
    '0,30,60,90,120,150', '0,30,60,90,120,150,180',
    '0,30,60,90,120,150,180,210',
    '0,30,60,90,120,150,180,210,240',
    '0,30,60,90,120,150,180,210,240,270',
    '0,30,60,90,120,150,180,210,240,270,300',
    '0,30,60,90,120,150,180,210,240,270,300,330',
    '0,30,60,90,120,150,180,210,240,270,300,360',
    '0,40', '0,40,80', '0,40,80,120', '0,40,80,120,160',
    '0,40,80,120,160,200', '0,40,80,120,160,200,240',
    '0,40,80,120,160,200,240,280',
    '0,40,80,120,160,200,240,280,320',
    '0,40,80,120,160,200,240,280,320,360',
    '0,60', '0,60,120', '0,60,120,180', '0,60,120,180,240',
    '0,60,120,180,240,300', '0,60,120,180,240,300,360'
]

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.3
MARKER_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.2
SELECTED_MARKER_INDICES = numpy.array([3, 13], dtype=int)

DEFAULT_FONT_SIZE = 20
COLOUR_BAR_FONT_SIZE = 25

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

DEFAULT_COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
BSS_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
MATCHING_DISTANCE_ARG_NAME = 'matching_distance_px'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT_DIR_HELP_STRING = (
    'Name of top-level directory with models.  Evaluation scores will be found '
    'therein.'
)
MATCHING_DISTANCE_HELP_STRING = (
    'Matching distance for neighbourhood evaluation (pixels).  Will plot scores'
    ' at this matching distance.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MATCHING_DISTANCE_ARG_NAME, type=float, required=True,
    help=MATCHING_DISTANCE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels, colour_map_object):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(x_tick_values, x_tick_labels, rotation=90.)
    pyplot.yticks(y_tick_values, y_tick_labels)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    W = number of L_2 weights
    L = number of lag-time combos

    :param score_matrix: W-by-L numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix) + 0.
    scores_1d[numpy.isnan(scores_1d)] = -numpy.inf
    sort_indices_1d = numpy.argsort(-scores_1d)

    i_sort_indices, j_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        j = j_sort_indices[k]

        print((
            '{0:d}th-highest {1:s} = {2:.4g} ... L_2 weight = 10^{3:.1f} ...'
            'lag times in seconds = {4:s}'
        ).format(
            k + 1, score_name, score_matrix[i, j],
            numpy.log10(L2_WEIGHTS[i]), LAG_TIME_STRINGS_SEC[j]
        ))


def _add_markers(figure_object, axes_object, best_marker_indices):
    """Adds markers to figure.

    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param best_marker_indices: length-2 numpy array of array indices for best
        model.
    """

    figure_width_px = (
        figure_object.get_size_inches()[0] * figure_object.dpi
    )
    marker_size_px = figure_width_px * (
        BEST_MARKER_SIZE_GRID_CELLS / len(LAG_TIME_STRINGS_SEC)
    )

    axes_object.plot(
        best_marker_indices[1], best_marker_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    figure_width_px = (
        figure_object.get_size_inches()[0] * figure_object.dpi
    )
    marker_size_px = figure_width_px * (
        BEST_MARKER_SIZE_GRID_CELLS / len(LAG_TIME_STRINGS_SEC)
    )

    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )


def _add_colour_bar(
        figure_file_name, colour_map_object, min_colour_value, max_colour_value,
        temporary_dir_name):
    """Adds colour bar to saved image file.

    :param figure_file_name: Path to saved image file.  Colour bar will be added
        to this image.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`
        or similar).
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
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

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )
    dummy_values = numpy.array([min_colour_value, max_colour_value])

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=extra_axes_object, data_matrix=dummy_values,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=1.25, font_size=COLOUR_BAR_FONT_SIZE
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


def _run(experiment_dir_name, matching_distance_px, output_dir_name):
    """Plots scores on hyperparameter grid for Experiment 13f.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param matching_distance_px: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_l2_weights = len(L2_WEIGHTS)
    num_lag_time_sets = len(LAG_TIME_STRINGS_SEC)
    dimensions = (num_l2_weights, num_lag_time_sets)

    aupd_matrix = numpy.full(dimensions, numpy.nan)
    max_csi_matrix = numpy.full(dimensions, numpy.nan)
    fss_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)

    x_tick_labels = LAG_TIME_STRINGS_MINUTES
    y_tick_labels = [
        '{0:.1f}'.format(numpy.log10(w)) for w in L2_WEIGHTS
    ]
    y_tick_labels = [r'10$^{' + l + '}$' for l in y_tick_labels]

    x_axis_label = 'Lag times (minutes)'
    y_axis_label = r'L$_{2}$ weight'

    for i in range(num_l2_weights):
        for j in range(num_lag_time_sets):
            this_score_file_name = (
                '{0:s}/l2-weight={1:.10f}_lag-times-sec={2:s}/'
                'validation/partial_grids/evaluation/'
                'matching_distance_px={3:.6f}/'
                'advanced_scores_gridded=0.p'
            ).format(
                experiment_dir_name, L2_WEIGHTS[i],
                LAG_TIME_STRINGS_SEC[j], matching_distance_px
            )

            print('Reading data from: "{0:s}"...'.format(
                this_score_file_name
            ))
            t = evaluation.read_advanced_score_file(this_score_file_name)

            aupd_matrix[i, j] = gg_model_eval.get_area_under_perf_diagram(
                pod_by_threshold=t[evaluation.POD_KEY].values,
                success_ratio_by_threshold=
                t[evaluation.SUCCESS_RATIO_KEY].values
            )
            max_csi_matrix[i, j] = numpy.nanmax(t[evaluation.CSI_KEY].values)
            fss_matrix[i, j] = t[evaluation.FSS_KEY].values[0]
            bss_matrix[i, j] = t[evaluation.BRIER_SKILL_SCORE_KEY].values[0]

    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=aupd_matrix, score_name='AUPD')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=max_csi_matrix, score_name='Max CSI')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=fss_matrix, score_name='FSS')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=bss_matrix, score_name='BSS')
    print(SEPARATOR_STRING)

    this_index = numpy.argmax(numpy.ravel(aupd_matrix))
    max_aupd_indices = numpy.unravel_index(this_index, aupd_matrix.shape)

    this_index = numpy.argmax(numpy.ravel(max_csi_matrix))
    max_csi_indices = numpy.unravel_index(this_index, max_csi_matrix.shape)

    this_index = numpy.argmax(numpy.ravel(fss_matrix))
    max_fss_indices = numpy.unravel_index(this_index, fss_matrix.shape)

    this_index = numpy.argmax(numpy.ravel(bss_matrix))
    max_bss_indices = numpy.unravel_index(this_index, bss_matrix.shape)

    # Plot AUPD.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=aupd_matrix,
        min_colour_value=numpy.nanpercentile(aupd_matrix, 1),
        max_colour_value=numpy.nanpercentile(aupd_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=max_aupd_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)

    axes_object.set_title('Area under performance diagram')
    figure_file_name = '{0:s}/aupd_grid.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=numpy.nanpercentile(aupd_matrix, 1),
        max_colour_value=numpy.nanpercentile(aupd_matrix, 99),
        temporary_dir_name=output_dir_name
    )

    # Plot max CSI.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=max_csi_matrix,
        min_colour_value=numpy.nanpercentile(max_csi_matrix, 1),
        max_colour_value=numpy.nanpercentile(max_csi_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=max_csi_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)

    axes_object.set_title('Critical success index')
    figure_file_name = '{0:s}/csi_grid.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=numpy.nanpercentile(max_csi_matrix, 1),
        max_colour_value=numpy.nanpercentile(max_csi_matrix, 99),
        temporary_dir_name=output_dir_name
    )

    # Plot FSS.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=fss_matrix,
        min_colour_value=numpy.nanpercentile(fss_matrix, 1),
        max_colour_value=numpy.nanpercentile(fss_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=max_fss_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)

    axes_object.set_title('Fractions skill score')
    figure_file_name = '{0:s}/fss_grid.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=numpy.nanpercentile(fss_matrix, 1),
        max_colour_value=numpy.nanpercentile(fss_matrix, 99),
        temporary_dir_name=output_dir_name
    )

    # Plot BSS.
    this_max_value = numpy.nanpercentile(numpy.absolute(bss_matrix), 99.)
    this_max_value = min([this_max_value, 1.])
    this_min_value = -1 * this_max_value

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=bss_matrix,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=BSS_COLOUR_MAP_OBJECT
    )

    _add_markers(
        figure_object=figure_object, axes_object=axes_object,
        best_marker_indices=max_bss_indices
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)

    axes_object.set_title('Brier skill score')
    figure_file_name = '{0:s}/bss_grid.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _add_colour_bar(
        figure_file_name=figure_file_name,
        colour_map_object=BSS_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        temporary_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        matching_distance_px=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
