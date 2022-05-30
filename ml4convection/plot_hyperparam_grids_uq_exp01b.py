"""Plots evaluation scores vs. hyperparameters for UQ Experiment 1b."""

import os
import sys
import argparse
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
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TOP_LEVEL_SKIP_DROPOUT_RATES = numpy.linspace(0, 0.8, num=5)
PENULTIMATE_LAYER_DROPOUT_RATES = numpy.linspace(0.2, 0.8, num=5)
OUTPUT_LAYER_DROPOUT_RATES = numpy.linspace(0.4, 0.8, num=5)

DEFAULT_FONT_SIZE = 20

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

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=0.8, font_size=DEFAULT_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    S = number of dropout rates for top-level skip connection
    P = number of dropout rates for penultimate layer
    L = number of dropout rates for last (output) layer

    :param score_matrix: S-by-P-by-L numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix) + 0.
    scores_1d[numpy.isnan(scores_1d)] = -numpy.inf
    sort_indices_1d = numpy.argsort(-scores_1d)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-highest {1:s} = {2:.4g} ... dropout rate for top-level '
            'skip connection, second-last layer, last layer = '
            '{3:.3f}, {4:.3f}, {5:.3f}'
        ).format(
            m + 1, score_name, score_matrix[i, j, k],
            TOP_LEVEL_SKIP_DROPOUT_RATES[i], PENULTIMATE_LAYER_DROPOUT_RATES[j],
            OUTPUT_LAYER_DROPOUT_RATES[k]
        ))


def _run(experiment_dir_name, matching_distance_px, output_dir_name):
    """Plots evaluation scores vs. hyperparameters for UQ Experiment 1b.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param matching_distance_px: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_skip_level_rates = len(TOP_LEVEL_SKIP_DROPOUT_RATES)
    num_penultimate_layer_rates = len(PENULTIMATE_LAYER_DROPOUT_RATES)
    num_output_layer_rates = len(OUTPUT_LAYER_DROPOUT_RATES)
    dimensions = (
        num_skip_level_rates, num_penultimate_layer_rates,
        num_output_layer_rates
    )

    aupd_matrix = numpy.full(dimensions, numpy.nan)
    max_csi_matrix = numpy.full(dimensions, numpy.nan)
    fss_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)

    y_tick_labels = ['{0:.3f}'.format(d) for d in TOP_LEVEL_SKIP_DROPOUT_RATES]
    x_tick_labels = [
        '{0:.3f}'.format(d) for d in PENULTIMATE_LAYER_DROPOUT_RATES
    ]

    y_axis_label = 'Dropout rate for third-last layer'
    x_axis_label = 'Dropout rate for second-last layer'

    for i in range(num_skip_level_rates):
        for j in range(num_penultimate_layer_rates):
            for k in range(num_output_layer_rates):
                this_score_file_name = (
                    '{0:s}/top-level-skip-dropout={1:.3f}_'
                    'penultimate-layer-dropout={2:.3f}_'
                    'output-layer-dropout={3:.3f}/'
                    'validation_sans_dropout/partial_grids/evaluation/'
                    'matching_distance_px={4:.6f}/'
                    'advanced_scores_gridded=0.p'
                ).format(
                    experiment_dir_name, TOP_LEVEL_SKIP_DROPOUT_RATES[i],
                    PENULTIMATE_LAYER_DROPOUT_RATES[j],
                    OUTPUT_LAYER_DROPOUT_RATES[k],
                    matching_distance_px
                )

                print('Reading data from: "{0:s}"...'.format(
                    this_score_file_name
                ))
                t = evaluation.read_advanced_score_file(this_score_file_name)

                aupd_matrix[i, j, k] = (
                    gg_model_eval.get_area_under_perf_diagram(
                        pod_by_threshold=numpy.mean(
                            t[evaluation.POD_KEY].values, axis=0
                        ),
                        success_ratio_by_threshold=numpy.mean(
                            t[evaluation.SUCCESS_RATIO_KEY].values, axis=0
                        )
                    )
                )

                max_csi_matrix[i, j, k] = numpy.nanmax(
                    numpy.mean(t[evaluation.CSI_KEY].values, axis=0)
                )
                fss_matrix[i, j, k] = numpy.mean(t[evaluation.FSS_KEY].values)
                bss_matrix[i, j, k] = numpy.mean(
                    t[evaluation.BRIER_SKILL_SCORE_KEY].values
                )

    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=aupd_matrix, score_name='AUPD')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=max_csi_matrix, score_name='Max CSI')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=fss_matrix, score_name='FSS')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=bss_matrix, score_name='BSS')
    print(SEPARATOR_STRING)

    for k in range(num_output_layer_rates):

        # Plot AUPD.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=aupd_matrix[..., k],
            min_colour_value=numpy.nanpercentile(aupd_matrix, 1),
            max_colour_value=numpy.nanpercentile(aupd_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)

        axes_object.set_title(
            'AUPD; dropout rate for last layer = {0:.3f}'.format(
                OUTPUT_LAYER_DROPOUT_RATES[k]
            )
        )
        figure_file_name = '{0:s}/aupd_output-layer-dropout={1:.3f}.jpg'.format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot max CSI.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=max_csi_matrix[..., k],
            min_colour_value=numpy.nanpercentile(max_csi_matrix, 1),
            max_colour_value=numpy.nanpercentile(max_csi_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)

        axes_object.set_title(
            'Max CSI; dropout rate for last layer = {0:.3f}'.format(
                OUTPUT_LAYER_DROPOUT_RATES[k]
            )
        )
        figure_file_name = '{0:s}/csi_output-layer-dropout={1:.3f}.jpg'.format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot FSS.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=fss_matrix[..., k],
            min_colour_value=numpy.nanpercentile(fss_matrix, 1),
            max_colour_value=numpy.nanpercentile(fss_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)

        axes_object.set_title(
            'FSS; dropout rate for last layer = {0:.3f}'.format(
                OUTPUT_LAYER_DROPOUT_RATES[k]
            )
        )
        figure_file_name = '{0:s}/fss_output-layer-dropout={1:.3f}.jpg'.format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot BSS.
        this_max_value = numpy.nanpercentile(numpy.absolute(bss_matrix), 99.)
        this_max_value = min([this_max_value, 1.])
        this_min_value = -1 * this_max_value

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=bss_matrix[..., k],
            min_colour_value=this_min_value, max_colour_value=this_max_value,
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=BSS_COLOUR_MAP_OBJECT
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)

        axes_object.set_title(
            'BSS; dropout rate for last layer = {0:.3f}'.format(
                OUTPUT_LAYER_DROPOUT_RATES[k]
            )
        )
        figure_file_name = '{0:s}/bss_output-layer-dropout={1:.3f}.jpg'.format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        matching_distance_px=getattr(
            INPUT_ARG_OBJECT, MATCHING_DISTANCE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
