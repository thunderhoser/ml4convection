"""Plots evaluation scores vs. hyperparameters for UQ Experiment 1."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from scipy.stats import rankdata

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import evaluation
import uq_evaluation
import gg_model_evaluation as gg_model_eval
import gg_plotting_utils
import imagemagick_utils
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TOP_LEVEL_SKIP_DROPOUT_RATES = numpy.linspace(0, 0.8, num=5)
PENULTIMATE_LAYER_DROPOUT_RATES = numpy.linspace(0.2, 0.8, num=5)
OUTPUT_LAYER_DROPOUT_RATES = numpy.linspace(0.4, 0.8, num=5)

DEFAULT_FONT_SIZE = 34

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

PANEL_SIZE_PX = int(2.5e6)

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


def _print_ranking_all_scores(
        aupd_matrix, csi_matrix, fss_matrix, bss_matrix, ssrel_matrix,
        mean_predictive_stdev_matrix, monotonicity_fraction_matrix,
        rank_mainly_by_fss):
    """Prints ranking for all scores.

    S = number of dropout rates for top-level skip connection
    P = number of dropout rates for penultimate layer
    L = number of dropout rates for last (output) layer

    :param aupd_matrix: S-by-P-by-L numpy array with AUPD (area under
        performance diagram).
    :param csi_matrix: Same but for critical success index.
    :param fss_matrix: Same but for fractions skill score.
    :param bss_matrix: Same but for Brier skill score.
    :param ssrel_matrix: Same but for spread-skill reliability.
    :param mean_predictive_stdev_matrix: Same but for mean stdev of predictive
        distribution.
    :param monotonicity_fraction_matrix: Same but for monotonicity fraction.
    :param rank_mainly_by_fss: Boolean flag.  If True (False), will rank mainly
        by FSS (SSREL).
    """

    if rank_mainly_by_fss:
        these_scores = -1 * numpy.ravel(fss_matrix)
        these_scores[numpy.isnan(these_scores)] = -numpy.inf
    else:
        these_scores = numpy.ravel(ssrel_matrix)
        these_scores[numpy.isnan(these_scores)] = numpy.inf

    sort_indices_1d = numpy.argsort(these_scores)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, fss_matrix.shape
    )

    these_scores = -1 * numpy.ravel(aupd_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    aupd_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), aupd_matrix.shape
    )

    these_scores = -1 * numpy.ravel(csi_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    csi_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), csi_matrix.shape
    )

    these_scores = -1 * numpy.ravel(fss_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    fss_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), fss_matrix.shape
    )

    these_scores = -1 * numpy.ravel(bss_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    bss_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), bss_matrix.shape
    )

    these_scores = numpy.ravel(ssrel_matrix)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    ssrel_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), ssrel_matrix.shape
    )

    these_scores = -1 * numpy.ravel(mean_predictive_stdev_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    stdev_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        mean_predictive_stdev_matrix.shape
    )

    these_scores = -1 * numpy.ravel(monotonicity_fraction_matrix)
    these_scores[numpy.isnan(these_scores)] = -numpy.inf
    mf_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        monotonicity_fraction_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-best model ... '
            'dropout rates = {1:.3f}, {2:.3f}, {3:.3f} ... '
            'AUPD rank = {4:.1f} ... CSI rank = {5:.1f} ... '
            'FSS rank = {6:.1f} ... BSS rank = {7:.1f} ... '
            'SSREL rank = {8:.1f} ... MF rank = {9:.1f} ... '
            'predictive-stdev rank = {10:.1f}'
        ).format(
            m + 1, TOP_LEVEL_SKIP_DROPOUT_RATES[i],
            PENULTIMATE_LAYER_DROPOUT_RATES[j],
            OUTPUT_LAYER_DROPOUT_RATES[k],
            aupd_rank_matrix[i, j, k], csi_rank_matrix[i, j, k],
            fss_rank_matrix[i, j, k], bss_rank_matrix[i, j, k],
            ssrel_rank_matrix[i, j, k], mf_rank_matrix[i, j, k],
            stdev_rank_matrix[i, j, k]
        ))


def _run(experiment_dir_name, matching_distance_px, output_dir_name):
    """Plots evaluation scores vs. hyperparameters for UQ Experiment 1.

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
    ssrel_matrix = numpy.full(dimensions, numpy.nan)
    mean_predictive_stdev_matrix = numpy.full(dimensions, numpy.nan)
    monotonicity_fraction_matrix = numpy.full(dimensions, numpy.nan)

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

                this_score_file_name = (
                    '{0:s}/top-level-skip-dropout={1:.3f}_'
                    'penultimate-layer-dropout={2:.3f}_'
                    'output-layer-dropout={3:.3f}/'
                    'validation_with_dropout/partial_grids/evaluation/'
                    'spread_vs_skill_matching-distance-px=0.000000.nc'
                ).format(
                    experiment_dir_name, TOP_LEVEL_SKIP_DROPOUT_RATES[i],
                    PENULTIMATE_LAYER_DROPOUT_RATES[j],
                    OUTPUT_LAYER_DROPOUT_RATES[k]
                )

                print('Reading data from: "{0:s}"...'.format(
                    this_score_file_name
                ))
                result_dict = uq_evaluation.read_spread_vs_skill(
                    this_score_file_name
                )

                ssrel_matrix[i, j, k] = result_dict[
                    uq_evaluation.SPREAD_SKILL_QUALITY_SCORE_KEY
                ]

                non_zero_indices = numpy.where(
                    result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY] > 0
                )[0]
                mean_predictive_stdev_matrix[i, j, k] = numpy.average(
                    result_dict[uq_evaluation.MEAN_PREDICTION_STDEVS_KEY][
                        non_zero_indices
                    ],
                    weights=
                    result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY][
                        non_zero_indices
                    ]
                )

                this_score_file_name = (
                    '{0:s}/top-level-skip-dropout={1:.3f}_'
                    'penultimate-layer-dropout={2:.3f}_'
                    'output-layer-dropout={3:.3f}/'
                    'validation_with_dropout/partial_grids/evaluation/xentropy/'
                    'discard_test_matching-distance-px=0.000000.nc'
                ).format(
                    experiment_dir_name, TOP_LEVEL_SKIP_DROPOUT_RATES[i],
                    PENULTIMATE_LAYER_DROPOUT_RATES[j],
                    OUTPUT_LAYER_DROPOUT_RATES[k]
                )

                print('Reading data from: "{0:s}"...'.format(
                    this_score_file_name
                ))
                monotonicity_fraction_matrix[i, j, k] = (
                    uq_evaluation.read_discard_results(this_score_file_name)[
                        uq_evaluation.MONOTONICITY_FRACTION_KEY
                    ]
                )

    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        aupd_matrix=aupd_matrix, csi_matrix=max_csi_matrix,
        fss_matrix=fss_matrix, bss_matrix=bss_matrix, ssrel_matrix=ssrel_matrix,
        mean_predictive_stdev_matrix=mean_predictive_stdev_matrix,
        monotonicity_fraction_matrix=monotonicity_fraction_matrix,
        rank_mainly_by_fss=True
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        aupd_matrix=aupd_matrix, csi_matrix=max_csi_matrix,
        fss_matrix=fss_matrix, bss_matrix=bss_matrix, ssrel_matrix=ssrel_matrix,
        mean_predictive_stdev_matrix=mean_predictive_stdev_matrix,
        monotonicity_fraction_matrix=monotonicity_fraction_matrix,
        rank_mainly_by_fss=False
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=aupd_matrix, score_name='AUPD')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=max_csi_matrix, score_name='max CSI')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=fss_matrix, score_name='FSS')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(score_matrix=bss_matrix, score_name='BSS')
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=ssrel_matrix, score_name='negative SSREL'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=mean_predictive_stdev_matrix,
        score_name='mean predictive stdev'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=monotonicity_fraction_matrix,
        score_name='monotonicity fraction'
    )
    print(SEPARATOR_STRING)

    aupd_panel_file_names = [''] * num_output_layer_rates
    csi_panel_file_names = [''] * num_output_layer_rates
    fss_panel_file_names = [''] * num_output_layer_rates
    bss_panel_file_names = [''] * num_output_layer_rates
    ssrel_panel_file_names = [''] * num_output_layer_rates
    stdev_panel_file_names = [''] * num_output_layer_rates
    mf_panel_file_names = [''] * num_output_layer_rates
    letter_label = None

    for k in range(num_output_layer_rates):
        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        # Plot AUPD.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=aupd_matrix[..., k],
            min_colour_value=numpy.nanpercentile(aupd_matrix, 1),
            max_colour_value=numpy.nanpercentile(aupd_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        title_string = 'AUPD; dropout rate for last layer = {0:.3f}'.format(
            OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        aupd_panel_file_names[k] = (
            '{0:s}/aupd_output-layer-dropout={1:.3f}.jpg'
        ).format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(aupd_panel_file_names[k]))
        figure_object.savefig(
            aupd_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=aupd_panel_file_names[k],
            output_file_name=aupd_panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

        # Plot max CSI.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=max_csi_matrix[..., k],
            min_colour_value=numpy.nanpercentile(max_csi_matrix, 1),
            max_colour_value=numpy.nanpercentile(max_csi_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        title_string = 'Max CSI; dropout rate for last layer = {0:.3f}'.format(
            OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        csi_panel_file_names[k] = (
            '{0:s}/csi_output-layer-dropout={1:.3f}.jpg'
        ).format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(csi_panel_file_names[k]))
        figure_object.savefig(
            csi_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=csi_panel_file_names[k],
            output_file_name=csi_panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

        # Plot FSS.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=fss_matrix[..., k],
            min_colour_value=numpy.nanpercentile(fss_matrix, 1),
            max_colour_value=numpy.nanpercentile(fss_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        title_string = 'FSS; dropout rate for last layer = {0:.3f}'.format(
            OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        fss_panel_file_names[k] = (
            '{0:s}/fss_output-layer-dropout={1:.3f}.jpg'
        ).format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(fss_panel_file_names[k]))
        figure_object.savefig(
            fss_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=fss_panel_file_names[k],
            output_file_name=fss_panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

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

        title_string = 'BSS; dropout rate for last layer = {0:.3f}'.format(
            OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        bss_panel_file_names[k] = (
            '{0:s}/bss_output-layer-dropout={1:.3f}.jpg'
        ).format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(bss_panel_file_names[k]))
        figure_object.savefig(
            bss_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=bss_panel_file_names[k],
            output_file_name=bss_panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

        # Plot SSREL.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=ssrel_matrix[..., k],
            min_colour_value=numpy.nanpercentile(ssrel_matrix, 1),
            max_colour_value=numpy.nanpercentile(ssrel_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        title_string = 'SSREL; dropout rate for last layer = {0:.3f}'.format(
            OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        ssrel_panel_file_names[k] = (
            '{0:s}/ssrel_output-layer-dropout={1:.3f}.jpg'
        ).format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(ssrel_panel_file_names[k]))
        figure_object.savefig(
            ssrel_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=ssrel_panel_file_names[k],
            output_file_name=ssrel_panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

        # Plot mean predictive stdev.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=mean_predictive_stdev_matrix[..., k],
            min_colour_value=
            numpy.nanpercentile(mean_predictive_stdev_matrix, 1),
            max_colour_value=
            numpy.nanpercentile(mean_predictive_stdev_matrix, 99),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        title_string = (
            'Mean predictive stdev;\ndropout rate for last layer = {0:.3f}'
        ).format(
            OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        stdev_panel_file_names[k] = (
            '{0:s}/mean_predictive_stdev_output-layer-dropout={1:.3f}.jpg'
        ).format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(stdev_panel_file_names[k]))
        figure_object.savefig(
            stdev_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=stdev_panel_file_names[k],
            output_file_name=stdev_panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

        # Plot monotonicity fraction.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=monotonicity_fraction_matrix[..., k],
            min_colour_value=numpy.nanmin(monotonicity_fraction_matrix),
            max_colour_value=numpy.nanmax(monotonicity_fraction_matrix),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
            colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
        )

        title_string = (
            'Monotonicity fraction;\ndropout rate for last layer = {0:.3f}'
        ).format(
            OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(title_string)
        gg_plotting_utils.label_axes(
            axes_object=axes_object,
            label_string='({0:s})'.format(letter_label)
        )

        mf_panel_file_names[k] = (
            '{0:s}/monotonicity_fraction_output-layer-dropout={1:.3f}.jpg'
        ).format(
            output_dir_name, OUTPUT_LAYER_DROPOUT_RATES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(mf_panel_file_names[k]))
        figure_object.savefig(
            mf_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.resize_image(
            input_file_name=mf_panel_file_names[k],
            output_file_name=mf_panel_file_names[k],
            output_size_pixels=PANEL_SIZE_PX
        )

    num_panel_columns = int(numpy.floor(
        numpy.sqrt(num_output_layer_rates)
    ))
    num_panel_rows = int(numpy.ceil(
        float(num_output_layer_rates) / num_panel_columns
    ))

    aupd_concat_file_name = '{0:s}/aupd.jpg'.format(output_dir_name)
    print('Concatenating figures to: "{0:s}"...'.format(aupd_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=aupd_panel_file_names,
        output_file_name=aupd_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=aupd_concat_file_name,
        output_file_name=aupd_concat_file_name
    )

    csi_concat_file_name = '{0:s}/csi.jpg'.format(output_dir_name)
    print('Concatenating figures to: "{0:s}"...'.format(csi_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=csi_panel_file_names,
        output_file_name=csi_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=csi_concat_file_name,
        output_file_name=csi_concat_file_name
    )

    fss_concat_file_name = '{0:s}/fss.jpg'.format(output_dir_name)
    print('Concatenating figures to: "{0:s}"...'.format(fss_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=fss_panel_file_names,
        output_file_name=fss_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=fss_concat_file_name,
        output_file_name=fss_concat_file_name
    )

    bss_concat_file_name = '{0:s}/bss.jpg'.format(output_dir_name)
    print('Concatenating figures to: "{0:s}"...'.format(bss_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=bss_panel_file_names,
        output_file_name=bss_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=bss_concat_file_name,
        output_file_name=bss_concat_file_name
    )

    ssrel_concat_file_name = '{0:s}/ssrel.jpg'.format(output_dir_name)
    print('Concatenating figures to: "{0:s}"...'.format(ssrel_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=ssrel_panel_file_names,
        output_file_name=ssrel_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=ssrel_concat_file_name,
        output_file_name=ssrel_concat_file_name
    )

    stdev_concat_file_name = '{0:s}/mean_predictive_stdev.jpg'.format(
        output_dir_name
    )
    print('Concatenating figures to: "{0:s}"...'.format(stdev_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=stdev_panel_file_names,
        output_file_name=stdev_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=stdev_concat_file_name,
        output_file_name=stdev_concat_file_name
    )

    mf_concat_file_name = '{0:s}/monotonicity_fraction.jpg'.format(
        output_dir_name
    )
    print('Concatenating figures to: "{0:s}"...'.format(mf_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=mf_panel_file_names,
        output_file_name=mf_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=mf_concat_file_name,
        output_file_name=mf_concat_file_name
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
