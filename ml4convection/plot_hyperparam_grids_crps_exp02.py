"""Plots evaluation metrics vs. hyperparameters for CRPS Experiment 2."""

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
import neural_net
import gg_model_evaluation as gg_model_eval
import gg_plotting_utils
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NEIGH_HALF_WINDOW_SIZES_PX = numpy.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=int)
CRPS_WEIGHTS = numpy.logspace(0, 2, num=9)

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

    if numpy.isnan(min_colour_value) or numpy.isnan(max_colour_value):
        min_colour_value = 0.
        max_colour_value = 1.

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

    N = number of FSS neighbourhood sizes
    W = number of CRPS weights

    :param score_matrix: N-by-W numpy array of scores.
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
            '{0:d}th-highest {1:s} = {2:.4g} ... FSS neighbourhood size = '
            '{3:d} x {3:d} ... CRPS weight = 10^{4:.2f}'
        ).format(
            k + 1, score_name, score_matrix[i, j],
            int(numpy.round(NEIGH_HALF_WINDOW_SIZES_PX[i])),
            numpy.log10(CRPS_WEIGHTS[j])
        ))


def _print_ranking_all_scores(
        aupd_matrix, csi_matrix, fss_matrix, bss_matrix, ssrel_matrix,
        mean_predictive_stdev_matrix, monotonicity_fraction_matrix,
        rank_mainly_by_fss):
    """Prints ranking for all scores.

    N = number of FSS neighbourhood sizes
    W = number of CRPS weights

    :param aupd_matrix: N-by-W numpy array with AUPD (area under performance
        diagram).
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
    i_sort_indices, j_sort_indices = numpy.unravel_index(
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

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        j = j_sort_indices[k]

        print((
            '{0:d}th-best model ... FSS neighbourhood size = {1:d} x {1:d} ... '
            'CRPS weight = 10^{2:.2f} ... '
            'AUPD rank = {3:.1f} ... CSI rank = {4:.1f} ... '
            'FSS rank = {5:.1f} ... BSS rank = {6:.1f} ... '
            'SSREL rank = {7:.1f} ... MF rank = {8:.1f} ... '
            'predictive-stdev rank = {9:.1f}'
        ).format(
            k + 1, int(numpy.round(NEIGH_HALF_WINDOW_SIZES_PX[i])),
            numpy.log10(CRPS_WEIGHTS[j]),
            aupd_rank_matrix[i, j], csi_rank_matrix[i, j],
            fss_rank_matrix[i, j], bss_rank_matrix[i, j],
            ssrel_rank_matrix[i, j], mf_rank_matrix[i, j],
            stdev_rank_matrix[i, j]
        ))


def _run(experiment_dir_name, matching_distance_px, output_dir_name):
    """Plots evaluation metrics vs. hyperparameters for CRPS Experiment 2.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param matching_distance_px: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_neigh_sizes = len(NEIGH_HALF_WINDOW_SIZES_PX)
    num_crps_weights = len(CRPS_WEIGHTS)
    dimensions = (num_neigh_sizes, num_crps_weights)

    aupd_matrix = numpy.full(dimensions, numpy.nan)
    max_csi_matrix = numpy.full(dimensions, numpy.nan)
    fss_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)
    ssrel_matrix = numpy.full(dimensions, numpy.nan)
    mean_predictive_stdev_matrix = numpy.full(dimensions, numpy.nan)
    monotonicity_fraction_matrix = numpy.full(dimensions, numpy.nan)

    y_tick_labels = [
        '{0:d} x {0:d}'.format(int(numpy.round(n)))
        for n in NEIGH_HALF_WINDOW_SIZES_PX
    ]
    x_tick_labels = ['{0:.2f}'.format(numpy.log10(w)) for w in CRPS_WEIGHTS]
    x_tick_labels = [r'10$^{' + l + r'}$' for l in x_tick_labels]

    y_axis_label = 'FSS neighbourhood size'
    x_axis_label = 'CRPS weight'

    for i in range(num_neigh_sizes):
        for j in range(num_crps_weights):
            this_loss_function_name = neural_net.metric_params_to_name(
                score_name=neural_net.FSS_PLUS_PIXELWISE_CRPS_NAME,
                half_window_size_px=NEIGH_HALF_WINDOW_SIZES_PX[i],
                crps_weight=CRPS_WEIGHTS[j]
            )

            this_score_file_name = (
                '{0:s}/{1:s}/validation/partial_grids/evaluation/'
                'matching_distance_px={2:.6f}/advanced_scores_gridded=0.p'
            ).format(
                experiment_dir_name, this_loss_function_name,
                matching_distance_px
            )

            if os.path.isfile(this_score_file_name):
                print('Reading data from: "{0:s}"...'.format(
                    this_score_file_name
                ))
                t = evaluation.read_advanced_score_file(this_score_file_name)

                try:
                    aupd_matrix[i, j] = (
                        gg_model_eval.get_area_under_perf_diagram(
                            pod_by_threshold=numpy.mean(
                                t[evaluation.POD_KEY].values, axis=0
                            ),
                            success_ratio_by_threshold=numpy.mean(
                                t[evaluation.SUCCESS_RATIO_KEY].values, axis=0
                            )
                        )
                    )
                except:
                    pass

                max_csi_matrix[i, j] = numpy.nanmax(
                    numpy.mean(t[evaluation.CSI_KEY].values, axis=0)
                )
                fss_matrix[i, j] = numpy.mean(t[evaluation.FSS_KEY].values)
                bss_matrix[i, j] = numpy.mean(
                    t[evaluation.BRIER_SKILL_SCORE_KEY].values
                )

            this_score_file_name = (
                '{0:s}/{1:s}/validation/partial_grids/evaluation/'
                'spread_vs_skill_matching-distance-px=0.000000.nc'
            ).format(
                experiment_dir_name, this_loss_function_name
            )

            if os.path.isfile(this_score_file_name):
                print('Reading data from: "{0:s}"...'.format(
                    this_score_file_name
                ))
                result_dict = uq_evaluation.read_spread_vs_skill(
                    this_score_file_name
                )

                ssrel_matrix[i, j] = result_dict[
                    uq_evaluation.SPREAD_SKILL_QUALITY_SCORE_KEY
                ]

                non_zero_indices = numpy.where(
                    result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY] > 0
                )[0]
                mean_predictive_stdev_matrix[i, j] = numpy.average(
                    result_dict[uq_evaluation.MEAN_PREDICTION_STDEVS_KEY][
                        non_zero_indices
                    ],
                    weights=
                    result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY][
                        non_zero_indices
                    ]
                )

            this_score_file_name = (
                '{0:s}/{1:s}/validation/partial_grids/evaluation/'
                'discard_test_matching-distance-px=0.000000.nc'
            ).format(
                experiment_dir_name, this_loss_function_name
            )

            if os.path.isfile(this_score_file_name):
                print('Reading data from: "{0:s}"...'.format(
                    this_score_file_name
                ))
                monotonicity_fraction_matrix[i, j] = (
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
        score_matrix=-ssrel_matrix, score_name='negative SSREL'
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

    # Plot AUPD.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=aupd_matrix,
        min_colour_value=numpy.nanpercentile(aupd_matrix, 1),
        max_colour_value=numpy.nanpercentile(aupd_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Area under performance diagram')
    figure_file_name = '{0:s}/aupd.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot max CSI.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=max_csi_matrix,
        min_colour_value=numpy.nanpercentile(max_csi_matrix, 1),
        max_colour_value=numpy.nanpercentile(max_csi_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Max critical success index')
    figure_file_name = '{0:s}/csi.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot FSS.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=fss_matrix,
        min_colour_value=numpy.nanpercentile(fss_matrix, 1),
        max_colour_value=numpy.nanpercentile(fss_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Fractions skill score')
    figure_file_name = '{0:s}/fss.jpg'.format(output_dir_name)

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
        score_matrix=bss_matrix,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=BSS_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Brier skill score')
    figure_file_name = '{0:s}/bss.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot SSREL.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=ssrel_matrix,
        min_colour_value=numpy.nanpercentile(ssrel_matrix, 1),
        max_colour_value=numpy.nanpercentile(ssrel_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Spread-skill reliability')
    figure_file_name = '{0:s}/ssrel.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot mean predictive stdev.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=mean_predictive_stdev_matrix,
        min_colour_value=numpy.nanpercentile(mean_predictive_stdev_matrix, 1),
        max_colour_value=numpy.nanpercentile(mean_predictive_stdev_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Mean stdev of predictive distribution')
    figure_file_name = '{0:s}/mean_predictive_stdev.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot monotonicity fraction.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=monotonicity_fraction_matrix,
        min_colour_value=numpy.nanpercentile(monotonicity_fraction_matrix, 1),
        max_colour_value=numpy.nanpercentile(monotonicity_fraction_matrix, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Monotonicity fraction from discard test')
    figure_file_name = '{0:s}/monotonicity_fraction.jpg'.format(output_dir_name)

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
