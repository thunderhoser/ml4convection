"""Plots evaluation scores vs. hyperparameters for qFSS Experiment 2."""

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
import uq_evaluation
import gg_model_evaluation as gg_model_eval
import gg_plotting_utils
import imagemagick_utils
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

QUANTILE_LEVEL_SETS = [
    numpy.linspace(0.01, 0.99, num=99),
    numpy.linspace(0.01, 0.99, num=99)[::2],
    numpy.linspace(0.01, 0.99, num=99)[::3],
    numpy.linspace(0.01, 0.99, num=99)[::4],
    numpy.linspace(0.01, 0.99, num=99)[::5],
    numpy.linspace(0.01, 0.99, num=99)[::6],
    numpy.linspace(0.01, 0.99, num=99)[::7],
    numpy.linspace(0.01, 0.99, num=99)[::8],
    numpy.linspace(0.01, 0.99, num=99)[::9]
]

QUANTILE_LEVEL_SETS = [
    numpy.concatenate((
        numpy.array([0.025, 0.25, 0.5, 0.75, 0.975, 0.99]), s
    ))
    for s in QUANTILE_LEVEL_SETS
]

QUANTILE_LEVEL_SETS = [numpy.sort(numpy.unique(s)) for s in QUANTILE_LEVEL_SETS]

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
        fraction_of_axis_length=1., font_size=DEFAULT_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    M = number of models

    :param score_matrix: length-M numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix) + 0.
    scores_1d[numpy.isnan(scores_1d)] = -numpy.inf
    sort_indices_1d = numpy.argsort(-scores_1d)
    i_sort_indices, _ = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]

        print((
            '{0:d}th-highest {1:s} = {2:.4g} ... num quantile levels = {3:d}'
        ).format(
            m + 1, score_name, score_matrix[i, 0], len(QUANTILE_LEVEL_SETS[i])
        ))


def _run(experiment_dir_name, matching_distance_px, output_dir_name):
    """Plots evaluation scores vs. hyperparameters for qFSS Experiment 2.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param matching_distance_px: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    dimensions = (len(QUANTILE_LEVEL_SETS), 1)

    aupd_matrix = numpy.full(dimensions, numpy.nan)
    max_csi_matrix = numpy.full(dimensions, numpy.nan)
    fss_matrix = numpy.full(dimensions, numpy.nan)
    bss_matrix = numpy.full(dimensions, numpy.nan)
    ssrel_matrix = numpy.full(dimensions, numpy.nan)
    mean_predictive_stdev_matrix = numpy.full(dimensions, numpy.nan)
    monotonicity_fraction_matrix = numpy.full(dimensions, numpy.nan)

    y_tick_labels = ['{0:d}'.format(len(s)) for s in QUANTILE_LEVEL_SETS]
    x_tick_labels = ['']

    y_axis_label = 'Number of quantiles'
    x_axis_label = ''

    for i in range(len(QUANTILE_LEVEL_SETS)):
        # TODO(thunderhoser): Allow option of ignoring quantiles.
        
        this_score_file_name = (
            '{0:s}/num-quantile-levels={1:03d}/'
            'validation_sans_uq/partial_grids/evaluation/'
            'matching_distance_px={2:.6f}/'
            'advanced_scores_gridded=0.p'
        ).format(
            experiment_dir_name, len(QUANTILE_LEVEL_SETS[i]),
            matching_distance_px
        )

        print('Reading data from: "{0:s}"...'.format(
            this_score_file_name
        ))
        t = evaluation.read_advanced_score_file(this_score_file_name)

        aupd_matrix[i, 0] = (
            gg_model_eval.get_area_under_perf_diagram(
                pod_by_threshold=numpy.mean(
                    t[evaluation.POD_KEY].values, axis=0
                ),
                success_ratio_by_threshold=numpy.mean(
                    t[evaluation.SUCCESS_RATIO_KEY].values, axis=0
                )
            )
        )

        max_csi_matrix[i, 0] = numpy.nanmax(
            numpy.mean(t[evaluation.CSI_KEY].values, axis=0)
        )
        fss_matrix[i, 0] = numpy.mean(t[evaluation.FSS_KEY].values)
        bss_matrix[i, 0] = numpy.mean(
            t[evaluation.BRIER_SKILL_SCORE_KEY].values
        )

        this_score_file_name = (
            '{0:s}/num-quantile-levels={1:03d}/'
            'validation_with_uq/partial_grids/evaluation/'
            'spread_vs_skill_matching-distance-px=0.000000.nc'
        ).format(
            experiment_dir_name, len(QUANTILE_LEVEL_SETS[i])
        )

        print('Reading data from: "{0:s}"...'.format(
            this_score_file_name
        ))
        result_dict = uq_evaluation.read_spread_vs_skill(
            this_score_file_name
        )

        ssrel_matrix[i, 0] = result_dict[
            uq_evaluation.SPREAD_SKILL_QUALITY_SCORE_KEY
        ]

        non_zero_indices = numpy.where(
            result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY] > 0
        )[0]
        mean_predictive_stdev_matrix[i, 0] = numpy.average(
            result_dict[uq_evaluation.MEAN_PREDICTION_STDEVS_KEY][
                non_zero_indices
            ],
            weights=
            result_dict[uq_evaluation.EXAMPLE_COUNTS_KEY][
                non_zero_indices
            ]
        )

        this_score_file_name = (
            '{0:s}/num-quantile-levels={1:03d}/'
            'validation_with_uq/partial_grids/evaluation/'
            'discard_test_matching-distance-px=0.000000.nc'
        ).format(
            experiment_dir_name, len(QUANTILE_LEVEL_SETS[i])
        )

        print('Reading data from: "{0:s}"...'.format(
            this_score_file_name
        ))
        monotonicity_fraction_matrix[i, 0] = (
            uq_evaluation.read_discard_results(this_score_file_name)[
                uq_evaluation.MONOTONICITY_FRACTION_KEY
            ]
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
    axes_object.set_title('AUPD')

    this_file_name = '{0:s}/aupd.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
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
    axes_object.set_title('CSI')

    this_file_name = '{0:s}/csi.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
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
    axes_object.set_title('FSS')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    this_file_name = '{0:s}/fss.jpg'.format(output_dir_name)
    panel_file_names = [this_file_name]

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
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
    axes_object.set_title('BSS')

    this_file_name = '{0:s}/bss.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
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
    axes_object.set_title('SSREL')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    this_file_name = '{0:s}/ssrel.jpg'.format(output_dir_name)
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
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
    axes_object.set_title('MF')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')

    this_file_name = '{0:s}/monotonicity_fraction.jpg'.format(output_dir_name)
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
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
    axes_object.set_title('Mean stdev')

    this_file_name = '{0:s}/mean_predictive_stdev.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Concatenate.
    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=this_file_name, output_file_name=this_file_name,
            border_width_pixels=50
        )

    concat_file_name = '{0:s}/concat.jpg'.format(output_dir_name)
    print('Concatenating figures to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=1, num_panel_columns=3
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_file_name,
        output_file_name=concat_file_name
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
